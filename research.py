import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit user tracker for political subreddits').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
# joining dataframes is slow: https://spark.apache.org/docs/latest/sql-performance-tuning.html
spark.conf.set("spark.sql.adaptive.enabled",'True')
spark.conf.set("spark.sql.adaptive.skewJoin.enabled",'True')
assert sys.version_info >= (3,10) # make sure we have Python 3.10+
assert spark.version >= '3.5' # make sure we have Spark 3.5+

# Define paths
reddit_submissions_path = '/courses/datasets/reddit_submissions_repartitioned/'
reddit_comments_path = '/courses/datasets/reddit_comments_repartitioned/'
output = 'reddit-us-election-user-tracking-expanded'

# Define schema for comments data
comments_schema = types.StructType([
    types.StructField('author', types.StringType()),       # Reddit username (identifier for user)
    types.StructField('id', types.StringType()),           # Comment ID (unique identifier for each comment)
    types.StructField('body', types.StringType()),         # Comment text (content of the comment)
    types.StructField('subreddit', types.StringType()),    # Subreddit name
    types.StructField('subreddit_id', types.StringType()), # Subreddit ID
    types.StructField('year', types.IntegerType()),        # Year of the comment
    types.StructField('month', types.IntegerType()),
])

# Define schema for submissions data
submissions_schema = types.StructType([
    types.StructField('author', types.StringType()),       # Reddit username
    types.StructField('id', types.StringType()),           # Submission ID
    types.StructField('title', types.StringType()),        # Title of the submission
    types.StructField('selftext', types.StringType()),     # Submission text (if it's a self-post)
    types.StructField('subreddit', types.StringType()),    # Subreddit name
    types.StructField('subreddit_id', types.StringType()), # Subreddit ID
    types.StructField('year', types.IntegerType()),        # Year of the submission
    types.StructField('month', types.IntegerType()),
    types.StructField('created', types.LongType()),        # Timestamp of submission creation
])

def main():
    # Load submissions and comments data
    reddit_submissions = spark.read.json(reddit_submissions_path, schema=submissions_schema)
    reddit_comments = spark.read.json(reddit_comments_path, schema=comments_schema)
    
    # List of subreddits related to US presidential elections and political discussions
    election_subs = [
        'presidential_election', 'politics', 'PoliticalDiscussion', 'Ask_Politics',
        'uspolitics', 'election2020', 'election2016', 'election2012', 'election2008',
        'conservative', 'democrats', 'republicans', 'libertarian', 'progressive'
    ]
    # consider filtering years for smaller dataset
    #filter_years = list(map(functions.lit, [2016,2020]))
    election_subs = list(map(functions.lit, election_subs))
    
    # Filter comments from specified subreddits
    election_comments = reddit_comments \
        .where(reddit_comments['subreddit'].isin(election_subs)) \
       # .where(reddit_submissions['year'].isin(filter_years)) \
        .select('author', 'id', 'body', 'subreddit', 'subreddit_id', 'year', 'month')
    
    # Save filtered election-related comments
    election_comments.write.json(output + '/election_comments', mode='overwrite', compression='gzip')
    
    # Filter submissions from specified subreddits
    election_submissions = reddit_submissions \
        .where(reddit_submissions['subreddit'].isin(election_subs)) \
       # .where(reddit_submissions['year'].isin(filter_years)) \
        .select('author', 'id', 'title', 'selftext', 'subreddit', 'subreddit_id', 'year', 'month', 'created')
    
    # Save filtered election-related submissions
    election_submissions.write.json(output + '/election_submissions', mode='overwrite', compression='gzip')

    # based on distinct author in election subreddits,
    # grab comments and submissions from an author with activity in election subreddits
    # source: https://sparkbyexamples.com/pyspark/pyspark-select-distinct/

    # select distinct authors who make posts and/or comment and extract 'author' column
    # union with author ids in submissions
    # then select distinct
    election_authors = election_comments.select("author") \
        .union(election_submissions.select("author")) \
        .distinct() \
        .cache()

    # inner-join distinct author dataframe with all comments to filter for comments written by them in other subs
    all_comments = reddit_comments.select('author', 'id', 'body', 'subreddit', 'subreddit_id', 'year', 'month') \
                    #.where(reddit_submissions['year'].isin(filter_years)) \
                    .join(broadcast(election_authors), on='author')

    # filter all submissions in same way as above
    all_submissions = reddit_submissions.select('author', 'id', 'title', 'selftext', 'subreddit', 'subreddit_id',
                                                'year', 'month', 'created') \
                        #.where(reddit_submissions['year'].isin(filter_years)) \
                        .join(broadcast(election_authors), on='author')

    # save all comments as json
    all_comments.write.json(output + '/all_comments', mode='overwrite', compression='gzip')
    # save all submissions (posts) as json
    all_submissions.write.json(output + '/all_submissions', mode='overwrite', compression='gzip')

    # additional join filtering needed to find all comments and submissions for one user together
main()
