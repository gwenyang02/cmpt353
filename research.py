import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit user tracker for political subreddits').getOrCreate()

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
    
    # Filter comments from specified subreddits
    election_comments = reddit_comments \
        .where(reddit_comments['subreddit'].isin(election_subs)) \
        .select('author', 'id', 'body', 'subreddit', 'subreddit_id', 'year', 'month')
    
    # Save filtered election-related comments
    election_comments.write.json(output + '/election_comments', mode='overwrite', compression='gzip')
    
    # Filter submissions from specified subreddits
    election_submissions = reddit_submissions \
        .where(reddit_submissions['subreddit'].isin(election_subs)) \
        .select('author', 'id', 'title', 'selftext', 'subreddit', 'subreddit_id', 'year', 'month', 'created')
    
    # Save filtered election-related submissions
    election_submissions.write.json(output + '/election_submissions', mode='overwrite', compression='gzip')


    #based on distinct userid in election subreddits, 
    # grab comments and submissions from all_comments by same userid

    #TO BE FILLED IN

    # # Save all comments for broader analysis
    # all_comments = reddit_comments.select('author', 'id', 'body', 'subreddit', 'subreddit_id', 'year', 'month')
    # all_comments.write.json(output + '/all_comments', mode='overwrite', compression='gzip')

    # # Save all submissions for broader analysis
    # all_submissions = reddit_submissions.select('author', 'id', 'title', 'selftext', 'subreddit', 'subreddit_id', 'year', 'month', 'created')
    # all_submissions.write.json(output + '/all_submissions', mode='overwrite', compression='gzip')

main()
