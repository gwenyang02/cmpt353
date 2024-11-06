import sys
from pyspark.sql import SparkSession, functions, types, DataFrame
from pyspark.sql.functions import broadcast, col, lit


spark = SparkSession.builder.appName('reddit user tracker for political subreddits').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 10)  # make sure we have Python 3.10+
assert spark.version >= '3.5'  # make sure we have Spark 3.5+

def load_filtered_data(path: str, schema: types.StructType, subreddits: list, years: list) -> DataFrame:
   # Filter during load
   return spark.read.json(path, schema=schema) \
           .where(col('subreddit').isin(subreddits)) \
           .where(col('year').isin(years))

def main():
   # Define paths
   reddit_submissions_path = '/courses/datasets/reddit_submissions_repartitioned/'
   reddit_comments_path = '/courses/datasets/reddit_comments_repartitioned/'
   output = 'reddit-us-election-expanded'

   # Define schema for comments data
   comments_schema = types.StructType([
       types.StructField('author', types.StringType()),  #username
       types.StructField('id', types.StringType()),  # Comment ID 
       types.StructField('body', types.StringType()),  # Comment text
       types.StructField('subreddit', types.StringType()),  # Subreddit name
       types.StructField('subreddit_id', types.StringType()),  # Subreddit ID
       types.StructField('year', types.IntegerType()),  # Year of the comment
       types.StructField('month', types.IntegerType()), #comment month
   ])

   # Define schema for submissions data
   submissions_schema = types.StructType([
       types.StructField('author', types.StringType()),  # Reddit username
       types.StructField('id', types.StringType()),  # Submission ID
       types.StructField('title', types.StringType()),  # Title of the submission
       types.StructField('selftext', types.StringType()),  # Submission text (if it's a self-post)
       types.StructField('subreddit', types.StringType()),  # Subreddit name
       types.StructField('subreddit_id', types.StringType()),  # Subreddit ID
       types.StructField('year', types.IntegerType()),  # Year of the submission
       types.StructField('month', types.IntegerType()), #month of submission
       types.StructField('created', types.LongType()),  # Timestamp of submission creation
   ])
   
   # List of subreddits related to US presidential elections and political discussions
   election_subs = [
       'presidential_election', 'politics', 'uspolitics', 'election2016', 'democrats', 'republicans',
   ]
   
   # consider filtering years for smaller dataset
   filter_years = [2016]
   # Load filtered submissions and comments data directly
   reddit_submissions = load_filtered_data(reddit_submissions_path, submissions_schema, election_subs, filter_years).limit(100)
   reddit_comments = load_filtered_data(reddit_comments_path, comments_schema, election_subs, filter_years).limit(100)

   #writing comments and submissions to json
   reddit_comments.write.json(output + '/election_comments', mode='overwrite', compression='gzip')
   reddit_submissions.write.json(output + '/election_submissions', mode='overwrite', compression='gzip')
   
   # based on distinct author in election subreddits,
   # grab comments and submissions from an author with activity in election subreddits
   # source: https://sparkbyexamples.com/pyspark/pyspark-select-distinct/

   # select distinct authors who make posts and/or comment and extract 'author' column
   # union with author ids in submissions
   # then select distinct
   election_authors = reddit_comments.select("author") \
       .union(reddit_submissions.select("author")) \
       .where(col('author') != '[deleted]') \
       .distinct() \
       .cache()

   #print('election authors')
   #election_authors.show()

    # Join and label comments and submissions using column of 'type' to distinguish 
   all_comments = reddit_comments.select('author', 'id', 'body', 'subreddit', 'subreddit_id', 'year', 'month') \
              .join(broadcast(election_authors), on='author') \
              .withColumn("type", lit("comment")) \
              .withColumn("title", lit(None).cast("string")) \
              .withColumn("selftext", lit(None).cast("string")) \
              .withColumn("created", lit(None).cast("long"))

   all_submissions = reddit_submissions.select('author', 'id', 'title', 'selftext', 'subreddit', 'subreddit_id',
                                           'year', 'month', 'created') \
                  .join(broadcast(election_authors), on='author') \
                  .withColumn("type", lit("submission")) \
                  .withColumn("body", lit(None).cast("string"))
                  

   # Combine comments and submissions into a single DataFrame
   all_activity = all_comments.unionByName(all_submissions)

   # Write the combined DataFrame to a single output location
   all_activity.write.json(output + '/all_activity', mode='overwrite', compression='gzip')

if __name__ == '__main__':
   # spark configs
   spark.conf.set("spark.sql.adaptive.enabled", 'True')
   spark.conf.set("spark.sql.adaptive.skewJoin.enabled", 'True')
   
   main()

