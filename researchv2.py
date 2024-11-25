import sys
from pyspark.sql import SparkSession, functions, types, DataFrame, Window
from pyspark.sql.functions import broadcast, col, lit, concat, row_number, rand


spark = SparkSession.builder.appName('reddit user tracker for political subreddits').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 10)  # make sure we have Python 3.10+
assert spark.version >= '3.5'  # make sure we have Spark 3.5+

def load_filtered_data(path: str, schema: types.StructType, subreddits: list, year: int) -> DataFrame:
   '''
    load data and filter dataframe
   :param path: string representing path to reddit submissions or comments
   :param schema: a struct type representing the schema for each type of reddit data
   :param subreddits: a list of strings representing the subreddits we want to filter to
   :param years: a list of ints representing the years we want to filter to
   :return: a dataframe for either reddit submissions or comments
   '''
   df = spark.read.json(path, schema=schema) \
           .where(col('subreddit').isin(subreddits)) \
           .where(col('year')==year) \
           .where(col('month').isin([1,3,6,9,12]))

	# Check if the 'body' column exists => if it is it follows comment type schema
	# filter out the comments that are empty or deleted
   if 'body' in df.columns:
        df = df.filter(~col('body').rlike('\[deleted\]|\[removed\]'))

   # selecting a stratified sample
   # of 1200 per subreddit in random order
   # source: https://stackoverflow.com/questions/41516805/
   df_sampled = df.withColumn("row_num",row_number().over(Window.partitionBy("subreddit").orderBy(rand()))) \
                  .where(col("row_num") <= 2000) \
                  .drop("row_num")

   return df_sampled

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
       # timestamp (time of day) for comment creation that needs to be converted
       # https://stackoverflow.com/questions/16801162/
       types.StructField('created_utc', types.StringType()),
       types.StructField('score', types.LongType()),  # score (number of upvotes)

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
       types.StructField('created_utc', types.StringType()),  # Timestamp of submission creation
       types.StructField('score', types.LongType()),  # score (number of upvotes)
   ])
   
   # List of subreddits related to US presidential elections and political discussions
   # r/immigration: no anti-immigration sentiment allowed
   # r/guns: leans conservative
   # r/NeutralPolitics: has political discussions
   election_subs = [
       'politics', 'uspolitics', 'democrats', 'Republican',
       'immigration', 'Libertarian','Conservative','Economics','PoliticalDiscussion',
       'Liberal','guns','NeutralPolitics'
   ]
   
   # filter year range to 2016
   filter_years = 2016

   # Load filtered submissions and comments data directly
   reddit_submissions = load_filtered_data(reddit_submissions_path, submissions_schema, election_subs, filter_years) \
                        .cache()

   reddit_comments = load_filtered_data(reddit_comments_path, comments_schema, election_subs, filter_years) \
                     .cache()
   # number of rows = n rows per subreddit * 12 subreddits = 2000*12 = 24000
   print(f"Rows in comments: {reddit_comments.count()}")
   print(f"Rows in submissions: {reddit_submissions.count()}")

   # writing comments and submissions to json
   #reddit_comments.write.json(output + '/election_comments', mode='overwrite', compression='gzip')
   #reddit_submissions.write.json(output + '/election_submissions', mode='overwrite', compression='gzip')

   # writing comments and submissions to parquet
   # it is safe to .coalesce(3) because 24000 rows/3 = 8000 rows per partition
   reddit_comments.coalesce(3).write.parquet(output + '/election_comments', mode='overwrite')
   reddit_submissions.coalesce(3).write.parquet(output + '/election_submissions', mode='overwrite')

   # based on distinct author in election subreddits,
   # grab comments and submissions from an author with activity in election subreddits
   # source: https://sparkbyexamples.com/pyspark/pyspark-select-distinct/

   # select distinct authors who make posts and/or comment and extract 'author' column
   # union with author ids in submissions
   # then select distinct and filter for authors (accounts) that are not [deleted]
   election_authors = reddit_comments.select("author") \
       .union(reddit_submissions.select("author")) \
       .distinct() \
       .where(col('author') != '[deleted]') \
       .cache()

   print("\nUnique authors after union and removing [deleted]:", election_authors.count())
   #print('election authors')
   #election_authors.show()

    # Join and label comments and submissions using column of 'type' to distinguish 
   all_comments = reddit_comments.select('author','id','body','subreddit','subreddit_id',
                                         'created_utc','score','year','month') \
              .join(broadcast(election_authors), on='author') \
              .cache()

   all_submissions = reddit_submissions.select(
    'author', 
    'id', 
    'title', 
    'selftext', 
    'subreddit', 
    'subreddit_id', 
    'year', 
    'month', 
    'created_utc',
    'score',
    # we are treating title and selftext as one comment body
    concat(reddit_submissions['title'], lit(' '), reddit_submissions['selftext']).alias('body')) \
            .join(broadcast(election_authors),on='author') \
            .cache()

                   
   refined_submissions = all_submissions.select('author', 'id', 'body', 'subreddit', 'subreddit_id',
                                                'created_utc','score','year', 'month')
                  

   # Combine comments and submissions into a single DataFrame
   all_activity = all_comments.unionByName(refined_submissions, allowMissingColumns=True)
   all_activity.show(20)
   print(f"Rows in all_activity: {all_activity.count()}")

   # Write the combined DataFrame to a single output location
   # it is safe to .coalesce(5) because the combined all_activity df has upper limit of 48000 rows
   # 48000 rows / 5 partitions = 9600 rows per partition
   all_activity.coalesce(5).write.parquet(output + '/all_activity', mode='overwrite')

if __name__ == '__main__':
   # spark configs
   spark.conf.set("spark.sql.adaptive.enabled", 'True')
   spark.conf.set("spark.sql.adaptive.skewJoin.enabled", 'True')
   
   main()

