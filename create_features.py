import sys
import os
import pandas as pd
import nltk
import multiprocessing
from multiprocessing import Pool
from sentimentmodule import init_sentiment_analyzers, sentiment2
os.environ["OMP_NUM_THREADS"] = "6" # use 6 cores
def main(input_csv, output_csv):

    # Read the parquet file
    data = pd.read_parquet(input_csv)

    data['subreddit'].value_counts()

    # Ensure 'body' is of type string for all_activity data
    #data['body'] = data['body'].astype(str)

    # for posts data
    data['text'] = (data['title'] + ' ' + data['selftext'].fillna('')).astype(str)

    # ?
    # data['author'] = data['author'].str.strip()

    # Prepare texts for sentiment analysis
    texts = data['text'].tolist()
    subreddits = data['subreddit'].tolist()

    # Combine texts and subreddits into a list of tuples
    input_data = list(zip(texts, subreddits))

    # Initialize multiprocessing pool with 'spawn' to avoid tokenizer warnings
    multiprocessing.set_start_method('spawn', force=True)

    with Pool(processes=multiprocessing.cpu_count(), initializer=init_sentiment_analyzers) as pool:
        results = pool.map(sentiment2, input_data)

    # Convert results to DataFrame
    # Joining the two dataframes so I have sentiment analysis with comments and other columns
    sentiments_df = pd.DataFrame(results, columns=['sentiment', 'politician_mentioned','shifted'])
    data = pd.concat([data, sentiments_df], axis=1)

    # Save the DataFrame with sentiment per comment
    data.to_csv(output_csv, index=False)
    print(f"Sentiment analysis completed. Output saved to {output_csv}")

if __name__ == '__main__':
    input_csv = 'datafiles/allsubsmall2.parquet'
    output_csv = 'non_group_sent.csv'
    main(input_csv,output_csv)