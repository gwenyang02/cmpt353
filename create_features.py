# file to create sentiment scores features
# and output .csv containing posts or posts and comments
# with sentiment score column
import sys
import os
import pandas as pd
import nltk
import multiprocessing
from multiprocessing import Pool
from sentimentmodule import init_sentiment_analyzers, sentiment1, sentiment2
os.environ["OMP_NUM_THREADS"] = "4" # use 4 cores

def main(input_csv, output_csv):

    # Read the parquet file
    data = pd.read_csv(input_csv)

    data['subreddit'].value_counts()

    # Ensure 'body' is of type string for all_activity data
    data['body'] = data['body'].astype(str)

    # for posts data
    # data['text'] = (data['title'] + ' ' + data['selftext'].fillna('')).astype(str)

    # Prepare texts for sentiment analysis
    #texts = data['body'].tolist()
    texts = data['text'].tolist()
    subreddits = data['subreddit'].tolist()

    # Combine texts and subreddits into a list of tuples
    input_data = list(zip(texts, subreddits))

    # Initialize multiprocessing pool with 'spawn' to avoid tokenizer warnings
    multiprocessing.set_start_method('spawn', force=True)

    with Pool(processes=multiprocessing.cpu_count(), initializer=init_sentiment_analyzers) as pool:
        results = pool.map(sentiment2, texts)

    #with Pool(processes=multiprocessing.cpu_count(), initializer=init_sentiment_analyzers) as pool:
    #    results = pool.map(sentiment2, input_data)

    # Convert results to DataFrame
    # Joining the two dataframes so I have sentiment analysis with comments and other columns
    sentiments_df = pd.DataFrame(results, columns=['sentiment', 'politician_mentioned', 'shifted'])
    #sentiments_df = pd.DataFrame(results, columns=['sentiment'])
    data = pd.concat([data, sentiments_df], axis=1)

    # Save the DataFrame with sentiment per comment
    data.to_csv(output_csv, index=False)
    print(f"Sentiment analysis completed. Output saved to {output_csv}")

if __name__ == '__main__':
    input_csv = './justcommentsbigcsv.csv'
    output_csv = 'modifiedallcommentswithsentiment.csv'
    main(input_csv,output_csv)