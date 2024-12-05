import sys
import os
import pandas as pd
import nltk
import multiprocessing
from multiprocessing import Pool

os.environ["TOKENIZERS_PARALLELISM"] = "true"

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

sia = None
sentiment_pipeline = None

def init_sentiment_analyzers():
    """
    Initializer function for each worker process in the Pool.
    Initializes the SentimentIntensityAnalyzer and the Hugging Face sentiment pipeline.
    """
    global sia, sentiment_pipeline
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from transformers import pipeline

    sia = SentimentIntensityAnalyzer()
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model='distilbert-base-uncased-finetuned-sst-2-english',
        # model="SamLowe/roberta-base-go_emotions"
        device=-1
    )

def calculate_combined_sentiment(inputs):
    """
    Calculates combined sentiment using VADER and Hugging Face's sentiment-analysis pipeline.
    Also checks for mentions of specified politicians.
    """
    # unpack tuple (text, subreddit)
    text, subreddit = inputs

    global sia, sentiment_pipeline

    # Define lists of politicians
    republicans = ['Trump', 'Pence', 'DeSantis', 'McConnell', 'Cruz', 'Rubio']
    democrats = ['Hillary', 'Clinton', 'Biden', 'Harris', 'Pelosi', 'Sanders', 'Schumer']

    # Determine VADER sentiment score
    vader_score = sia.polarity_scores(text)['compound']

    # Determine Hugging Face sentiment label
    hf_result = sentiment_pipeline(text[:512])[0]  # Limit text to 512 tokens
    hf_label = hf_result['label']  

    # Combine VADER and Hugging Face results
    if hf_label == "POSITIVE":
        combined_score = abs(vader_score)
    else:  # NEGATIVE
        combined_score = -abs(vader_score)

    # edge case: check if "Harris" specifically is mentioned with "bipod"
    # this is a type of gun
    harris_mentioned = "harris" in text.lower()
    bipod_mentioned = "bipod" in text.lower()
    # Check for mentions of Republicans or Democrats
    politician_mentioned = False
    mentioned_group = None

    # Check if "Harris" specifically is mentioned without being around "bipod"
    # and if not around "bipod" and in gun-related subreddits
    # assume they are talking about the gun named harris
    if harris_mentioned and bipod_mentioned:
        if subreddit in ['guns', 'guncontrol', 'Firearms']: # Gun-related subreddits
            poltitician_mentioned = False  # Not Kamala-related
            return combined_score, poltitician_mentioned
        else:
            # Assume Kamala Harris for other subreddits
            mentioned_group = 'democrat'
            combined_score *= -1
            politician_mentioned = True

    if not politician_mentioned:
        for republican in republicans:
            if republican.lower() in text.lower():
                mentioned_group = 'republican'
                combined_score *= 1  # No change for Republicans
                politician_mentioned = True
                break

    if mentioned_group is None:  # Only check Democrats if no Republican match
        for democrat in democrats:
            if democrat.lower() in text.lower():
                mentioned_group = 'democrat'
                combined_score *= -1  # Multiply by -1 for Democrats
                politician_mentioned = True
                break

    # Return both combined_score and politician_mentioned
    return combined_score, politician_mentioned

def main(input_csv, output_csv):

    # Read the parquet file
    data = pd.read_parquet(input_csv)
    data['subreddit'].value_counts()

    # Ensure 'body' is of type string for all_activity data
    #data['body'] = data['body'].astype(str)
    # for posts data
    data['text'] = data['title'] + ' ' + data['selftext'].fillna('')

    data['author'] = data['author'].str.strip()


    # Prepare texts for sentiment analysis
    texts = data['text'].tolist()
    subreddits = data['subreddit'].tolist()

    # Combine texts and subreddits into a list of tuples
    input_data = list(zip(texts, subreddits))

    # Initialize multiprocessing pool with 'spawn' to avoid tokenizer warnings
    multiprocessing.set_start_method('spawn', force=True)

    with Pool(processes=multiprocessing.cpu_count(), initializer=init_sentiment_analyzers) as pool:
        results = pool.map(calculate_combined_sentiment, input_data)

    # Convert results to DataFrame
    sentiments_df = pd.DataFrame(results, columns=['sentiment', 'politician_mentioned'])
    data = data.reset_index(drop=True)
    sentiments_df = sentiments_df.reset_index(drop=True)
    # Joining the two dataframes so I have sentiment analysis with comments and other columns
    data = pd.concat([data, sentiments_df], axis=1)

    # Now filter out comments that don't mention politicians
    data = data[data['politician_mentioned'] == True]

    # Save the DataFrame with sentiment per comment
    data.to_csv(output_csv, index=False)
    print(f"Sentiment analysis completed. Output saved to {output_csv}")

if __name__ == '__main__':
    input_csv = '../allsubsmall2.parquet'
    output_csv = 'nongroupedsentiposts.csv'
    main(input_csv, output_csv)
