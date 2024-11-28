import os
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import multiprocessing
from multiprocessing import Pool

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Display all columns
pd.set_option('display.max_columns', None)

# Global variables for sentiment analysis
sia = None
sentiment_pipeline = None


#filter by politicans first
def filter_by_politicians(data):
    politicians = ['Trump', 'Pence', 'DeSantis', 'McConnell', 'Cruz', 'Rubio',
                   'Hillary', 'Clinton', 'Biden', 'Harris', 'Pelosi', 'Sanders', 'Schumer']
    # Combine the list into a regex pattern
    pattern = '|'.join(politicians)
    return data[data['body'].str.contains(pattern, case=False, na=False)]

# Feature functions
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
        model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
        device=-1
    )

def calculate_combined_sentiment(text):
    """
    Calculates combined sentiment using VADER and Hugging Face's sentiment-analysis pipeline.
    Also checks for mentions of specified politicians.
    """
    global sia, sentiment_pipeline

    # Define lists of politicians
    republicans = ['Trump', 'Pence', 'DeSantis', 'McConnell', 'Cruz', 'Rubio']
    democrats = ['Hillary', 'Clinton', 'Biden', 'Harris', 'Pelosi', 'Sanders', 'Schumer']

    # Determine VADER sentiment score
    vader_score = sia.polarity_scores(text)['compound']

    # Determine Hugging Face sentiment label
    hf_result = sentiment_pipeline(text)[0]
    hf_label = hf_result['label']  # POSITIVE or NEGATIVE

    # Combine VADER and Hugging Face results
    if hf_label == "POSITIVE":
        combined_score = abs(vader_score)
    else:  # NEGATIVE
        combined_score = -abs(vader_score)

    # Check for mentions of Republicans or Democrats
    politicianmentioned = False
    mentioned_group = None
    for republican in republicans:
        if republican.lower() in text.lower():
            mentioned_group = 'republican'
            combined_score *= 1  # No change for Republicans
            politicianmentioned = True
            break

    if mentioned_group is None:  # Only check Democrats if no Republican match
        for democrat in democrats:
            if democrat.lower() in text.lower():
                mentioned_group = 'democrat'
                combined_score *= -1  # Multiply by -1 for Democrats
                politicianmentioned = True
                break

    return combined_score, politicianmentioned

def compute_sentiment_features(data):
    """
    Aggregates sentiment scores per author: mean, min, and max.
    """
    author_sentiment = data.groupby('author')['sentiment'].agg(['mean', 'min', 'max']).reset_index()
    author_sentiment.rename(columns={
        'mean': 'sentiment_mean',
        'min': 'sentiment_min',
        'max': 'sentiment_max'
    }, inplace=True)
    return author_sentiment

def rm_stopwords_tokenize(text_series):
    """
    Tokenizes text and removes English stopwords.
    """
    stop_words = set(stopwords.words('english'))
    tokenized_list = []
    for text in text_series:
        tokens = word_tokenize(text)
        # Filter out stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        tokenized_list.append(' '.join(filtered_tokens))  # Join tokens back into a string
    return tokenized_list

def compute_ngram_features(data, ngram_range=(1, 2), top_n=10):
    """
    Computes TF-IDF n-gram features and selects the top_n most frequent n-grams.
    """
    # Tokenize and remove stopwords
    tokenized_data = rm_stopwords_tokenize(data['body'])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(tokenized_data)
    features = vectorizer.get_feature_names_out()
    
    # Convert the sparse matrix X to dense and create ngram_df
    ngram_df = pd.DataFrame(X.todense(), columns=features)
    ngram_df['author'] = data['author'].values
    user_ngrams = ngram_df.groupby('author').sum().reset_index()
    
    counts = np.array(X.sum(axis=0)).flatten()
    feature_counts = list(zip(features, counts))
    
    # Sort the features by counts (descending)
    sorted_feature_counts = sorted(feature_counts, key=lambda x: x[1], reverse=True)
    
    top_ngrams = [feature for feature, count in sorted_feature_counts[:top_n]]
    
    user_ngrams = user_ngrams.loc[:, user_ngrams.columns.isin(top_ngrams + ['author'])]
    
    return user_ngrams

def compute_time_features(data):
    """
    Extracts the most common posting hour per author.
    """
    if data['created_utc'].dtype == 'int64' or data['created_utc'].dtype == 'float64':
        data['created_utc'] = pd.to_datetime(data['created_utc'], unit='s')

    # Extract hour 
    data['hour'] = data['created_utc'].dt.hour

    # Most common posting hour per author
    author_time = data.groupby('author')['hour'].agg(lambda x: x.value_counts().index[0]).reset_index()
    author_time.rename(columns={'hour': 'most_common_hour'}, inplace=True)
    return author_time

def compute_avg_score_per_subreddit_per_author(data):
    """
    Calculates the average score per subreddit per author and pivots the data.
    """
    # Calculate the average score per subreddit per author
    author_subreddit_scores = data.groupby(['author', 'subreddit'])['score'].mean().reset_index()
    author_subreddit_scores.rename(columns={'score': 'avg_score'}, inplace=True)

    # Pivot the data to have subreddits as columns
    avg_score_per_subreddit = author_subreddit_scores.pivot(index='author', columns='subreddit', values='avg_score').reset_index()

    # Rename columns
    avg_score_per_subreddit.columns = ['author'] + [f'avg_score_{col}' for col in avg_score_per_subreddit.columns if col != 'author']

    # Fill NaN values with 0
    avg_score_per_subreddit.fillna(0, inplace=True)

    return avg_score_per_subreddit

def compute_subreddit_features(data):
    """
    One-hot encodes subreddits and aggregates counts per author.
    """
    subreddit_encoded = pd.get_dummies(data['subreddit'], prefix='subreddit')
    subreddit_data = pd.concat([data[['author']], subreddit_encoded], axis=1)
    subreddit_counts = subreddit_data.groupby('author').sum().reset_index()
    return subreddit_counts

def perform_pca(X):
    """
    Performs PCA on the feature set and reduces it to 2 components.
    """
    pca_model = make_pipeline(
        StandardScaler(),
        PCA(n_components=2)
    )

    # Fitting and transforming the data
    X2 = pca_model.fit_transform(X)
    assert X2.shape == (X.shape[0], 2)

    return X2

def perform_Kmeans(data):
    """
    Performs KMeans clustering on the data.
    """
    clusters = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42))

    clusters.fit(data)

    return clusters.predict(data)

def main():
    # Read all activity dataframe
    data = pd.read_parquet('./allactivity.parquet')

    print(data.head(10))

    max_length = 512  # Maximum sequence length for the model
    data['text_too_long'] = data['body'].apply(lambda x: len(x.split()) > max_length)

    # Filter out rows where text is too long so that Hugging Face works
    data = data[~data['text_too_long']]

    data = filter_by_politicians(data)

    # Standardize author names to lowercase and strip spaces to ensure consistency
    data['author'] = data['author'].str.strip().str.lower()

    # Prepare texts for sentiment analysis
    texts = data['body'].tolist()

    # Initialize multiprocessing pool with 'spawn' to avoid tokenizer warnings
    multiprocessing.set_start_method('spawn', force=True)

    with Pool(processes=multiprocessing.cpu_count(), initializer=init_sentiment_analyzers) as pool:
        results = pool.map(calculate_combined_sentiment, texts)

    # Convert results to DataFrame
    sentiments_df = pd.DataFrame(results, columns=['sentiment', 'politicianmentioned'])
    data = data.reset_index(drop=True)
    sentiments_df = sentiments_df.reset_index(drop=True)
    data = pd.concat([data, sentiments_df], axis=1)

    # Throw out rows where overall sentiment analysis returns < 0.2 or politicianmentioned is false
    data = data[(data['sentiment'].abs() >= 0.2) & (data['politicianmentioned'])]

    # Compute additional sentiment features
    author_sentiment = compute_sentiment_features(data)

    # Get n-grams feature
    user_ngrams = compute_ngram_features(data, ngram_range=(1, 2), top_n=10)

    # Compute time-based features
    author_time = compute_time_features(data)

    # Compute average score per subreddit per author
    avg_score_per_subreddit = compute_avg_score_per_subreddit_per_author(data)

    # One-hot encoded subreddit features
    author_subreddit_counts = compute_subreddit_features(data)

    # Verify the number of unique authors
    unique_authors = data['author'].nunique()
    print(f"Number of unique authors: {unique_authors}")

    # Merge all author-level features
    author_features = user_ngrams.merge(author_sentiment, on='author', how='left')
    author_features = author_features.merge(author_time, on='author', how='left')
    author_features = author_features.merge(avg_score_per_subreddit, on='author', how='left')
    author_features = author_features.merge(author_subreddit_counts, on='author', how='left')

    # Fill any remaining NaN values
    author_features.fillna(0, inplace=True)

    # Proceed with your machine learning model using 'author_features'
    print(author_features.head())
    author_features.to_csv('featuresoutputallactivity.csv', index=False)

    # Unsupervised KMeans clustering
    numeric_author_features = author_features.drop(columns='author')
    p_components = perform_pca(numeric_author_features)
    kmeans_clusters = perform_Kmeans(numeric_author_features)

    # Plotting KMeans
    plt.figure(figsize=(10,6))
    plt.scatter(p_components[:,0], p_components[:,1], c=kmeans_clusters, cmap='Set1', s=30)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='cluster')
    plt.savefig('KMeans.png')

    return

if __name__ == '__main__':
    main()
