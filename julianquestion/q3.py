import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to remove stopwords and tokenize text
def rm_stopwords_tokenize(text_series):
    stop_words = set(stopwords.words('english'))
    tokenized_list = []
    for text in text_series:
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        tokenized_list.append(' '.join(filtered_tokens))
    return tokenized_list

# Function to compute n-gram features
def compute_ngram_features(data, ngram_range=(1, 2), top_n=10):
    tokenized_data = rm_stopwords_tokenize(data['body'])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(tokenized_data)
    features = vectorizer.get_feature_names_out()

    # Convert sparse matrix to dense and aggregate by author
    ngram_df = pd.DataFrame(X.todense(), columns=features)
    ngram_df['author'] = data['author'].values
    user_ngrams = ngram_df.groupby('author').sum().reset_index()

    # Select top N n-grams
    counts = np.array(X.sum(axis=0)).flatten()
    feature_counts = list(zip(features, counts))
    sorted_feature_counts = sorted(feature_counts, key=lambda x: x[1], reverse=True)
    top_ngrams = [feature for feature, count in sorted_feature_counts[:top_n]]
    user_ngrams = user_ngrams.loc[:, user_ngrams.columns.isin(top_ngrams + ['author'])]

    return user_ngrams

# Function to compute time features
def compute_time_features(data):
    # Convert UTC timestamp to datetime
    data['created_utc'] = pd.to_datetime(data['created_utc'], unit='s')
    data['hour'] = data['created_utc'].dt.hour

    # Group by author to find the most common posting hour
    author_time = data.groupby('author')['hour'].agg(lambda x: x.value_counts().index[0]).reset_index()
    author_time.rename(columns={'hour': 'most_common_hour'}, inplace=True)

    # Classify hours into time slots
    def get_time_slot(hour):
        if 0 <= hour <= 6:
            return 'Night'
        elif 7 <= hour <= 12:
            return 'Morning'
        elif 13 <= hour <= 18:
            return 'Afternoon'
        else:
            return 'Evening'

    author_time['time_slot'] = author_time['most_common_hour'].apply(get_time_slot)
    return author_time

# Main function
def main():
    # Read data
    data = pd.read_parquet('../allactivitysmall.parquet')

    # Ensure author names are standardized
    data['author'] = data['author'].str.strip().str.lower()

    # Compute n-gram features
    user_ngrams = compute_ngram_features(data, ngram_range=(1, 2), top_n=20)

    # Compute time features
    author_time = compute_time_features(data)

    # Merge features
    author_features = user_ngrams.merge(author_time, on='author', how='left')

    ngram_columns = [col for col in author_features.columns if col not in ['author', 'most_common_hour', 'time_slot']]

    # Group n-grams by time slot
    ngrams_by_time_slot = author_features.groupby('time_slot')[ngram_columns].mean()

    # Save and display results
    print("Average N-gram Occurrence by Time Slot:")
    print(ngrams_by_time_slot)

    # Plot N-grams by time slot
    ngrams_by_time_slot.T.plot(kind='bar', figsize=(12, 6))
    plt.title('Average N-gram Occurrence by Time Slot')
    plt.xlabel('N-grams')
    plt.ylabel('Average Occurrence')
    plt.xticks(rotation=45)
    plt.legend(title='Time Slot')
    plt.tight_layout()
    plt.savefig('ngrams_by_time_slot.png')
    plt.show()

if __name__ == '__main__':
    main()
