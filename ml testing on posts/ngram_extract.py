import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
# using TfidfVecotrizer is better for clustering:
# https://scikit-learn.org/1.5/modules/feature_extraction.html#stop-words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def rm_stopwords_tokenize(text_series):
    stop_words = set(stopwords.words('english'))
    tokenized_list = []
    # each elem in data['body'] is a string
    # word_tokenize works on a string only
    for text in text_series:
        tokens = word_tokenize(text)
        # filter out stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        tokenized_list.append(' '.join(filtered_tokens))  # join tokens back into a string
    return tokenized_list

def main():
    data = pd.read_csv('../allactivityoutput.csv')
    # following tutorial:
    # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/#removing-stop-words-with-sklearn
    # https://www.ibm.com/reference/python/countvectorizer
    # tokenize the data and remove stopwords using NLTK

    tokenized_data = rm_stopwords_tokenize(data['body'])
    # get unigrams and bigrams using TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(tokenized_data)
    # extract feature names
    features = vectorizer.get_feature_names_out()

    # convert the sparse array (X) to dense array
    # https://stackoverflow.com/questions/30416695/
    # make result into a dataframe to count the unigrams and bigrams each user uses
    ngram_df = pd.DataFrame(X.todense(), columns=features)
    ngram_df['author'] = data['author']
    user_ngrams = ngram_df.groupby('author').sum().reset_index()

    #print(user_ngrams[['author','trump']])

    # convert to array sum occurrences of each feature (unigrams and bigrams) across the text data
    # flatten to 1D array
    counts = np.array(X.sum(axis=0)).flatten()

    # pair the features with their counts
    feature_counts = list(zip(features, counts))

    # sort the features by count (descending)
    # sorting feature_counts which is a tuple
    sorted_feature_counts = sorted(feature_counts, key=lambda x: x[1], reverse=True)

    # print out top 10 unigrams and bigrams
    top_ngrams = []
    print("Top 10 unigrams and bigrams:")
    for feature, count in sorted_feature_counts[:10]:
        print(f"{feature}: {count}")
        top_ngrams.append(feature)

    # https://stackoverflow.com/questions/48198021/
    # filter user_ngrams dataframe for column names in top_ngrams
    user_ngrams = user_ngrams.loc[:, user_ngrams.columns.isin(top_ngrams + ['author'])]
    print(user_ngrams)
    # output shows that for each author, the frequency they use each ngram, each ngram is a feature
    # we can now join user_ngrams with user_data_filtered in 'ml testing on posts/post_features_ml.py'
    # or we can keep things separate and do clustering here

    # do Kmeans

    # reduce the dimensionality to 2D using PCA

    return

if __name__ == '__main__':
    main()