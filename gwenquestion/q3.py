import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
def compute_ngram_features(data, ngram_range=(1, 2),top_n=50):
    tokenized_data = rm_stopwords_tokenize(data['body'])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(tokenized_data)
    features = vectorizer.get_feature_names_out()

    # convert the sparse array (X) to dense array
    # https://stackoverflow.com/questions/30416695/
    # make result into a dataframe to count the unigrams and bigrams each user uses
    ngram_df = pd.DataFrame(X.todense(), columns=features)
    ngram_df['subreddit'] = data['subreddit'].values
    user_ngrams = ngram_df.groupby('subreddit').sum().reset_index()

    # get top_n ngrams
    # convert to array sum occurrences of each feature (unigrams and bigrams) across the text data
    # flatten to 1D array
    counts = np.array(X.sum(axis=0)).flatten()
    # pair the features with their counts
    feature_counts = list(zip(features, counts))
    sorted_feature_counts = sorted(feature_counts, key=lambda x: x[1], reverse=True)
    # source: https://stackoverflow.com/questions/48198021/
    # filter user_ngrams dataframe for column names in top_ngrams
    top_ngrams = [feature for feature, count in sorted_feature_counts[:top_n]]
    user_ngrams = user_ngrams.loc[:, user_ngrams.columns.isin(top_ngrams + ['subreddit'])]

    return user_ngrams

# Main function
def main():
    # Read data
    data = pd.read_parquet('../datafiles/allactivitysmall2.parquet')

    # Compute n-gram features
    user_ngrams = compute_ngram_features(data, ngram_range=(1, 2),top_n=50)
    # Drop the 'subreddit' column before applying KMeans
    user_ngrams_data = user_ngrams.drop(columns=['subreddit'])

    # Standardize the features (optional but recommended)
    scaler = StandardScaler()
    user_ngrams_scaled = scaler.fit_transform(user_ngrams_data)

    # Apply KMeans clustering
    # each point in the plot is a subreddit
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(user_ngrams_scaled)

    # user_ngrams is the dataframe with the KMeans labels
    # dropping subreddit means that it will be the points on the plot
    X = user_ngrams.drop(columns=['subreddit'])
    X_pca_scaled = scaler.fit_transform(X)

    # reduce to 2 dimensions to visualize
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_pca_scaled)
    feature_names = X.columns.tolist()
    pca_components = pca.components_

    # get pca contributions into a dataframe
    # source: https://www.jcchouinard.com/pca-loadings/
    pca_feature_importance = pd.DataFrame(
        pca_components.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    # plot the most imporant features
    # source: https://stackoverflow.com/questions/50796024/
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c= kmeans.labels_, s=50, alpha=0.7)
    plt.title('Subreddit Clusters Based on N-Grams')
    plt.xlabel(
        f'PCA Component 1 (Top features: {", ".join(pca_feature_importance["PC1"].abs()
                                                    .sort_values(ascending=False).head(5).index.tolist())})')
    plt.ylabel(
        f'PCA Component 2 (Top features: {", ".join(pca_feature_importance["PC2"].abs()
                                                    .sort_values(ascending=False).head(5).index.tolist())})')

    # put labels for each of the subreddit points
    # source: https://python-graph-gallery.com/515-intro-pca-graph-python/
    # source: https://stackoverflow.com/questions/60786421/how-do-you-offset-text-in-a-scatter-plot-in-matplotlib
    for i, subreddit in enumerate(user_ngrams['subreddit']):
        plt.annotate(subreddit, (X_pca[i, 0], X_pca[i, 1]),
                     xytext=(5,5),textcoords="offset points", fontsize=8)
    plt.colorbar(label='Cluster')
    plt.savefig("subredditclusters.png")

    return


if __name__ == '__main__':
    main()
