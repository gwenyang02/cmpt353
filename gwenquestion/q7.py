# supervised ML
# Can we predict the score level of a post based on features such as the words used,
# sentiment score, date, subreddit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from matplotlib import cm


OUTPUT_TEMPLATE = (
    'Rand forest:           {rand_forest_train:.3f} {rand_forest_valid:.3f}\n'
    'Gradient Boosting:     {grad_boost_train:.3f} {grad_boost_valid:.3f}\n'
    'MLP Classifier:        {mlp_train:.3f} {mlp_valid:.3f}\n'
)
# idea: aggregated by user sentiment analysis x user aggregated activity on subreddits
def get_unixtime(datetime_col):
    '''
     source: https://stackoverflow.com/questions/11865458/
    :param datetime_col: an array with datetime values
    :return: an array with posix values
    '''
    return datetime_col.astype('datetime64[s]').astype('int')

def categorize_score(x):
    '''
    :param x: an integer representing post score
    :return: an integer representing post score group
    '''
    if x < -10:
        return 1 #'very_negative'
    elif -10 <= x < 0:
        return 2 #'slightly_negative'
    elif x == 0:
        return 3 #'neutral'
    elif 0 < x <= 10:
        return 4 #'slightly_positive'
    else:
        return 5 #'very_positive'

def get_clusters(X):
    """
    Find clusters of the data.
    """
    model = make_pipeline(
        ColumnTransformer(
            transformers=[
            ('subreddit', OneHotEncoder(handle_unknown='ignore'), ['subreddit']),
            ('body', TfidfVectorizer(max_features=100,stop_words='english'), 'body'), # limiting to 100 words
            ('created_utc', FunctionTransformer(get_unixtime), ['created_utc'])
        ], remainder='passthrough'), # sentiment column doesn't need processing
        KMeans(n_clusters=5)
    )
    model.fit(X)

    return model.predict(X)

def main():
    # sentiment_no_polarity.csv provides the sentiment score for each post evaluated
    # by VADER and HuggingFace, without any post processing to label
    # sentiment as republican leaning or democrat leaning
    data = pd.read_csv('../csvfiles/sentiment_no_polarity.csv',
                       usecols=['body','subreddit','sentiment','created_utc','score'],
                       index_col=False)


    # visualize the data for scores
    plt.hist(data['score'],bins=20)
    plt.show()
    # use linear regresion: Do posts with higher sentiment scores have higher scores?
    #plt.scatter(data['sentiment'],data['score'])
    # data for scores is highly skewed, most are under 1000
    #plt.show()

    # remove scores > 500 for outliers
    data = data[data['score']<500]
    data['score_category'] = data['score'].apply(categorize_score)
    plt.hist(data['score_category'],bins=5)
    #plt.show()
   # plt.scatter(data['sentiment'], data['score'])
    #plt.show()

    # make a copy of the data but unlabelled
    unlabelled_data = data.head(200).copy()
    unlabelled_data['score_category'] = ''

    # labelled data uses the rest of the original data from line 200 onwards
    labelled_data = data[200:]
    # check the number of posts from each subreddit
    # remove subreddits with less than 100 posts for more balanced data
    print(labelled_data['subreddit'].value_counts())
    # guncontrol and prochoice subreddits have less than 100 posts
    labelled_data = labelled_data[~labelled_data['subreddit']
                    .isin(['guncontrol','prochoice'])]
    # next smallest subreddit is PoliticalDiscussion = 174
    smallest_sub = labelled_data[labelled_data['subreddit'] == 'PoliticalDiscussion']
    # sample each subreddit the same size as smallest_sub
    balanced_data = labelled_data.sample(n=smallest_sub.shape[0])
    X = balanced_data[['body','subreddit','sentiment','created_utc']]
    y = balanced_data['score_category'] # array with shape (n,) of scores.



    # split labelled data to training and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X, y)
    X_train3, X_valid3, y_train3, y_valid3 = train_test_split(X, y)

    # build model to predict y from X
    # print model's accuracy score
    # use one-hot-encoding for text like author and subreddit
    # source: https://dev.to/aashwinkumar/countvectorizer-vs-tfidfvectorizer-1kck
    # use tfidVectorizer to get weighted matrix of most frequent words in all the text
    # tfidVectorizer is similar to one-hot-encoding for each word but
    # each column is weighted
    rand_forest = make_pipeline(
        ColumnTransformer(
            transformers=[
            #('author', OneHotEncoder(handle_unknown='ignore'), ['author']),
            ('subreddit', OneHotEncoder(handle_unknown='ignore'), ['subreddit']),
            ('body', TfidfVectorizer(max_features=100,stop_words='english'), 'body'), # limiting to 100 words
            ('created_utc', FunctionTransformer(get_unixtime), ['created_utc'])
        ], remainder='passthrough'), # sentiment column doesn't need processing
        RandomForestClassifier(n_estimators = 100,
                                   max_depth = 6,
                                   min_samples_leaf = 20,
                                   min_samples_split = 30)

    )
    rand_forest.fit(X_train, y_train)

    # gradient boosting classifier
    grad_boost = make_pipeline(
        ColumnTransformer(
            transformers=[
            ('subreddit', OneHotEncoder(handle_unknown='ignore'), ['subreddit']),
            ('body', TfidfVectorizer(max_features=100,stop_words='english'), 'body'), # limiting to 100 words
            ('created_utc', FunctionTransformer(get_unixtime), ['created_utc'])
        ], remainder='passthrough'), # sentiment column doesn't need processing
        GradientBoostingClassifier(n_estimators = 100,
                                   max_depth = 6,
                                   min_samples_leaf = 50,
                                   min_samples_split = 70)
    )

    grad_boost.fit(X_train2, y_train2)

    # neural network: mlp classifier
    mlp = make_pipeline(
        ColumnTransformer(
            transformers=[
            ('subreddit', OneHotEncoder(handle_unknown='ignore'), ['subreddit']),
            ('body', TfidfVectorizer(max_features=100,stop_words='english'), 'body'), # limiting to 100 words
            ('created_utc', FunctionTransformer(get_unixtime), ['created_utc'])
        ], remainder='passthrough'), # sentiment column doesn't need processing
        MLPClassifier(solver='adam',
                            hidden_layer_sizes=(8,4), activation='logistic',
                      alpha=0.05, max_iter=20000)
    )

    mlp.fit(X_train3, y_train3)

    print(OUTPUT_TEMPLATE.format(
        rand_forest_train=rand_forest.score(X_train, y_train),
        rand_forest_valid=rand_forest.score(X_valid, y_valid),
        grad_boost_train=grad_boost.score(X_train2, y_train2),
        grad_boost_valid=grad_boost.score(X_valid2, y_valid2),
        mlp_train=mlp.score(X_train3, y_train3),
        mlp_valid=mlp.score(X_valid3, y_valid3)
    ))
    # use the features in the unlabelled data to predict using the model
    predictions = grad_boost.predict(unlabelled_data.drop(columns=['score_category']))
    pd.Series(predictions).to_csv('predicted_vals.csv', index=False, header=False)
    df = pd.DataFrame({'truth': y_valid, 'prediction': rand_forest.predict(X_valid)})
    print(df[df['truth'] != df['prediction']])

    # plot most imporant features from random forest
    # get feature names and feature importances from rand_forest pipeline
    # source: https://stackoverflow.com/questions/38787612/
    # source: https://scikit-learn.org/1.5/auto_examples/ensemble/plot_forest_importances.html
    created_utc_feature_name = ['created_utc']

    subreddit_feature_names = rand_forest.named_steps['columntransformer'] \
        .named_transformers_['subreddit'].get_feature_names_out(['subreddit'])

    body_feature_names = rand_forest.named_steps['columntransformer'] \
        .named_transformers_['body'].get_feature_names_out()
    sentiment_feature_name = ['sentiment']
    # concat feature names into one array
    feature_names = np.concatenate([subreddit_feature_names, body_feature_names,
                                    created_utc_feature_name, sentiment_feature_name])

    importances =  rand_forest.named_steps['randomforestclassifier'].feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    std = np.std([tree.feature_importances_ for tree in
                  rand_forest.named_steps['randomforestclassifier'].estimators_], axis=0)

    # sort feature importances from largest to smallest
    # source: https://stackoverflow.com/questions/62536918/
    sorted_indices = np.argsort(importances)[::-1]  # get indices of importances sorted
    sorted_features = feature_names[sorted_indices]  # features sorted by importance
    sorted_importances = importances[sorted_indices]

    # display the top 5 features
    top_features = sorted_features[:5]
    top_importances = sorted_importances[:5]
    plt.figure(figsize=(10,6))
    plt.bar(top_features, top_importances)
    plt.title("Feature importances using MDI")
    plt.ylabel("Mean decrease in impurity")
    plt.show()
    import matplotlib.dates as mdates
    # top 2 features for plotting
    X_utc = X['created_utc'].astype('datetime64[s]').astype('int')
    X_sentiment = X['sentiment']

    # Stack them into a 2D array with shape (n, 2)
    X_array = np.column_stack((X_utc, X_sentiment))
    clusters = get_clusters(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(pd.to_datetime(X_array[:, 0], unit='s'), X_array[:, 1], c=clusters, cmap='Set1', edgecolor='k', s=30)

    plt.savefig('clusters.png')

    df = pd.DataFrame({
        'cluster': clusters,
        'score': y.values,
    })
    counts = pd.crosstab(df['score'], df['cluster'])
    print(counts)


if __name__ == '__main__':
    main()


