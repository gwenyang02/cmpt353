# supervised ML
# Can we predict the score of a post based on features such as the words used,
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
from sklearn.linear_model import LinearRegression

OUTPUT_TEMPLATE = (
    'Rand forest regressor:     {rand_forest_train:.3f} {rand_forest_valid:.3f}\n'
   # 'Linear regression:         {lin_reg_train:.3f} {lin_reg_valid:.3f}\n'
    'Gradient Boosting:         {grad_boost_train:.3f} {grad_boost_valid:.3f}\n'
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
def main():
    # sentiment_no_polarity.csv provides the sentiment score for each post evaluated
    # by VADER and HuggingFace, without any post processing to label
    # sentiment as republican leaning or democrat leaning
    data = pd.read_csv('../csvfiles/sentiment_no_polarity.csv',
                       usecols=['body','subreddit','sentiment','created_utc','score'],index_col=False)


    # visualize the data for scores
    #plt.hist(data['score'],bins=20)
    #plt.show()
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
    #data = data.groupby(['author','subreddit']).mean().reset_index()
    # make a copy of the data but unlabelled
    unlabelled_data = data.head(200).copy()
    unlabelled_data['score_category'] = ''

    # labelled data uses the rest of the original data from line 200 onwards
    labelled_data = data[200:]
    # check the number of posts from each subreddit
    # remove subreddits with less than 100 posts for more balanced data
    print(labelled_data['subreddit'].value_counts())
    # guncontrol and prochoice subreddits have less than 100 posts
    labelled_data = labelled_data[~labelled_data['subreddit'].isin(['guncontrol','prochoice'])]
    # next smallest subreddit is PoliticalDiscussion = 174
    smallest_sub = labelled_data[labelled_data['subreddit'] == 'PoliticalDiscussion']
    # sample each subreddit the same size as smallest_sub
    balanced_data = labelled_data.sample(n=smallest_sub.shape[0], random_state=42)
    X = balanced_data[['body','subreddit','sentiment','created_utc']]
    y = balanced_data['score_category'] # array with shape (n,) of scores.

    # split labelled data to training and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # build model to predict y from X
    # print model's accuracy score
    # use one-hot-encoding for text like author and subreddit
    # source: https://dev.to/aashwinkumar/countvectorizer-vs-tfidfvectorizer-1kck
    # use tfidVectorizer to get weighted matrix of most frequent words in all the text
    # tfidVectorizer is similar to one-hot-encoding for each word
    rand_forest = make_pipeline(
        ColumnTransformer(
            transformers=[
            #('author', OneHotEncoder(handle_unknown='ignore'), ['author']),
            ('subreddit', OneHotEncoder(handle_unknown='ignore'), ['subreddit']),
            ('body', TfidfVectorizer(max_features=100), 'body'), # limiting to 100 words
            ('created_utc', FunctionTransformer(get_unixtime), ['created_utc'])
        ], remainder='passthrough'), # sentiment column doesn't need processing
        RandomForestClassifier(n_estimators = 100,
                                   max_depth = 6,
                                   min_samples_leaf = 40,
                                   min_samples_split = 50)
    )
    rand_forest.fit(X_train, y_train)

# gradient boosting

    grad_boost = make_pipeline(
        ColumnTransformer(
            transformers=[
            #('author', OneHotEncoder(handle_unknown='ignore'), ['author']),
            ('subreddit', OneHotEncoder(handle_unknown='ignore'), ['subreddit']),
            ('body', TfidfVectorizer(max_features=100), 'body'), # limiting to 100 words
            ('created_utc', FunctionTransformer(get_unixtime), ['created_utc'])
        ], remainder='passthrough'), # sentiment column doesn't need processing
        GradientBoostingClassifier(n_estimators = 100,
                                   max_depth = 6,
                                   min_samples_leaf = 40,
                                   min_samples_split = 50)
    )

    grad_boost.fit(X_train, y_train)
    print(OUTPUT_TEMPLATE.format(
        rand_forest_train=rand_forest.score(X_train, y_train),
        rand_forest_valid=rand_forest.score(X_valid, y_valid),
        grad_boost_train=grad_boost.score(X_train, y_train),
        grad_boost_valid=grad_boost.score(X_valid, y_valid),
    ))
    # use the features in the unlabelled data to predict using the model
    predictions = grad_boost.predict(unlabelled_data.drop(columns=['score_category']))
    print(pd.Series(predictions))
    #pd.Series(predictions).to_csv('predicted_vals.csv', index=False, header=False)
    df = pd.DataFrame({'truth': y_valid, 'prediction': grad_boost.predict(X_valid)})
    print(df[df['truth'] != df['prediction']])

if __name__ == '__main__':
    main()


# potenitally interesting clustering if possible (not needed)


# MOVING FORWARD USING CHAT TO GIVE LABELS THEN USING OUR SENTIMENT ANALYSIS AS A FEATURE COMBINED WITH ALL OUR OTHERS FOR CLASSIFICATION/PREDICTING CHAT LABEL
#(also we can remove rows where chat thinks there is not an indicator of sentiment )

# clustering with subsets of the data

#then we can actually do train test split on that 


#do this for comments and titles


# justification about using chatgpt explain our process




#something else interesting we could do polynomial regression on it to predict magnitudes (food for thought not needed to do EXTRA)
