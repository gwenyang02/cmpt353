# supervised ML
# Can we predict a post's engagement score (e.g., upvotes or score) based on features such as the author,
# sentiment score, and other metadata?
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer



# idea: aggregated by user sentiment analysis x user aggregated activity on subreddits
def get_unixtime(datetime_col):
    '''
     source: https://stackoverflow.com/questions/11865458/
    :param datetime_col: an array with datetime values
    :return: an array with posix values
    '''
    return datetime_col.astype('datetime64[s]').astype('int')

def main():
    # sentiment_no_polarity.csv provides the sentiment score for each post evaluated
    # by VADER and HuggingFace, without any post processing to label
    # sentiment as republican leaning or democrat leaning
    data = pd.read_csv('../csvfiles/sentiment_no_polarity.csv',
                       usecols=['author','subreddit','sentiment','score'],index_col=False)
    data = data.groupby(['author','subreddit']).mean().reset_index()
    # make a copy of the data but unlabelled
    unlabelled_data = data.head(200)[['author', 'subreddit', 'sentiment']].copy()
    unlabelled_data['score'] = np.nan

    # labelled data uses the rest of the original data from line 200 onwards
    labelled_data = data[200:]
    # check the number of posts from each subreddit
    # remove subreddits with less than 100 posts for more balanced data
    print(labelled_data['subreddit'].value_counts())
    # guncontrol and prochoice subreddits have less than 100 posts
    labelled_data = labelled_data[~labelled_data['subreddit'].isin(['AmericanPolitics','guncontrol','prochoice'])]
    # next smallest subreddit is PoliticalDiscussion = 174
    smallest_sub = labelled_data[labelled_data['subreddit'] == 'uspolitics']
    # sample each subreddit the same size as smallest_sub
    balanced_data = labelled_data.groupby('subreddit').sample(n=smallest_sub.shape[0], random_state=42)
    X = balanced_data[['author','subreddit','sentiment']]
    y = balanced_data['score'].values # array with shape (n,) of scores.

    # split labelled data to training and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # build model to predict y from X
    # print model's accuracy score
    # use one-hot-encoding for text like author and subreddit
    # source: https://dev.to/aashwinkumar/countvectorizer-vs-tfidfvectorizer-1kck
    # use tfidVectorizer to get weighted matrix of most frequent words in all the text
    # tfidVectorizer is similar to one-hot-encoding for each word
    model = make_pipeline(
        ColumnTransformer([
            ('author', OneHotEncoder(handle_unknown='ignore'), ['author']),
            ('subreddit', OneHotEncoder(handle_unknown='ignore'), ['subreddit'])
            #('body', TfidfVectorizer(max_features=100), 'body'), # limiting to 100 words
            #('created_utc', FunctionTransformer(get_unixtime), ['created_utc'])
        ], remainder='passthrough'), # sentiment column doesn't need processing
        RandomForestRegressor()
    )
    model.fit(X_train, y_train)
    print(f"validation score: {model.score(X_valid, y_valid)}")
    print(f"training score: {model.score(X_train, y_train)}")  # check for overfitting

    # use the features in the unlabelled data to predict using the model
    predictions = model.predict(unlabelled_data.iloc[:,:3])
    print(pd.Series(predictions))
    #pd.Series(predictions).to_csv('predicted_vals.csv', index=False, header=False)
    df = pd.DataFrame({'truth': y_valid, 'prediction': model.predict(X_valid)})
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
