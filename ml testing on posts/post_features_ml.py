import pandas as pd
import nltk
#nltk.download('vader_lexicon') #for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from transformers import pipeline, GPT2Tokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sys


#defining calculate sentiment function
def calculate_combined_sentiment(text, republicans, democrats, sia, sentiment_pipeline):
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
    mentioned_group = None
    for republican in republicans:
        if republican.lower() in text.lower():
            mentioned_group = 'republican'
            combined_score *= 1  # No change for Republicans
            break

    if mentioned_group is None:  # Only check Democrats if no Republican match
        for democrat in democrats:
            if democrat.lower() in text.lower():
                mentioned_group = 'democrat'
                combined_score *= -1  # Multiply by -1 for Democrats
                break

    return combined_score, mentioned_group, hf_label


def aggregate_user_data(posts_df):
    # Aggregate user data
    user_data = posts_df.groupby('author').agg({
        'sentiment': 'mean',  # Average sentiment score
        **{col: 'sum' for col in posts_df.columns if col.startswith('subreddit_')},  # Count subreddit comments
        'id': 'count',  # Total number of posts/comments per user
    }).rename(columns={'id': 'post_count'}).reset_index()

    return user_data

def perform_pca(X):
    #pca and standardize
    pca_model = make_pipeline(
        StandardScaler(),
        PCA(n_components=2)
    )

    #fitting and transforming the returned array
    X2 = pca_model.fit_transform(X)
    assert X2.shape == (X.shape[0],2)

    return X2

def perform_pca_and_plot(data):
    #keep only numberic values from data
    numeric_df = data.drop(columns=['author', 'subreddit_id'])

    #use perform_pca function
    X2 = perform_pca(numeric_df)

    #plot
    plt.figure(figsize=(10,6))
    plt.scatter(X2[:,0], X2[:,1], c=numeric_df['sentiment'], cmap = 'coolwarm', s=30)
    #plt.show()
    plt.savefig('PCA.png')

#main 
def main(in_data):
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    #reading in post data
    #posts_df = pd.read_csv(in_data)
    posts_df = pd.read_parquet(in_data)

    #posts_df = pd.read_csv('../allactivityoutput.csv')
    # print(posts_df.head())

    # Combine 'title' and 'selftext' into a single 'text' column
    posts_df['text'] = posts_df['title'] + ' ' + posts_df['selftext'].fillna('')

    # turning the posts into numerical feature using the algorithm i came up with (most basic version just using politican names) 
    #defining arrays of republicans and democrats (note sample is from 2016 might expand/adjust these arrays)
    republicans = ['Trump', 'Pence', 'DeSantis', 'McConnell', 'Cruz', 'Rubio']
    democrats = ['Hillary', 'Clinton', 'Biden', 'Harris', 'Pelosi', 'Sanders', 'Schumer']

    #intialize sentiment pipeline 
    sentiment_pipeline = pipeline("sentiment-analysis",
                                  model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
                                  device=-1)

    # add sentiment and group columns
    # if using allactivityouput.csv, change posts_df['text'] to posts_df['body']
    posts_df['sentiment'], posts_df['group'], posts_df['hf_label'] = zip(*posts_df['text'].apply(
        lambda x: calculate_combined_sentiment(x, republicans, democrats,sia,sentiment_pipeline)
    ))
    
    #one hot encoding the name of the subreddit as a feature. Right now not very big set of subreddits 
    #but will be scalable for when we get more. If you think about it different subreddits will 
    #likely have more voters for different parties. 

    # One-hot encode the subreddit column and add as columns to original df 
    subreddit_encoded = pd.get_dummies(posts_df['subreddit'], prefix='subreddit')
    posts_df = pd.concat([posts_df, subreddit_encoded], axis=1)

    #another potential feature: upvotes ('ups'?)

    # DONE ADDING FEATURES AS OF HERE
    # aggregate a users sentimental score (average) and add features of subreddits they've commented on.
    user_data = aggregate_user_data(posts_df)
    
    # Drop rows with sentiment scores near zero  
    user_data_filtered = user_data[user_data['sentiment'].abs() > 0.2]

    #print(user_data_filtered.head(10))

    # Perform PCA and plot
    perform_pca_and_plot(user_data_filtered)

    # running unsupervised ml 
    #TODO

    # running supervised ml
    #TODO

    # score 
    #TODO 


if __name__ == '__main__':
    in_data = sys.argv[1]
    main(in_data)
