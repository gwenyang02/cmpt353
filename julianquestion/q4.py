import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_csv('./postsdates.csv')

# Convert 'created_utc' to datetime
data['created_utc'] = pd.to_datetime(data['created_utc'])

# Add weeks to election
election_date = pd.Timestamp('2016-11-08')  # Election date
data['weeks_to_election'] = (election_date - data['created_utc']).dt.days // 7

# Filter for election-related subreddits
election_subreddits = ["The_Donald", "AmericanPolitics", "Republican", "politics", "democrats", "prochoice", "immigration", "guncontrol"]
election_data = data[data['subreddit'].isin(election_subreddits)]

# Group by subreddit and weeks to election, then calculate activity counts
activity_by_week = (
    election_data.groupby(['subreddit', 'weeks_to_election'])
    .size()
    .reset_index(name='post_count')
)

# Plot activity trends for each subreddit with LOESS smoothing
plt.figure(figsize=(12, 8))
for subreddit in election_subreddits:
    subreddit_data = activity_by_week[activity_by_week['subreddit'] == subreddit]
    x = subreddit_data['weeks_to_election']
    y = subreddit_data['post_count']
    
    # Apply LOESS smoothing
    smoothed = lowess(y, x, frac=0.2)  
    
    # Plot smoothed data
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'{subreddit} (smoothed)', linewidth=2)

# Reverse x-axis (weeks count down to the election)
plt.gca().invert_xaxis()

# Add labels and title
plt.title('Activity Trends for Election-Related Subreddits with LOESS Smoothing')
plt.xlabel('Weeks to Election')
plt.ylabel('Number of Posts')
plt.legend(title='Subreddits', loc='upper left')
plt.tight_layout()
plt.show()

# Linear regression for overall trend analysis
overall_activity = (
    election_data.groupby('weeks_to_election')
    .size()
    .reset_index(name='post_count')
)

X = overall_activity['weeks_to_election'].values.reshape(-1, 1)
y = overall_activity['post_count'].values
reg = LinearRegression().fit(X, y)

# Output regression details
print("Overall Linear Regression Coefficients:")
print(f"Slope: {reg.coef_[0]}, Intercept: {reg.intercept_}")

# Plot overall trend
plt.figure(figsize=(10, 6))
plt.scatter(overall_activity['weeks_to_election'], overall_activity['post_count'], alpha=0.7, label='Activity')
plt.plot(overall_activity['weeks_to_election'], reg.predict(X), color='red', label='Trend Line')
plt.gca().invert_xaxis()  # Reverse axis so it counts down to the election
plt.title('Overall Activity Trend Across Election-Related Subreddits')
plt.xlabel('Weeks to Election')
plt.ylabel('Total Posts')
plt.legend()
plt.tight_layout()
plt.show()

#classification of a comment regarding an event based on the date as well as ngrams used/ or unsupervised ML based on date and ngrams used and 
# classifying debates or primaries. ... etc so for example the 


#function for classification
# Label posts based on proximity to events
# Label posts based on proximity to events
def label_event(row, event_weeks, window=2):
    for name, event_week in event_weeks:
        if abs(row['weeks_to_election'] - event_week) <= window:
            return name
    return "No Event"
#____________________

#classification portion
# Define key events with their dates
events = [
    ("Iowa Caucuses", pd.Timestamp('2016-02-01')),
    ("Super Tuesday", pd.Timestamp('2016-03-01')),
    ("Orlando Shooting", pd.Timestamp('2016-06-12')),
    ("Republican Convention", pd.Timestamp('2016-07-18')),
    ("Democratic Convention", pd.Timestamp('2016-07-25'))
]

# Add weeks to election for events
event_weeks = [(name, (election_date - date).days // 7) for name, date in events]

# Apply labeling function
election_data['event'] = election_data.apply(
    lambda row: label_event(row, event_weeks, window=2), axis=1
)

# Group by event and subreddit, calculate post counts
activity_by_event = (
    election_data.groupby(['event', 'subreddit'])
    .size()
    .reset_index(name='post_count')
)

print(activity_by_event)

# Visualize activity for events
plt.figure(figsize=(12, 8))
for event_name in activity_by_event['event'].unique():
    event_data = activity_by_event[activity_by_event['event'] == event_name]
    plt.bar(event_data['subreddit'], event_data['post_count'], label=event_name)

plt.title('Activity by Event for Each Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Number of Posts')
plt.legend(title='Events')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- N-Grams Functionality ---
def rm_stopwords_tokenize(text_series):
    stop_words = set(stopwords.words('english'))
    tokenized_list = []
    for text in text_series:
        tokens = word_tokenize(str(text))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        tokenized_list.append(' '.join(filtered_tokens))  # Join tokens back into a string
    return tokenized_list

def compute_ngram_features(data, ngram_range=(1, 2), top_n=10):
    tokenized_data = rm_stopwords_tokenize(data['body'])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(tokenized_data)
    features = vectorizer.get_feature_names_out()
    
    # Convert the sparse matrix X to dense and create ngram_df
    ngram_df = pd.DataFrame(X.todense(), columns=features)
    counts = np.array(X.sum(axis=0)).flatten()
    feature_counts = list(zip(features, counts))
    
    # Sort the features by counts (descending)
    sorted_feature_counts = sorted(feature_counts, key=lambda x: x[1], reverse=True)
    top_ngrams = [feature for feature, count in sorted_feature_counts[:top_n]]
    return top_ngrams

# Compute n-grams for each event
for event_name in activity_by_event['event'].unique():
    event_data = election_data[election_data['event'] == event_name]
    ngrams = compute_ngram_features(event_data, ngram_range=(1, 2), top_n=10)
    print(f"Top n-grams for {event_name}:")
    print(ngrams)
