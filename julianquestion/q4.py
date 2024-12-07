import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
nltk.download('punkt')

data = pd.read_parquet('./allactivity3.parquet')

data['created_utc'] = pd.to_datetime(data['created_utc'])

# Add weeks to election
election_date = pd.Timestamp('2016-11-08')  # Election date
data['weeks_to_election'] = (election_date - data['created_utc']).dt.days // 7

# Filter uneeded/complicated subreddits that add confusion
election_subreddits = ["The_Donald", "AmericanPolitics", "Republican", "politics", "democrats", "prochoice", "immigration", "guncontrol", "environment", "immigration", "healthcare", "economics"]
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
    
    smoothed = lowess(y, x, frac=0.2)  
    
    plt.plot(smoothed[:, 0], smoothed[:, 1], label=f'{subreddit} (smoothed)', linewidth=2)

# Reverse x-axis counting down to election
plt.gca().invert_xaxis()

#labels 
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
plt.gca().invert_xaxis()  
plt.title('Overall Activity Trend Across Election-Related Subreddits')
plt.xlabel('Weeks to Election')
plt.ylabel('Total Posts')
plt.legend()
plt.tight_layout()
plt.show()

#classification of a a subreddits comments aggregated utilizng activity and ngrams as features to predict event labels
#function for classification
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
    ("Election Day", pd.Timestamp('2016-11-08')),
    ("Iowa Caucuses", pd.Timestamp('2016-02-01')),  # First major contest in the primaries
    ("Super Tuesday", pd.Timestamp('2016-03-01')),  # Key day in the primary elections
    ("Pulse Nightclub Shooting", pd.Timestamp('2016-06-12')),  # Major gun control and LGBTQ+ rights debate
    ("Paris Agreement Signing", pd.Timestamp('2016-04-22')),  # Global environmental milestone
    ("Trump’s Border Wall Speech", pd.Timestamp('2016-08-31')),  # Immigration-related speech
    ("First Presidential Debate", pd.Timestamp('2016-09-26')),  # Highly watched, pivotal election moment
    ("Access Hollywood Tape Release", pd.Timestamp('2016-10-07')),  # Major controversy impacting Republicans
    ("Third Presidential Debate", pd.Timestamp('2016-10-19')),  # Trump reiterates stance on abortion
]

# Sort events chronologically
events = sorted(
    events, 
    key=lambda x: pd.Timestamp(x[1])  
)

# Add weeks to election for events
event_weeks = [(name, (election_date - date).days // 7) for name, date in events]

# Apply labeling function
election_data = election_data.copy()
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

# Prepare data for timeline plot
activity_by_week = (
    election_data.groupby(['weeks_to_election', 'subreddit'])
    .size()
    .reset_index(name='post_count')
)

plt.figure(figsize=(14, 8))

# activity for each subreddit
for subreddit in election_subreddits:
    subreddit_data = activity_by_week[activity_by_week['subreddit'] == subreddit]
    plt.plot(
        subreddit_data['weeks_to_election'], 
        subreddit_data['post_count'], 
        label=subreddit, linewidth=2
    )

# Add vertical markers for events
for event_name, event_week in event_weeks:
    plt.axvline(x=event_week, color='red', linestyle='--', alpha=0.7)
    plt.text(event_week, plt.gca().get_ylim()[1] * 0.9, event_name, rotation=90, fontsize=8, color='red')

plt.gca().invert_xaxis()

plt.title('Timeline of Subreddit Activity with Events')
plt.xlabel('Weeks to Election')
plt.ylabel('Number of Posts')
plt.legend(title='Subreddits', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- N-Grams Functionality ---
def rm_stopwords_tokenize(text_series):
    stop_words = set(stopwords.words('english'))
    tokenized_list = []
    for text in text_series:
        tokens = word_tokenize(str(text))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        tokenized_list.append(' '.join(filtered_tokens))  
    return tokenized_list

def compute_ngram_features(data, ngram_range=(1, 2), top_n=10):
    tokenized_data = rm_stopwords_tokenize(data['body'])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(tokenized_data)
    features = vectorizer.get_feature_names_out()
   
    ngram_df = pd.DataFrame(X.todense(), columns=features)
    counts = np.array(X.sum(axis=0)).flatten()
    feature_counts = list(zip(features, counts))
    
    sorted_feature_counts = sorted(feature_counts, key=lambda x: x[1], reverse=True)
    top_ngrams = [feature for feature, count in sorted_feature_counts[:top_n]]
    return top_ngrams

# Compute n-grams for each event
for event_name in activity_by_event['event'].unique():
    event_data = election_data[election_data['event'] == event_name]
    ngrams = compute_ngram_features(event_data, ngram_range=(1, 2), top_n=10)
    print(f"Top n-grams for {event_name}:")
    print(ngrams)

#------- needed functions
def prepare_clustering_features(data, vectorizer=None):
    tokenized_data = rm_stopwords_tokenize(data['body'])
    if not vectorizer:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_tfidf = vectorizer.fit_transform(tokenized_data)
    return X_tfidf, vectorizer

# Calculate total subreddit activity within a 2-week window
def compute_activity_metrics(data, event_weeks, window=2):
    activity_metrics = []
    for name, event_week in event_weeks:
        # Filter data for the event window
        event_window = data[
            (data['weeks_to_election'] >= event_week - window) &
            (data['weeks_to_election'] <= event_week + window)
        ]
        
        # Calculate total activity and activity ratio
        activity_count = (
            event_window.groupby('subreddit').size().reset_index(name='activity_count')
        )
        total_activity = activity_count['activity_count'].sum()
        avg_activity = data.groupby('subreddit').size().mean()
        activity_ratio = total_activity / avg_activity if avg_activity > 0 else 0
        
        # Store metrics
        activity_metrics.append({
            "event": name,
            "total_activity": total_activity,
            "activity_ratio": activity_ratio
        })
    
    return pd.DataFrame(activity_metrics)

# Compute activity metrics for all events
activity_metrics = compute_activity_metrics(election_data, event_weeks)

# Prepare the dataset for supervised learning
def prepare_classification_data(data, event_weeks, top_n_ngrams=1000, window=2):
    # extract top ngrams
    tokenized_data = rm_stopwords_tokenize(data['body'])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=top_n_ngrams)
    X_tfidf = vectorizer.fit_transform(tokenized_data)

    #converting to a dense matrix for PCA
    X_dense = X_tfidf.toarray()

    pca = PCA(n_components=100)
    X_reduced = pca.fit_transform(X_dense)
    
    # One-hot encode subreddit names
    subreddit_features = pd.get_dummies(data['subreddit']).values
    
    # Compute activity metrics
    data = data.copy()  
    data['activity_level'] = data.groupby('subreddit')['weeks_to_election'].transform('count')
    data['activity_ratio'] = data['activity_level'] / data.groupby('subreddit')['activity_level'].transform('mean')
    
    # Combine all features
    X_combined = np.hstack((X_reduced, subreddit_features, data[['activity_level', 'activity_ratio']].values))
    
    # Target Labels
    data['event'] = data.apply(lambda row: label_event(row, event_weeks, window=window), axis=1)
    y = data['event']
    
    return X_combined, y, vectorizer


# classification_data = election_data[election_data['event'] != 'No Event'] to filter by noevent data
X, y, vectorizer = prepare_classification_data(election_data, event_weeks)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=100, max_depth = 6)
                         
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy:.2f}")

test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")