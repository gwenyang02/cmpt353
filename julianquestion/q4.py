import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
data = pd.read_csv('./allactivity.csv')

# Convert 'created_utc' to datetime
data['created_utc'] = pd.to_datetime(data['created_utc'])

# Add weeks to election
election_date = pd.Timestamp('2016-11-08')  # Election date
data['weeks_to_election'] = (election_date - data['created_utc']).dt.days // 7

# Filter for election-related subreddits
election_subreddits = ["The_Donald", "HillaryClinton", "politics", "Conservative", "Democrats"]
election_data = data[data['subreddit'].isin(election_subreddits)]

# Group by subreddit and weeks to election, then calculate activity counts
activity_by_week = (
    election_data.groupby(['subreddit', 'weeks_to_election'])['id']
    .count()
    .reset_index(name='post_count')
)

# Plot activity trends for each subreddit
plt.figure(figsize=(12, 8))
for subreddit in election_subreddits:
    subreddit_data = activity_by_week[activity_by_week['subreddit'] == subreddit]
    plt.plot(subreddit_data['weeks_to_election'], subreddit_data['post_count'], label=subreddit)

# Reverse x-axis (weeks count down to the election)
plt.gca().invert_xaxis()

# Add labels and title
plt.title('Activity Trends for Election-Related Subreddits')
plt.xlabel('Weeks to Election')
plt.ylabel('Number of Posts')
plt.legend(title='Subreddits')
plt.tight_layout()
plt.show()

# Linear regression for overall trend analysis
overall_activity = (
    election_data.groupby('weeks_to_election')['id']
    .count()
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
