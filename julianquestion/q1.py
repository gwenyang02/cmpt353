# plotting for every subreddit, the distributions of sentiment analysis
# and do an ANOVA on that for the data i just collected 
# Could indicate if some subreddits lean more to one or the other democrat/republican
# this might be better if there's more subreddits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

pd.set_option('display.max_columns', None) #added to view all cols

# Load the data
# nongroupedsentiall.csv uses allactivitysmall2 (comments and posts)
# nongroupedsentiposts.csv uses allsubsmall2 (posts) only
data = pd.read_csv('./allcommentsbigwithsentiment.csv')

data = data[data['shifted'] != 0]
# Get the count of comments per subreddit
subreddit_counts = data['subreddit'].value_counts()
# filter for subreddits with more than 40 counts
# by CLT, assuming data is normally distributed
data =  data.groupby('subreddit').filter(lambda x: len(x) > 40)
print(subreddit_counts)

data2 = data[data['subreddit']=="The_Donald"]
print(data2.head(10))

# Set the figure size
plt.figure(figsize=(12, 8))

# Create a boxplot
sns.boxplot(x='subreddit', y='shifted', data=data)

# Rotate x-axis labels for readability
plt.xticks(rotation=90)

# Set titles and labels
plt.title('Distribution of Sentiment Scores per Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Sentiment Score')
# Show the plot
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.violinplot(x='subreddit', y='shifted', data=data)
plt.xticks(rotation=45, fontsize=10, ha='right')
plt.title('Distribution of Sentiment Scores per Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Sentiment Score')
# Show the plot
plt.tight_layout()
plt.show()


groups = []

# Group the data by subreddit
subreddit_grouped = data.groupby('subreddit')

for name, group in subreddit_grouped:
    sentiment_scores = group['shifted'].values
    groups.append(sentiment_scores)

anova_result = f_oneway(*groups)
print("ANOVA result:")
print(f" p-value: {anova_result.pvalue}")

# Perform Tukey HSD test for pairwise comparison
# Create a new DataFrame with 'shifted' and 'subreddit' columns for Tukey HSD
tukey_data = data[['shifted', 'subreddit']]

tukey = pairwise_tukeyhsd(endog=tukey_data['shifted'], groups=tukey_data['subreddit'], alpha=0.05)

# Display Tukey HSD results
print("\nTukey HSD results:")
print(tukey)

#visualizing
tukey.plot_simultaneous(figsize=(10, 6))
plt.title('Tukey HSD Pairwise Comparisons')
plt.show()


# Calculate the percentage of rows where the sentiment differs from the shifted value
total_rows = len(data)
different_rows = len(data[data['sentiment'] != data['shifted']])

# Calculate the percentage
percentage_different = (different_rows / total_rows) * 100

# Print the result
print(f"Percentage of rows where sentiment differs from shifted: {percentage_different:.2f}%")


mean_sentiment_per_subreddit = data.groupby('subreddit')['shifted'].mean()

# Sort by the most positive sentiment
most_positive_subreddits = mean_sentiment_per_subreddit.sort_values(ascending=False)

# Output the most positive subreddits
print("Most Positive Subreddits:")
print(most_positive_subreddits.head(10))  # Show top 5 most positive subreddits


#fact checking
subreddits_of_interest = ["prochoice", "healthcare", "The_Donald", "Republican", "Liberal"]
filtered_data = data[data['subreddit'].isin(subreddits_of_interest)]
output_columns = ['subreddit', 'body', 'sentiment', 'shifted', 'score']
filtered_data[output_columns].to_csv('factchecking.csv', index=False)
print(f"Exported sentiment analysis data for the following subreddits to 'subreddit_sentiment_analysis.csv': {', '.join(subreddits_of_interest)}")
