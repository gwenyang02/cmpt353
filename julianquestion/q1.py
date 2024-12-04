# plotting for every subreddit, the distributions of sentiment analysis
# and do an ANOVA on that for the data i just collected 
# Could indicate if some subreddits lean more to one or the other democrat/republican
# this might be better if there's more subreddits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
# nongroupedsentiment2.csv uses allactivity
# nongroupedsentiment3.csv uses posts only
data = pd.read_csv('nongroupedsentiment3.csv')

# Get the count of comments per subreddit
subreddit_counts = data['subreddit'].value_counts()
# filter for subreddits with more than 40 counts
# by CLT, assuming data is normally distributed
data =  data.groupby('subreddit').filter(lambda x: len(x) > 40)
print(subreddit_counts)

# Set the figure size
plt.figure(figsize=(12, 8))

# Create a boxplot
sns.boxplot(x='subreddit', y='sentiment', data=data)

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
sns.violinplot(x='subreddit', y='sentiment', data=data)
plt.title('Distribution of Sentiment Scores per Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Sentiment Score')
# Show the plot
plt.tight_layout()
plt.show()

#doing an ANOVA t test on sentiment analysis for all differnet subreddits
#looks weird the graph kinda think the sentiment analysis is messed up 










