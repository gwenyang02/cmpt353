# like for example is a positive comment going to get more positive score on a subreddit?
# potentially chi2 test 
import pandas as pd
from scipy.stats import chi2_contingency

def categorize_score(x):
    if x < -10:
        return 'very_negative'
    elif -10 <= x < 0:
        return 'slightly_negative'
    elif x == 0:
        return 'neutral'
    elif 0 < x <= 10:
        return 'slightly_positive'
    else:
        return 'very_positive'

#read csv
data = pd.read_csv('../csvfiles/sentiment_no_polarity.csv')

#just do this for many different subredit names
subreddit_name = 'The_Donald'
gunrights_data = data[data['subreddit'] == subreddit_name]

# make category sentiment >0 or <0
gunrights_data['sentiment_category'] = gunrights_data['sentiment'].apply(
    lambda x: 'positive' if x >= 0 else 'negative'
)

gunrights_data['score_category'] = gunrights_data['score'].apply(categorize_score)


contingency_table = pd.crosstab(
    gunrights_data['sentiment_category'],
    gunrights_data['score_category']
)

print(contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-squared Test Results")
print(f"Subreddit: {subreddit_name}")
print(f"Chi2 Statistic: {chi2}")
print(f"P-value: {p}")

#weird unexpected results with sentiment analysis tbh 