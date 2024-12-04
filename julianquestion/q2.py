# like for example is a pro republican comment going to get more positive score on a gun rights subreddit? 
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
data = pd.read_csv('nongroupedsentiment.csv')

#just do this for many different subredit names
gunrights_data = data[data['subreddit'] == 'immigration']

# make category sentiment >0 or <0
gunrights_data['sentiment_category'] = gunrights_data['sentiment'].apply(
    lambda x: 'pro_republican' if x >= 0 else 'pro_democrat'
)

gunrights_data['score_category'] = gunrights_data['score'].apply(
    lambda x: 'positive' if x >= 0 else 'negative'
)

gunrights_data['score_category'] = gunrights_data['score'].apply(categorize_score)

import pandas as pd

contingency_table = pd.crosstab(
    gunrights_data['sentiment_category'],
    gunrights_data['score_category']
)

print(contingency_table)


chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-squared Test Results")
print(f"Chi2 Statistic: {chi2}")
print(f"P-value: {p}")

#weird unexpected results with sentiment analysis tbh 