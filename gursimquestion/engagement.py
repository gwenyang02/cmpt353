import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import ttest_ind

#Q1: Which political posts (Republican or Democratic) tend to get more engagement and is there a significant difference between them?
#Input: use author_data.csv to run it
#Output: engagement.csv that shows a break down of scores (mean, sum, max) and t-test pvalue

def get_political_leaning(score):
    if score > 0:
        return 'Republican'
    else:
        return 'Democratic'

def do_ttest(data):
    democratic = data[data['political_leaning'] == 'Democratic']
    republican = data[data['political_leaning'] == 'Republican']

    score_ttest = ttest_ind(democratic['score'], republican['score'], equal_var = False)

    return score_ttest


#main 
def main(in_data):
    #read in data
    data = pd.read_csv(in_data)
    
    #add column to determine political leaning
    data['political_leaning'] = data['sentiment'].apply(get_political_leaning)

    #create a new df that calculates the average score
    score_df = data.groupby('political_leaning').agg({'score': ['mean',
                                'sum', 'max']}).reset_index()

    score_df.to_csv('engagement.csv', index = False)

    ttest_result = do_ttest(data)

    print(ttest_result.pvalue)


if __name__ == '__main__':
    in_data = sys.argv[1]
    main(in_data)
