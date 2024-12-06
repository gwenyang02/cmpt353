import pandas as pd
import sys
from scipy.stats import ttest_ind
from scipy.stats import levene


#Q1: Which political posts (Republican or Democratic) tend to get more engagement and is there a significant difference between them?
#Running: use author_data.csv on command line
#Output: engagement.csv and p-values for equal variance and t test

def get_political_leaning(score):
    if score > 0:
        return 'Republican'
    elif score < 0:
        return 'Democratic'
    else:
        return 'Neutral'

#main 
def main(in_data):
    #read in data
    data = pd.read_csv(in_data)
    
    #add column to determine political leaning
    data['political_leaning'] = data['shifted'].apply(get_political_leaning)

    #create a new df that calculates the average score
    score_df = data.groupby('political_leaning').agg({'score': ['mean',
                                'sum', 'max']}).reset_index()

    #output the score_df to see the values of socre mean, sum, and max
    score_df.to_csv('engagement.csv', index = False)

    #create democratic and republican dataframes
    democratic = data[data['political_leaning'] == 'Democratic']
    republican = data[data['political_leaning'] == 'Republican']
    
    #check equal variance
    equal_var = levene(democratic['score'], republican['score'])
    print(f"Equal variance p-value: {equal_var.pvalue}")

    #do t-test and output the p-value
    ttest_result = ttest_ind(democratic['score'], republican['score'], equal_var = False)
    print(f"T-test p-value: {ttest_result.pvalue}")

if __name__ == '__main__':
    in_data = sys.argv[1]
    main(in_data)
