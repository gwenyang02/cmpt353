import pandas as pd

#read in data
data = pd.read_csv('non_group_sent.csv')

#remove deleted authors
data = data[data['author'] != '[deleted]']

#remove sentiment zero
#data = data[data['shifted'] != 0]

#get specific columns
data = data[['author', 'score', 'shifted']]
    
#group by author
data = data.groupby('author').agg({'shifted': 'mean', 'score': 'mean'}).reset_index()

#turn into a csv for the engagement question
data.to_csv('author_data.csv', index = False)
