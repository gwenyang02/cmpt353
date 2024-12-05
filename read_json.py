import pandas as pd
import numpy as np
import glob
import os


def read_json_df(file_list):
    '''
    read multiple json files into one dataframe
    source: https://stackoverflow.com/questions/57067551/
    :param file_list:
    :return: a dataframe
    '''
    dfs = []  # an empty list to store the data frames
    for file in file_list:
        data = pd.read_json(file, lines=True)  # read data frame from json file
        dfs.append(data)  # append the data frame to the list
    # concatenate all the data frames in the list
    temp = pd.concat(dfs, ignore_index=True)
    return temp

def main():
    all_activity = 'reddit-us-election-expanded2/all_activity/'
    comments = 'reddit-us-election-expanded2/election_comments'
    posts = 'reddit-us-election-expanded2/election_submissions'

    # get list of all .json.gz files in each folder
    all_list = glob.glob(os.path.join(all_activity, '*.json.gz'))
    comments_list = glob.glob(os.path.join(comments, '*.json.gz'))
    posts_list = glob.glob(os.path.join(posts, '*.json.gz'))

    # store data from each folder into their own dataframes
    all_activity_df = read_json_df(all_list)
    comments_df = read_json_df(comments_list)
    posts_df = read_json_df(posts_list)

    print(comments_df)
    auths = pd.concat([comments_df['author'],posts_df['author']]).unique()
    auths = auths[auths != '[deleted]']
    auths
    #all_activity_df.to_csv('all_activity.csv')
    #comments_df.to_csv('comments.csv')
    #posts_df.to_csv('posts.csv')

if __name__ == "__main__":
    main()