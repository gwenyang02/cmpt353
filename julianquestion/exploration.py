#rows are authors, cols are features incluing:
# ngrams data, subreddits one hot encoded, subreddits with averages for each
# sentiment min max and mean and most_common_hour of posting(should this be changed to 3 hours)? 

#got all this feature data from running the code on the spark data

# i also have reddit comment data straight from spark i can work with and write another file to manipulate 

#interesting question I want to ask: 
# is the average score for a given comment on a subreddit dependant of the commment's political leaning/sentiment analysis? 
# potentially a visualization for this
# like for example is a pro republican comment going to get more positive score on a gun rights subreddit? 
# potentially chi2 test 
#-------------------------- Q1









#another interesting question i can ask given i have access to this redit data for 2016 political subreddits mostly some are 
# others like gun control subreddits or abortion subreddits 

#author,id,body,subreddit,subreddit_id,created_utc,score,year,month

#potential other questions: does the time of day impact subreddit activity? could this be tied to timezones?

#considering its the US could look at posting on night time hours or odd hours for most of the us to infer if people from
#outside of the US are commenting

#another idea #-------------
# do a users sentiment change over time? can we chart that nearer to the elction do they become stronger sentiment comments?
# do people who comment a lot/more activity on a given subreddit have stronger sentiment analysis 


