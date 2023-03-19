import pandas as pd

# 1. group by label. keep just positive and negative. 50% is control set (positive) and 50% is negative set (negative)
df = pd.read_csv('../data/Tweets_labelled.csv',sep=',', encoding='latin-1', header=None,
                 names=['target', 'ids', 'date', 'flag', 'user', 'text'])

negative_df, positive_df = df.groupby(by=df['target'])
# 2. Random partition: 70% training set, 15% development set, 15% test set (50'000 tweets in total). Each set has to have 50% as control set and 50% for negative set.
Nr_set= 25000
positive_set = positive_df[1].sample(Nr_set, random_state=10)
negative_set = negative_df[1].sample(Nr_set, random_state=10)

Nr_train= int(Nr_set*0.7)
Nr_develop= int(Nr_set*0.15)


train_set_pos= positive_set.iloc[:Nr_train,]
train_set_neg= negative_set.iloc[:Nr_train,]
develop_set_pos=positive_set.iloc[Nr_train:Nr_train+Nr_develop,]
develop_set_neg=negative_set.iloc[Nr_train:Nr_train+Nr_develop,]

test_set_pos=positive_set.iloc[Nr_train+Nr_develop:,]
test_set_neg=negative_set.iloc[Nr_train+Nr_develop:,]

# 3. Output: a csv file for each set
train_set=pd.concat([train_set_neg,train_set_pos])
develop_set=pd.concat([develop_set_neg,develop_set_pos])
test_set=pd.concat([test_set_neg,test_set_pos])

train_set.to_csv('../data/train_set.csv')
develop_set.to_csv('../data/develop_set.csv')
test_set.to_csv('../data/test_set.csv')

