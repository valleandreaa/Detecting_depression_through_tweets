import pandas as pd

def ExtractorDataset(Nr_set):
    '''
    Extraction of the required dataset and random partition in training set (70%), development set (15%),
    test set (15%). It prints out the dataset.csv
    :param Nr_set: Number of text file for each sentiment [int]
    :return:
    '''
    df = pd.read_csv('data/Tweets_labelled.csv',sep=',', encoding='latin-1', header=None,
                     names=['target', 'ids', 'date', 'flag', 'user', 'text', 'type'])

    negative_df, positive_df = df.groupby(by=df['target'])

    positive_set = positive_df[1].sample(Nr_set, random_state=10)
    negative_set = negative_df[1].sample(Nr_set, random_state=10)

    Nr_train= int(Nr_set*0.7)
    Nr_develop= int(Nr_set*0.15)


    train_set_pos= positive_set.iloc[:Nr_train,]
    train_set_pos['type']='train'
    train_set_neg= negative_set.iloc[:Nr_train,]
    train_set_neg['type']= 'train'
    develop_set_pos=positive_set.iloc[Nr_train:Nr_train+Nr_develop,]

    develop_set_pos['type']='develop'
    develop_set_neg=negative_set.iloc[Nr_train:Nr_train+Nr_develop,]
    develop_set_neg['type']='develop'
    test_set_pos=positive_set.iloc[Nr_train+Nr_develop:,]

    test_set_pos['type']='test'
    test_set_neg=negative_set.iloc[Nr_train+Nr_develop:,]
    test_set_neg['type']='test'

    dataset=pd.concat([train_set_neg,train_set_pos, develop_set_neg,develop_set_pos,
                         test_set_neg,test_set_pos])


    dataset.to_csv('data/dataset.csv')

