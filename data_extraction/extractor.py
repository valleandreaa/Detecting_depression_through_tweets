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
    df.replace(to_replace = 4, value = 1, inplace=True)
    negative_df, positive_df = df.groupby(by=df['target'])
    positive_df[1]['traget'] = 1
    positive_set = positive_df[1].sample(Nr_set, random_state=10)
    negative_set = negative_df[1].sample(Nr_set, random_state=10)

    Nr_train= int(Nr_set*0.8)



    train_set_pos= positive_set.iloc[:Nr_train,]
    train_set_pos['type']='train'
    train_set_neg= negative_set.iloc[:Nr_train,]
    train_set_neg['type']= 'train'

    develop_set_pos=positive_set.iloc[Nr_train:,]
    develop_set_pos['type']='test'
    develop_set_neg=negative_set.iloc[Nr_train:,]
    develop_set_neg['type']='test'




    dataset=pd.concat([train_set_neg,train_set_pos, develop_set_neg,develop_set_pos])


    dataset.to_csv('data/dataset.csv')

