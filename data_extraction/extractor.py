import pandas as pd

def ExtractorDataset(Nr_set):
    '''
    Extraction of the required dataset and random partition in training set (70%), development set (15%),
    test set (15%). It prints out the dataset.csv
    :param Nr_set: Number of text file for each sentiment [int]
    :return:
    '''
    # import of the original dataset
    df = pd.read_csv('data/Tweets_labelled.csv',sep=',', encoding='latin-1', header=None,
                     names=['target', 'ids', 'date', 'flag', 'user', 'text', 'type'])
    # replace the target from 4 to 1
    df.replace(to_replace = 4, value = 1, inplace=True)
    # split the dataset in positive and negative tweets
    negative_df, positive_df = df.groupby(by=df['target'])
    # sampling of the datasets
    positive_set = positive_df[1].sample(Nr_set, random_state=10)
    negative_set = negative_df[1].sample(Nr_set, random_state=10)

    # Slicing of the 80% of the dataset for training the models
    Nr_train= int(Nr_set*0.8)


    # Training dataset
    train_set_pos= positive_set.iloc[:Nr_train,]
    train_set_pos['type']='train'
    train_set_neg= negative_set.iloc[:Nr_train,]
    train_set_neg['type']= 'train'

    # Test dataset
    test_set_pos=positive_set.iloc[Nr_train:,]
    test_set_pos['type']='test'
    test_set_neg=negative_set.iloc[Nr_train:,]
    test_set_neg['type']='test'

    # combination of the dataset
    dataset=pd.concat([train_set_neg,train_set_pos, test_set_neg, test_set_pos])

    # export dataset
    dataset.to_csv('data/dataset.csv')

