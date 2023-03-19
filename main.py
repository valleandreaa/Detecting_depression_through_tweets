from preprocessing.tokenizer import TextTokenizer
import pandas as pd
dataset='test'

df = pd.read_csv('data/%s_set.csv' %dataset)
def main(df):
    token = TextTokenizer(df)

    token.remove_characters()
    token.remove_stop_words()
    token.stem_text()
    token.lemmatize_text()
    df = token.df_cleaned_text(df)
    df[['target', 'ids',  'user', 'text', 'text_cleaned' ]].to_csv('data/%s_set_clean.csv' %dataset, index= False)


if __name__ == "__main__":
    main(df)