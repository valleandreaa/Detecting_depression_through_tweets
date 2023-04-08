import nltk
import string
import pandas as pd
nltk.download('punkt')
from nltk.corpus import words
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import opinion_lexicon
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
nltk.download('opinion_lexicon')
nltk.download('words')

class TextTokenizer:
    '''
    Text Tokenizer
    :param target: label [str]
    :param text: text document [str]
    '''
    def __init__(self, df):
        self.target = df['target']
        self.text = df['text']

    def remove_characters(self):
        '''
        Remove special characters, numbers, puntctuations

        :return: cleaned text [str]
        '''

        # Remove tags and links
        self.text = self.text.apply(lambda t:" ".join(filter(lambda word: not word.startswith(('@', 'http')), t.split())))

        # Remove punctuation
        self.text = self.text.apply(lambda t: t.translate(str.maketrans('', '', string.punctuation)))

        # Remove digits
        self.text =self.text.apply(lambda t: t.translate((str.maketrans('', '', string.digits))))

        # Remove extra whitespaces
        self.text = self.text.apply(lambda t: ' '.join(t.split()))


    def sentiment_dictionary(self):
        '''
        Generation of the sentiment dictionary
        :return: positive words [list], negative words [list]
        '''
        positive_wds = set(opinion_lexicon.positive())
        negative_wds = set(opinion_lexicon.negative())
        return positive_wds, negative_wds

    def remove_stop_words(self):
        '''
        Remove stop words
        :return:
        '''
        # Tokenize the text into words
        tokens_list = self.text.apply(lambda t: nltk.word_tokenize(t))

        # English stop words list
        stop_words = set(stopwords.words('english'))

        # Remove the stop words from the tokens
        for id, tokens in enumerate(tokens_list):
            filtered_token =([token for token in tokens if token.lower() not in stop_words])
            # filtered_tokens.append(' '.join(filtered_token))
            self.text[id]=' '.join(filtered_token)


    def stem_text(self):
        '''
        Stemming of a text
        :return:
        '''
        # Tokenize the text into words
        tokens_list = self.text.apply(lambda t: nltk.tokenize.word_tokenize(t))

        # Snowball stemmer
        stemmer = nltk.stem.SnowballStemmer('english')

        # stemming of the documents
        for id, tokens in enumerate(tokens_list):
            stemmed_tokens = [stemmer.stem(word) for word in tokens]
            self.text[id] = ' '.join(stemmed_tokens)

    def is_word(self, word):
        '''
        Existance of a words
        :param word: word [str]
        :return: exist [bool]
        '''
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                if word == lemma.name():
                    return True
        return False
    def remove_non_existing_words(self):
        '''
        Remove not existing words
        :return:
        '''
        # Tokenize the text into words
        tokens_list = self.text.apply(lambda t: nltk.tokenize.word_tokenize(t))

        # Remove non-existing words from the tokens
        for id, tokens in enumerate(tokens_list):
            existing_tokens = [token for token in tokens if self.is_word(token)]
            self.text[id] = ' '.join(existing_tokens)

    def lemmatize_text(self):
        '''
        Lemmatization of a text
        :return:
        '''
        # Tokenize the text into words
        tokens_list = self.text.apply(lambda t: nltk.tokenize.word_tokenize(t))

        # Create a WordNet lemmatizer object
        lemmatizer = nltk.stem.WordNetLemmatizer()

        # Lemmatize words from the tokens
        for id, tokens in enumerate(tokens_list):
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
            self.text[id] = ' '.join(lemmatized_tokens)

    def df_cleaned_text(self, df):
        '''
        Assign dataset from object
        :param df: dataset [df]
        :return: updated dataset [df]
        '''
        return df.assign(text_cleaned=self.text)


def PreProcessing():

    df = pd.read_csv('data/%s.csv' % ('dataset'))
    token = TextTokenizer(df)
    token.remove_characters()
    token.stem_text()
    token.remove_stop_words()
    token.lemmatize_text()
    df = token.df_cleaned_text(df)
    df[['target', 'ids', 'user', 'text','type', 'text_cleaned']].to_csv('data/dataset_clean.csv', index=False)

