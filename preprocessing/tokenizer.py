import nltk
import string
nltk.download('punkt')



nltk.download('stopwords')
from nltk.corpus import stopwords





# 3.  Stemming:  Stemming is the process of reducing the words to their word stem or root form. The objective of stemming is to reduce related words to the same stem even if the stem is not a dictionary word.
nltk.download('averaged_perceptron_tagger')
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize




# text = "I am running and eating a burger"
# stemmed_text = stem_text(text)
# print(stemmed_text)
# 4. Lemmatization: lemmatization reduces words to their base word, reducing the inflected words properly and ensuring that the root word belongs to the language.
nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize



# text = "I am running and eating a burger"
# lemmatized_text = lemmatize_text(text)
# print(lemmatized_text)

class TextTokenizer:
    def __init__(self, df):
        self.target = df['target']
        self.text = df['text']

    def remove_characters(self):
        '''
        Remove special characters, numbers, puntctuations
        :param text: text message [str]
        :return: cleaned text [str]
        '''



        self.text = self.text.apply(lambda t:" ".join(filter(lambda word: not word.startswith(('@', 'http')), t.split())))

        # Remove punctuation
        self.text = self.text.apply(lambda t: t.translate(str.maketrans('', '', string.punctuation)))

        # Remove punctuation
        self.text = self.text.apply(lambda t: t.translate(str.maketrans('', '', string.punctuation)))
        # (str.maketrans('', '', string.punctuation))

        # Remove digits
        self.text =self.text.apply(lambda t: t.translate((str.maketrans('', '', string.digits))))

        # Remove extra whitespaces
        self.text = self.text.apply(lambda t: ' '.join(t.split()))
        # text = ' '.join(text.split())

        # # Tokenize the text into words
        # tokens = nltk.word_tokenize(text)

        # # Join the words back into a string
        # text = ' '.join(tokens)



    def remove_stop_words(self):

        # Tokenize the text into words
        tokens_list = self.text.apply(lambda t: nltk.word_tokenize(t))


        # Get the English stop words list
        stop_words = set(stopwords.words('english'))

        # Remove the stop words from the tokens
        for id, tokens in enumerate(tokens_list):
            filtered_token =([token for token in tokens if token.lower() not in stop_words])
            # filtered_tokens.append(' '.join(filtered_token))
            self.text[id]=' '.join(filtered_token)

    def stem_text(self):
        # Tokenize the text into words
        tokens_list = self.text.apply(lambda t: nltk.tokenize.word_tokenize(t))

        # Create a Porter stemmer object
        stemmer = nltk.stem.PorterStemmer()

        # Apply stemming to each word in the tokens list
        for id, tokens in enumerate(tokens_list):
            stemmed_tokens = [stemmer.stem(word) for word in tokens]
            # filtered_tokens.append(' '.join(filtered_token))
            self.text[id] = ' '.join(stemmed_tokens)

    def lemmatize_text(self):
        # Tokenize the text into words

        tokens_list = self.text.apply(lambda t: nltk.tokenize.word_tokenize(t))

        # Create a WordNet lemmatizer object
        # lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()

        for id, tokens in enumerate(tokens_list):
            # Apply lemmatization to each word in the tokens list
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

            # Join the lemmatized tokens back into a string
            self.text[id] = ' '.join(lemmatized_tokens)

        print(self.text)

    def df_cleaned_text(self, df):
        return df.assign(text_cleaned=self.text)




# output: a dataframe with just the cleaned data
