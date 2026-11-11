"""CS5720 Home Assignment 4, Part II Q1: basic NLP preprocessing."""
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

def preprocess(sentence):
    tokens = wordpunct_tokenize(sentence)
    filtered = [t for t in tokens if t.isalpha() and t.lower() not in STOPWORDS]
    stemmer = PorterStemmer()
    stems = [stemmer.stem(t.lower()) for t in filtered]
    print('Original Tokens:', tokens)
    print('Tokens Without Stopwords:', filtered)
    print('Stemmed Words:', stems)
    return tokens, filtered, stems

if __name__ == '__main__':
    preprocess('NLP techniques are used in virtual assistants like Alexa and Siri.')
