import pandas as pd

# Base
from IPython.display import YouTubeVideo
# Object from Base Python
from collections import Counter

# Plotting
import squarify
import matplotlib.pyplot as plt
import seaborn as sns

# NLP Libraries
import spacy
from spacy.tokenizer import Tokenizer
from nltk.stem import PorterStemmer
nlp = spacy.load("en_core_web_lg")
nlp.Defaults.stop_words

# Read data form URL
url = "https://raw.githubusercontent.com/LambdaSchool/DS-Unit-4-Sprint-1-NLP/master/module1-text-data/data/yelp_coffeeshop_review_data.csv"
shops = pd.read_csv(url)

#clean up data
shops['date'] = shops['full_review_text'].apply(lambda x: x.split()[0])
shops['review'] = shops['full_review_text'].apply(lambda x: " ".join(x.split()[1:]))


# Tokenizer
STOP_WORDS = nlp.Defaults.stop_words.union(["it's", '1', "i'm", "i've", 'place', "-"])
tokenizer = Tokenizer(nlp.vocab)
tokens = []
""" tokens w/o stopwords"""
for doc in tokenizer.pipe(shops['full_review_text'], batch_size=500):
    doc_tokens = []
    for token in doc:
        if (token.text.lower() not in STOP_WORDS) & (token.is_punct == False):
            doc_tokens.append(token.text.lower())
    tokens.append(doc_tokens)
shops['tokens'] = tokens

# View Counts by Rating 
shops.loc[(shops.star_rating == ' 5.0 star rating ') | (shops.star_rating == ' 4.0 star rating '),
          'rating'] = 'good'
shops.loc[(shops.star_rating == ' 3.0 star rating ') | (shops.star_rating == ' 2.0 star rating ') | (shops.star_rating == ' 1.0 star rating '),
          'rating'] = 'bad'

