# -*- coding: utf-8 -*-
import re
import nltk
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def tokenize(text):
            """
            function to process the messages text, normalization, remove stop words,
            then word lemmatization.
        
            Parameters
            ----------
            text : str
                message text.
        
            Returns
            -------
            cleaned_tokens : str
                cleaned message.
        
            """
            # Define url pattern
            url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

            # Detect and replace urls
            detected_urls = re.findall(url_re, text)
            for url in detected_urls:
                text = text.replace(url, "urlplaceholder")

            # tokenize sentences
            tokens = nltk.word_tokenize(text)
            lemmatizer = nltk.WordNetLemmatizer()

            # save cleaned tokens
            clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

            # remove stopwords
            STOPWORDS = list(set(stopwords.words('english')))
            clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]

            return ' '.join(clean_tokens)

        return pd.Series(X).apply(tokenize).values