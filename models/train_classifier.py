import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

import dill

from sklearn.base import BaseEstimator, TransformerMixin
import os

cwd = os.getcwd()
cwd_temp = cwd[:-6] # remove models
sys.path.insert(0, cwd_temp)

import custom_transformers.transformer as cus_transformer 
sys.path.insert(0, cwd)

from sklearn.model_selection import GridSearchCV
import joblib


# from custom_transformers.transformer import Tokenizer 


def load_data(database_filepath):
    """
    load data from SQL database, and split it into X, y. Note that the table
    name is set to: DisasterTable.  

    Parameters
    ----------
    database_filepath : str 
        the path to the SQL database.

    Returns
    -------
    X : pandas serie
        messages data used as our model input.
    y : pandas dataframe
        categories used our model labels.
    category_names : str
        list of categories names.

    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterTable', con = engine)
    df = df.dropna()
    X = df["message"]
    category_names = list(df.columns[4:])
    y = df[category_names]
    return X, y, category_names


# class Tokenizer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

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
    # Normilize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize
    tokens = nltk.word_tokenize(text)
    # remove stop words
    tokens = [tok for tok in tokens if tok not in stopwords.words('english')]
    # initiate lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return cleaned_tokens


def build_model():
    """
    function to build a pipeline for multi-output classification using 
    RandomForest classifier.

    Returns
    -------
    pipeline : sklearn object
        scklearn gridSearch optimisation object.

    """
    pipeline = Pipeline([
        ('custom', cus_transformer.Tokenizer()),
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])



    # add grid search optimization
    # parameters can be ploted using pipeline.get_params().
    parameters = {
        'clf__estimator__n_estimators': [20],
        'clf__estimator__max_features': ["auto"],
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    model evaluation using test data. This function plots the results for 
    each label. 

    Parameters
    ----------
    model : sklearn object
        sklearn pipeline.
    X_test : pandas dataframe
        cleaned messages test data.
    Y_test : pandas dataframe
        test ground truth used to compare prediction results.
    category_names : str
        labels names.

    Returns
    -------
    None.

    """
    y_pred = pd.DataFrame(model.predict(X_test), columns= category_names)
    # plot the prediction score of each output
    for col in category_names:
        print("score of {} output".format(col))
        print(classification_report(Y_test[col], y_pred[col]))




def save_model(model, model_filepath):
    """
    function to save the trained model in a pickle format.

    Parameters
    ----------
    model : sklearn object
        trained sklearn pipeline.
    model_filepath : str
        path to save the model. Example: "trained_model.pkl".

    Returns
    -------
    None.

    """
    # use dill instead to avoid pickl load bug
    with open(model_filepath,'wb') as io:
        dill.dump(model,io)
    #joblib.dump(model, model_filepath, compress=5)
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model ...')
        model.fit(X_train, Y_train)
        
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')
        print("Done")

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
