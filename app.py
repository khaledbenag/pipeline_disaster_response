import json
import plotly
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import numpy as np
from custom_transformers.transformer import Tokenizer

app = Flask(__name__)

# def tokenize(text):
#     """
#     function to process text 

#     Parameters
#     ----------
#     text : str
#         message to be processed.

#     Returns
#     -------
#     clean_tokens : str
#         processed message.

#     """
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterTable', engine)

# load model
#model = joblib.load("models/classifier.pkl")
import dill
with open("models/classifier.dill",'rb') as io:
    model=dill.load(io)
    
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # plot counts by genre
    
    genre_labels, genre_count = np.unique(df['genre'].values, return_counts=True)
    
    # plot counts by column
    counts = df.iloc[:, 4:].sum()

    # plot how many messages shares the same number of labels
    messages_per_row = df.iloc[:,4:].sum(axis = 1)
    messages_counts,n = np.unique(messages_per_row, return_counts= True)


    graphs = [
        {
            'data': [
                Bar(
                    x=genre_labels,
                    y=genre_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=counts.index,
                    y=counts.values
                )
            ],

            'layout': {
                'title': 'Message Categories distribution',
                'yaxis': {
                    'title': "Count (messages)"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x= messages_counts,
                    y= n
                )
            ],

            'layout': {
                'title': 'Number of messages based on shared labels ',
                'yaxis': {
                    'title': "Count (messages)"
                },
                'xaxis': {
                    'title': "Messages distribution by shared labels"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()