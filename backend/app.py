import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import re

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'yelp.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    park_dict = {}
    for entry in data:
        park_dict[entry['name']] = entry['reviews']

app = Flask(__name__)
CORS(app)

def tokenize(text):
    """
    Function to tokenize the given input string.
    """
    return re.findall(r'[a-z]+', text.lower())

def aggregate_reviews(park_dict) -> dict[str, dict[str, int]]:
    """
    Function to create aggregated term frequency dictionaries for the reviews
    associated with each distinct amusement park in `reviews_df`.
    """
    park_token_dict = {}
    for park, reviews in park_dict.items():
        token_dict = {}
        for review in reviews:
            tokens = tokenize(review['text'])
            for token in tokens:
                if token_dict.get(token) is None:
                    token_dict[token] = 1
                else:
                    token_dict[token] += 1
        park_token_dict[park] = token_dict
    return park_token_dict

def find_similar_parks(query_tokens, park_token_dict) -> dict[str, int]:
    """
    Function to create and return a dictionary that maps amusement park names to 
    the cosine similarity scores between their associated tokenized reviews and
    the input tokenized query. Returns the dictionary sorted in descending
    cosine similarity score order.
    """
    scores = {}
    n_query_tokens = len(query_tokens)
    for park, park_tokens in park_token_dict.items():
        dot_product = 0
        common_tokens = 0     # variable to store number of tokens in common
                              # between the query and the reviews for this park  
        for token in query_tokens:
            if park_tokens.get(token) is not None:
                dot_product += park_tokens.get(token) * query_tokens.get(token)
                common_tokens += 1
        total_tokens = n_query_tokens + len(park_tokens) - common_tokens
        scores[park] = dot_product / total_tokens
    scores = {park : score for park, score in sorted(scores.items, lambda x:x[1], 
                                                     reverse=True)}
    return scores

# Sample search using json with pandas
def json_search(query):
    query_tokens = {}
    for token in tokenize(query):
        if query_tokens.get(token) is None:
            query_tokens[token] = 1
        else:
            query_tokens[token] += 1
    park_token_dict = aggregate_reviews(park_dict)
    similar_parks = find_similar_parks(query_tokens, park_token_dict)
    return similar_parks.to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)