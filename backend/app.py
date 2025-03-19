import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'yelp.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    park_names = []
    park_reviews = []
    for entry in data:
        reviews = []
        park_names.append(entry['name'])
        for review in entry['reviews']:
            reviews.append(review)
    parks_df = pd.DataFrame({'Park Name' : park_names, 'Reviews' : reviews})

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(parks_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['name'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['name', 'state']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)