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
        park_dict[entry['business_id']] = {
            'name': entry['name'],
            'state': entry['state'],
            'reviews': entry['reviews']
        }
        # if good for kids is in attributes, add it to the dictionary
        if entry['attributes'] and 'GoodForKids' in entry['attributes']:
            park_dict[entry['business_id']]['good_for_kids'] = entry['attributes']['GoodForKids']
        else:
            park_dict[entry['business_id']]['good_for_kids'] = "False"
app = Flask(__name__)
CORS(app)

def tokenize(text):
    """
    Function to tokenize the given input string.
    """
    return re.findall(r'[a-z]+', text.lower())

def num_docs(parks) -> int:
    """
    Function to determine the total number of amusement park reviews contained
    in yelp.json.
    """
    sum = 0
    for attributes in parks.values():
        sum += len(attributes['reviews'])
    return sum

def get_idf_values(parks, num_docs) -> dict[str, int]:
    """
    Function to create a dictionary mapping every term that appears in at least
    one park review to the total number of reviews in which the term appears
    across the entirety of yelp.json.
    """
    df_dict = {}
    for attributes in parks.values():
        for review in attributes['reviews']:
            tokens = set(tokenize(review['text']))
            for token in tokens:
                if df_dict.get(token) is None:
                    df_dict[token] = 1
                else:
                    df_dict[token] += 1
    for token, count in df_dict.items():
        df_dict[token] = num_docs / count
    return df_dict

def build_inverted_index(parks) -> dict[str, list[(str, int)]]:
    """
    Function to create an inverted index dictionary for all unique terms across
    the entire set of amusement park reviews. The dictionary maps each unique
    term to a list of tuples, where the first value in the tuple is a park name,
    and the second value represents the number of times that the token appears
    in a review for that park.
    """
    inverted_dict = {}
    for park, attributes in parks.values():
        for review in attributes['reviews']:
            for token in review:
                if inverted_dict.get(token) is None:
                    inverted_dict[token] = [(park, 1)]
                else:
                    last_pair = inverted_dict[token][len(inverted_dict[token] - 1)]
                    # check if the token has already been found in a review for
                    # this park
                    if last_pair[0] == park:
                        inverted_dict[token][len(inverted_dict) - 1] = \
                                                (last_pair[0], last_pair[1] + 1)
                    else:
                        inverted_dict[token].append((park, 1))
    return inverted_dict

def aggregate_reviews(parks, idf_dict) -> dict[str, dict[str, int]]:
    """
    Function to create, for each distinct amusement park in the input dictionary,
    a dictionary mapping terms that appear in that park's reviews to their 
    associated frequency values.
    """
    park_token_dict = {}
    for park, attributes in parks.items():
        token_dict = {}
        for review in attributes['reviews']:
            tokens = tokenize(review['text'])
            for token in tokens:
                if token_dict.get(token) is None:
                    token_dict[token] = 1
                else:
                    token_dict[token] += 1
        park_token_dict[park] = token_dict
    return park_token_dict

def calculate_average_ratings(parks) -> dict[str, int]:
    """
    Function to calculate each distinct park's average rating, out of five stars,
    across all of its associated reviews. Returns a dictionary mapping business 
    ids to their average ratings.
    """
    rating_dict = {}
    for park, attributes in parks.items():
        rating_sum = 0
        review_count = 0
        for review in attributes['reviews']:
            rating_sum += review['stars']
            review_count += 1
        rating_dict[park] = rating_sum / review_count
    return rating_dict

def calculate_similarities(query_tokens, inverted_dict, idf_dict) -> dict[str, int]:
    """
    Function to create and return a dictionary that maps business ids to 
    the cosine similarity scores between their associated tokenized reviews and
    the input tokenized query.
    """
    scores = {}
    for token, frequency in query_tokens:
        for park, count in inverted_dict[token]:
            # initialize or update score accumulator
            if scores.get(park) is None:
                scores[park] = frequency * idf_dict[token] \
                                * count * idf_dict[token]
            else:
                scores[park] += frequency * idf_dict[token] \
                                * count * idf_dict[token]
    return scores

def apply_filters(parks, locations=None, good_for_kids=None):
    """
    Function to apply location and good for kids filters to the park dictionary.
    """
    # filter by location
    if (parks.items() is None):
        print("No parks found")
        return {}
    if locations is not None and len(locations) > 0:
        parks = {k: v for (k, v) in parks.items() if v['state'] in locations}
    # filter by good for kids
    if good_for_kids == "yes":
        parks = {k: v for (k, v) in parks.items() if v['good_for_kids'] == "True"}
    return parks

# Sample search using json with pandas
def json_search(query, locations=None, good_for_kids=None):
    query_tokens = {}
    for token in tokenize(query):
        if query_tokens.get(token) is None:
            query_tokens[token] = 1
        else:
            query_tokens[token] += 1
    park_dict_filtered = apply_filters(park_dict, locations, good_for_kids)
    inverted_dict = build_inverted_index(park_dict_filtered)
    n_docs = num_docs(park_dict_filtered)
    idf_dict = get_idf_values(park_dict_filtered, n_docs)
    similarity_scores = calculate_similarities(query_tokens, inverted_dict, idf_dict)
    average_park_ratings = calculate_average_ratings(park_dict_filtered)

    # create a dataframe to store the parks and location and rating information
    park_df = pd.DataFrame(columns=['name', 'location', 'score', 'rating'])
    for park, score in similarity_scores.items():
        new_row = pd.DataFrame({'name': park_dict_filtered[park]['name'],
                                'location': park_dict_filtered[park]['state'],
                                'score': score,
                                'rating': average_park_ratings[park]}, index=[0])
        park_df = pd.concat([park_df, new_row])
    # sort the dataframe by score in descending order
    park_df = park_df.sort_values(by='score', ascending=False)
    # return the top 10 parks
    return park_df.head(10).to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/parks")
def episodes_search():
    text = request.args.get("title")
    # apply location filter
    locations = request.args.get("locations")
    if locations:
        locations = locations.split(",")
        locations = [location.strip() for location in locations]
    # apply good for kids filter
    good_for_kids = request.args.get("good_for_kids")
    return json_search(text, locations, good_for_kids)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)