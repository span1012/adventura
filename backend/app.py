from ctypes.wintypes import tagSIZE
from flask import Flask, render_template, request
from flask_cors import CORS
from regex import S
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np

import helper_functions
from helper_functions import average_park_ratings, park_token_dict, idf_dict
from svd import park_dict, truncated_mat, park_norms

# for region location
region_to_states = {
    "Southeast": ["FL", "TN", "MO", "LA"],
    "Northeast": ["PA", "NJ", "DE"],
    "Midwest": ["IN", "IL"],
    "Southwest": ["AZ"],
    "West": ["CA", "NV", "ID"]
}

app = Flask(__name__)
CORS(app)

def calculate_similarities(query_tokens, query_norm, inverted_dict, idf_dict, \
                            park_norms) -> dict[str, int]:
    """
    Function to create and return a dictionary that maps business ids to
    the cosine similarity scores between their associated tokenized reviews and
    the input tokenized query.
    """
    scores = {}
    for token, frequency in query_tokens.items():
        if inverted_dict.get(token) is not None and idf_dict.get(token) is not None:
            for park, count in inverted_dict[token]:
                # initialize or update score accumulator
                if scores.get(park) is None:
                    scores[park] = frequency * idf_dict[token] \
                                    * count * idf_dict[token]
                else:
                    scores[park] += frequency * idf_dict[token] \
                                    * count * idf_dict[token]
    for park, score in scores.items():
        scores[park] = score / (query_norm * park_norms[park])
    return scores

def find_similar_parks(query_tokens, park_token_dict, idf_dict) -> dict[str, int]:
     """
     Function to create and return a dictionary that maps amusement park names to
     Function to create and return a dictionary that maps business ids to
     the cosine similarity scores between their associated tokenized reviews and
     the input tokenized query.
     """
     scores = {}
     n_query_tokens = len(query_tokens)
     for park, park_tokens in park_token_dict.items():
         dot_product = 0
         common_tokens = 0     # variable to store number of tokens in common
                               # between the query and the reviews for this park
         for token in query_tokens:
             if park_tokens.get(token) is not None:
                 dot_product += query_tokens.get(token) * idf_dict[token] \
                                * park_tokens.get(token) * idf_dict[token]
                 common_tokens += 1
         total_tokens = n_query_tokens + len(park_tokens) - common_tokens
         scores[park] = (dot_product / total_tokens)
     return scores

# Sample search using json with pandas
def json_search(query, locations=None, latitude=None, longitude=None, distance=None, good_for_kids=None):
    # filter dataset according to user preferences
    park_dict_filtered, park_tokens_filtered = helper_functions.apply_filters(park_dict,
                                                                     park_token_dict,
                                                                     locations,
                                                                     latitude,
                                                                     longitude,
                                                                     distance,
                                                                     good_for_kids)
    # obtain term frequency dictionary
    all_tokens = helper_functions.unique_tokens(park_dict_filtered)
    
    # tokenize query
    query_tokens = {token : 0 for token in all_tokens}

    for token in helper_functions.tokenize(query):
        if query_tokens.get(token) is not None:
            query_tokens[token] += 1

    # obtain vector of query term frequency counts
    updated_query = list(query_tokens.values())
    
    # use cosine similarity to find the park most similar to the user query
    similar_parks = find_similar_parks(query_tokens, park_tokens_filtered, idf_dict)
    park_reverse_index = {park : index for index, park in enumerate(park_dict_filtered)}
    most_similar_park = max(similar_parks, key=similar_parks.get)
    top_park_index = park_reverse_index[most_similar_park]

    # use SVD matrix to find parks similar to top park from cosine similarity
    park_ids = [id for id in park_dict_filtered.keys()]
    inner_products = truncated_mat.dot(truncated_mat[top_park_index,:])
    cosine_sims = inner_products / (park_norms * np.inner(updated_query, updated_query))
    park_scores = sorted(zip(park_ids, cosine_sims))

    # create a dataframe to store the parks and their associated locations,
    # average user ratings, and similarity scores with the user query
    park_df = pd.DataFrame(columns=['name', 'location', 'score', 'rating', 'reviews'])
    for park, score in park_scores:
        top_reviews = [review['text'] for review in park_dict_filtered[park]['reviews'][:3]]

        image_url = park_dict_filtered[park].get('image_url')
        if not image_url or image_url == "None":
            image_url = "static/images/default-park.jpg"

        new_row = pd.DataFrame({
            'name': park_dict_filtered[park]['name'],
            'location': park_dict_filtered[park]['state'],
            'score': score,
            'rating': average_park_ratings[park],
            'reviews': [top_reviews],
            'image_url': image_url,
            'website_url': park_dict_filtered[park].get('website_url'),
            'tag1': park_dict_filtered[park]['tags'][0],
            'tag2': park_dict_filtered[park]['tags'][1],
            'tag3': park_dict_filtered[park]['tags'][2]
        }, index=[0])
        park_df = pd.concat([park_df, new_row])
    # sort the dataframe by score in descending order
    park_df = park_df.sort_values(by='score', ascending=False)
    # return the top 10 parks
    return park_df.head(10).to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/parks")
def park_search():
    text = request.args.get("title")
    if not text:
        text = ""
    # apply location option selected by user
    regions = request.args.get("regions")
    if regions:
        regions = regions.split(",")
        states = []
        for region in regions:
            states.extend(region_to_states.get(region, []))
    else:
        states = None

    # apply geolocation filter (distance)
    latitude = request.args.get("latitude")
    longitude = request.args.get("longitude")
    distance = request.args.get("travel-distance")

    # apply good for kids filter
    good_for_kids = request.args.get("good_for_kids")

    return json_search(text, states, latitude, longitude, distance, good_for_kids)

# if 'DB_NAME' not in os.environ:
#     app.run(debug=True,host="0.0.0.0",port=5000)