import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from regex import S
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import re
import math
from nltk import NLTKWordTokenizer, PorterStemmer
from sklearn.decomposition import TruncatedSVD

# for region location
region_to_states = {
    "Southeast": ["FL", "TN", "MO", "LA"],
    "Northeast": ["PA", "NJ", "DE"],
    "Midwest": ["IN", "IL"],
    "Southwest": ["AZ"],
    "West": ["CA", "NV", "ID"]
}

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'parks_details.json')

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
    stemmer = PorterStemmer()
    tokenizer = NLTKWordTokenizer()
    tokens = tokenizer.tokenize(text.lower())
    return [stemmer.stem(token) for token in tokens]

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
    one park review to its associated inverse document frequency value.
    """
    idf_dict = {}
    for attributes in parks.values():
        for review in attributes['reviews']:
            tokens = set(tokenize(review['text']))
            for token in tokens:
                # token = token.lower()
                if idf_dict.get(token) is None:
                    idf_dict[token] = 1
                else:
                    idf_dict[token] += 1
    for token, count in idf_dict.items():
        idf_dict[token] = num_docs / count
    return idf_dict

def unique_tokens(parks):
    """
    Function to return a set of all of the unique terms that appear at least
    once across all park reviews in the dataset.
    """
    all_tokens = set()
    for attributes in parks.values():
        for review in attributes['reviews']:
            all_tokens = all_tokens.union(tokenize(review['text']))
    all_tokens = sorted(all_tokens)
    return all_tokens

def get_term_park_matrix(parks, tokens):
    """
    Function to obtain matrix that stores, for each distinct park in the dataset,
    frequency counts of each unique term from across all park reviews
    """
    park_reverse_index = {park : index for index, park in enumerate(parks)}
    term_reverse_index = {token : index for index, token in enumerate(tokens)}
    mat = np.zeros((len(park_reverse_index), len(term_reverse_index)))
    for park, attributes in parks.items():
        for review in attributes['reviews']:
            tokens = tokenize(review['text'])
            for token in tokens:
                mat[park_reverse_index[park]][term_reverse_index[token]] += 1
    return mat

def build_inverted_index(parks) -> dict[str, list[(str, int)]]:
    """
    Function to create an inverted index dictionary for all unique terms across
    the entire set of amusement park reviews. The dictionary maps each unique
    term to a list of tuples, where the first value in the tuple is a park
    business ID, and the second value represents the number of times that the
    token appears in a review for that park.
    """
    inverted_dict = {}
    for park, attributes in parks.items():
        for review in attributes['reviews']:
            for token in tokenize(review['text']):
                # token = token.lower()
                if inverted_dict.get(token) is None:
                    inverted_dict[token] = [(park, 1)]
                else:
                    last_pair = inverted_dict[token][len(inverted_dict[token]) - 1]
                    # check if the token has already been found in a review for
                    # this park
                    if last_pair[0] == park:
                        inverted_dict[token][len(inverted_dict[token]) - 1] = \
                                                (last_pair[0], last_pair[1] + 1)
                    else:
                        inverted_dict[token].append((park, 1))
    return inverted_dict

def aggregate_reviews(parks) -> dict[str, dict[str, int]]:
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

def compute_review_norms(park_reviews_dict, idf_dict):
    """
    Function to calculate and return the norm of each distinct park represented
    in yelp.json. The norms are computed by aggregating the TF-IDF weights of
    each park across all of its associated reviews and then taking the square 
    root of that sum.
    """
    norm_dict = {}
    for park, tf_dict in park_reviews_dict.items():
        sum = 0
        for token, count in tf_dict.items():
            sum += (count * idf_dict[token]) ** 2
        norm_dict[park] = math.sqrt(sum)
    return norm_dict

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
    # query_tokens = {}
    # for token in tokenize(query):
    #     # token = token.lower()
    #     if query_tokens.get(token) is None:
    #         query_tokens[token] = 1
    #     else:
    #         query_tokens[token] += 1
    
    # p04 changes
    park_dict_filtered = apply_filters(park_dict, locations, good_for_kids)
    all_tokens = unique_tokens(park_dict_filtered)
    
    query_tokens = {token : 0 for token in all_tokens}

    for token in tokenize(query):
        if query_tokens.get(token) is not None:
            query_tokens[token] += 1
    
    updated_query = list(query_tokens.values())
    # # p03 changes
    inverted_dict = build_inverted_index(park_dict_filtered)
    n_docs = num_docs(park_dict_filtered)
    idf_dict = get_idf_values(park_dict_filtered, n_docs)
    query_norm = 0
    for token, count in query_tokens.items():
        query_norm += (count * idf_dict[token]) ** 2
    query_norm = math.sqrt(query_norm)
    park_token_dict = aggregate_reviews(park_dict_filtered)
    park_norms = compute_review_norms(park_token_dict, idf_dict)
    similarity_scores = calculate_similarities(query_tokens, query_norm, \
                                                inverted_dict, idf_dict, park_norms)
    similarity_scores = find_similar_parks(query_tokens, park_token_dict, idf_dict)
    park_reverse_index = {park : index for index, park in enumerate(park_dict_filtered)}
    most_similar_park = max(similarity_scores, key=similarity_scores.get) 
    top_park_index = park_reverse_index[most_similar_park]
    
    term_park_mat = get_term_park_matrix(park_dict_filtered, all_tokens)
    svd = TruncatedSVD(n_components=200, n_iter=20)
    truncated_mat = svd.fit_transform(term_park_mat)
    park_ids = [id for id in park_dict_filtered.keys()]
    inner_products = truncated_mat.dot(truncated_mat[top_park_index,:])
    park_norms = np.linalg.norm(truncated_mat, axis=1)
    cosine_sims = inner_products / (park_norms * np.inner(updated_query, updated_query))
    similarity_scores = sorted(zip(park_ids, cosine_sims))

    average_park_ratings = calculate_average_ratings(park_dict_filtered)

    # create a dataframe to store the parks and their associated locations,
    # average ratings, and similarity scores with the user query

    park_df = pd.DataFrame(columns=['name', 'location', 'score', 'rating', 'reviews'])
    for park, score in similarity_scores:
        top_reviews = [review['text'] for review in park_dict_filtered[park]['reviews'][:3]]

        new_row = pd.DataFrame({'name': park_dict_filtered[park]['name'],
                                'location': park_dict_filtered[park]['state'],
                                'score': score,
                                'rating': average_park_ratings[park],
                                'reviews': [top_reviews]}, index=[0])
        park_df = pd.concat([park_df, new_row])
    # sort the dataframe by score in descending order
    park_df = park_df.sort_values(by='score', ascending=False)
    # return the top 10 parks
    return park_df.head(10).to_json(orient='records')

# used for testing
def main():
    park_dict_filtered = apply_filters(park_dict, None, None)
    all_tokens = unique_tokens(park_dict_filtered)
    query = "roller coaster"
    query_tokens = {token : 0 for token in all_tokens}
    for token in tokenize(query):
        if query_tokens.get(token) is not None:
            query_tokens[token] += 1
    updated_query = list(query_tokens.values())
    
    inverted_dict = build_inverted_index(park_dict_filtered)
    n_docs = num_docs(park_dict_filtered)
    idf_dict = get_idf_values(park_dict_filtered, n_docs)
    query_norm = 0
    for token, count in query_tokens.items():
        query_norm += (count * idf_dict[token]) ** 2
    query_norm = math.sqrt(query_norm)
    park_token_dict = aggregate_reviews(park_dict_filtered)
    park_norms = compute_review_norms(park_token_dict, idf_dict)
    similarity_scores = calculate_similarities(query_tokens, query_norm, \
                                                inverted_dict, idf_dict, park_norms)
    similarity_scores = find_similar_parks(query_tokens, park_token_dict, idf_dict)
    park_reverse_index = {park : index for index, park in enumerate(park_dict_filtered)}
    most_similar_park = max(similarity_scores, key=similarity_scores.get) 
    top_park_index = park_reverse_index[most_similar_park]

    term_park_mat = get_term_park_matrix(park_dict_filtered, all_tokens)
    svd = TruncatedSVD(n_components=200, n_iter=20)
    truncated_mat = svd.fit_transform(term_park_mat)
    park_names = [attributes['name'] for attributes in park_dict_filtered.values()]
    inner_products = truncated_mat.dot(truncated_mat[top_park_index,:])
    park_norms = np.linalg.norm(truncated_mat, axis=1)
    cosine_sims = inner_products / (park_norms * np.inner(updated_query, updated_query))
    park_scores = sorted(zip(cosine_sims, park_names))

if __name__ == '__main__':
    main()

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/parks")
def episodes_search():
    text = request.args.get("title")
    # ––––––––––– Version 1 –––––––––––
    # # apply location filter
    # locations = request.args.get("locations")
    # if locations:
    #     locations = locations.split(",")
    #     locations = [location.strip() for location in locations]

    # ––––––––––– Version 2 –––––––––––
    # apply location filter
    # get regions + expand to states
    regions = request.args.get("regions")
    if regions:
        regions = regions.split(",")
        states = []
        for region in regions:
            states.extend(region_to_states.get(region, []))
    else:
        states = None

    # apply good for kids filter
    good_for_kids = request.args.get("good_for_kids")

    return json_search(text, states, good_for_kids)

# if 'DB_NAME' not in os.environ:
#     app.run(debug=True,host="0.0.0.0",port=5000)