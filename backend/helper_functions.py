"""
Helper file to perform data processing and auxiliary functions such as tokenization
and cosine similarity calculations.
"""

import json
import os
from nltk import NLTKWordTokenizer, PorterStemmer

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
            'reviews': entry['reviews'],
            'latitude': entry['latitude'],
            'longitude': entry['longitude'],
            'image_url': entry.get('image_url'),
            'website_url': entry.get('website_url')
        }
        # if good for kids is in attributes, add it to the dictionary
        if entry['attributes'] and 'GoodForKids' in entry['attributes']:
            park_dict[entry['business_id']]['good_for_kids'] = entry['attributes']['GoodForKids']
        else:
            park_dict[entry['business_id']]['good_for_kids'] = "False"

def apply_filters(parks, park_token_dict, locations=None, good_for_kids=None):
    """
    Function to apply location and good for kids filters to the park dictionary.
    """
    # filter by location
    if (parks.items() is None):
        print("No parks found")
        return {}
    if locations is not None and len(locations) > 0:
        parks = {k: v for (k, v) in parks.items() if v['state'] in locations}
        park_topark_token_dict = {k: v for (k, v) in park_token_dict.items() if parks[k]['state'] in locations}
    # filter by good for kids
    if good_for_kids == "yes":
        parks = {k: v for (k, v) in parks.items() if v['good_for_kids'] == "True"}
        park_token_dict = {k: v for (k, v) in park_token_dict.items() if parks[k]['good_for_kids'] == "True"}
    return parks, park_token_dict

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

all_tokens = unique_tokens(park_dict)
inverted_dict = build_inverted_index(park_dict)
n_docs = num_docs(park_dict)
idf_dict = get_idf_values(park_dict, n_docs)
park_token_dict = aggregate_reviews(park_dict)
average_park_ratings = calculate_average_ratings(park_dict)