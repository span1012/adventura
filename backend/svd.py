"""
Script to perform singular-value decomposition on dataset of amusement park
reviews.
"""

import pandas as pd
import numpy as np
import math
from helper_functions import park_dict, tokenize, all_tokens, idf_dict
from sklearn.decomposition import TruncatedSVD

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

# def compute_review_norms(park_reviews_dict, idf_dict):
#     """
#     Function to calculate and return the norm of each distinct park represented
#     in yelp.json. The norms are computed by aggregating the TF-IDF weights of
#     each park across all of its associated reviews and then taking the square
#     root of that sum.
#     """
#     norm_dict = {}
#     for park, tf_dict in park_reviews_dict.items():
#         sum = 0
#         for token, count in tf_dict.items():
#             sum += (count * idf_dict[token]) ** 2
#         norm_dict[park] = math.sqrt(sum)
#     return norm_dict

term_park_mat = get_term_park_matrix(park_dict, all_tokens)
svd = TruncatedSVD(n_components=200, n_iter=20)
truncated_mat = svd.fit_transform(term_park_mat)
park_norms = np.linalg.norm(truncated_mat, axis=1)