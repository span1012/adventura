"""
Script to perform singular-value decomposition on dataset of amusement park
reviews.
"""

import numpy as np
from helper_functions import park_dict, tokenize, all_tokens
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

def get_tfidf_park_matrix(parks, tokens, idf_dict):
    park_reverse_index = {park: i for i, park in enumerate(parks)}
    token_reverse_index = {token: i for i, token in enumerate(tokens)}
    mat = np.zeros((len(parks), len(tokens)))

    for park, attributes in parks.items():
        row = park_reverse_index[park]
        token_counts = {}
        for review in attributes['reviews']:
            for token in tokenize(review['text']):
                if token not in idf_dict:
                    continue
                token_counts[token] = token_counts.get(token, 0) + 1

        for token, count in token_counts.items():
            col = token_reverse_index[token]
            tfidf = count * idf_dict[token]
            mat[row][col] = tfidf

    return mat

term_park_mat = get_term_park_matrix(park_dict, all_tokens)

# token_freq = np.sum(term_park_mat > 0, axis=0) 
# doc_freq_threshold = len(park_dict) * 0.6
# keep_indices = np.where(token_freq < doc_freq_threshold)[0]  

# term_park_mat = term_park_mat[:, keep_indices]  
# all_tokens = [all_tokens[i] for i in keep_indices]  

svd = TruncatedSVD(n_components=15, n_iter=10)
truncated_mat = svd.fit_transform(term_park_mat)
park_norms = np.linalg.norm(truncated_mat, axis=1)

park_ids = []
for park_id, _ in park_dict.items():
    park_ids.append(park_id)

for r in range(len(truncated_mat)):
    sorted_dimensions = np.argsort(truncated_mat[r])[::-1]
    tags = set()
    d = 0
    while len(tags) < 3:
        dimension = sorted_dimensions[d]
        if dimension in [2, 4]:
            tags.add("Kid-Friendly")
        if dimension == 1:
            tags.add("High Thrill")
        if dimension in [2, 7, 8, 12]:
            tags.add("Water Rides")
        if dimension in [2, 3, 13, 14]:
            tags.add("Adventure")
        if dimension in [4, 9, 11, 13]:
            tags.add("Fantasy")
        if dimension in [10, 12]:
            tags.add("Holiday Light Shows")
        if dimension in [0, 2, 5, 7]:
            tags.add("Fun For Everyone")
        d += 1
    park_dict[park_ids[r]]['tags'] = list(tags)

## code to print parks most associated with each dimension
# truncated_df = pd.DataFrame(truncated_mat)
# truncated_df["Park Name"] = park_names
# print(truncated_df)
# for row in truncated_df.rows():
#     sorted_row = row.sort_values(ascending=False)

# for i in range(15):
#     print(truncated_df[[i, "Park Name"]].sort_values(by = i, ascending=False)[:10])

##  code to print top terms associated with each dimension
# top_n = 10
# terms = all_tokens  

# for i, component in enumerate(svd.components_):
#     top_indices = np.argsort(component)[::-1][:top_n]
#     top_terms = [terms[index] for index in top_indices]
#     print(f"Dimension {i}: {', '.join(top_terms)}")
