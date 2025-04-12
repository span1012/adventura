import json

# Load the JSON file
with open('busche_gardens_reviews.json', 'r') as file:
    data = json.load(file)

# Count the number of reviews
num_reviews = len(data['reviews'])
print(f"Number of reviews: {num_reviews}")