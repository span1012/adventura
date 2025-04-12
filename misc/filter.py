# import json

# def extract_business_info(json_data):
#     filtered_data = []
#     for business in json_data:
#         filtered_data.append({
#             "name": business.get("name"),
#             "city": business.get("city"),
#             "state": business.get("state")
#         })
#     return filtered_data

# # Load JSON from a file
# with open("backend/yelp.json", "r", encoding="utf-8") as file:
#     data = json.load(file)

# # Extract relevant information
# filtered_businesses = extract_business_info(data)

# # Save to a new JSON file
# with open("parks.json", "w", encoding="utf-8") as file:
#     json.dump(filtered_businesses, file, indent=2)

# # Print the result
# print(json.dumps(filtered_businesses, indent=2))