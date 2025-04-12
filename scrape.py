import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np

# webpage with list of amusement parks across U.S.
wiki_url = 'https://en.wikipedia.org/wiki/List_of_amusement_parks_in_the_Americas#.C2.A0United_States'

wiki_result = requests.get(wiki_url)
if wiki_result.status_code != 200:
  print("something went wrong:", wiki_result.status_code, wiki_result.reason)

with open("parks.html", "w") as writer:
  writer.write(wiki_result.text)

with open("parks.html", "r") as reader:
  html_source = reader.read()

page = BeautifulSoup(html_source, "html.parser")

# find all park names mentioned on page
lis = page.find_all("li")
print("there are", len(lis), "list items on the page")

parks = []
for item in lis:
  park_name = re.findall(r'.+\s–', item.text)
  if len(park_name) != 0:
    parks.append(park_name[0][:-2])

parks = pd.Series(parks)
# first listed park in U.S.
i1 = parks[parks=="4D Farm"].index[0]
# last listed park in U.S.
i2 = parks[parks=="Villa Campestre"].index[0]
print(f'index of first listed park in U.S.: {i1}')
print(f'index of last listed park in U.S.: {i2}')
# filter list to only include parks in U.S.
parks = parks[i1:i2+1]
parks = parks.reset_index(drop=True)
print(parks.iloc[0])
print(parks.iloc[-1])
print(f'there are {len(parks)} listed parks in the U.S.')

parks.to_csv('parks.csv', header=False, index=False)