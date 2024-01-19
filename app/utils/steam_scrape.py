
# This code utility was derived from Andrew Mullers medium article:
# https://andrew-muller.medium.com/scraping-steam-user-reviews-9a43f9e38c92
# I DID NOT WRITE THIS CODE PORTION. I took his open source code for scraping reviews. All other scripts are written by me.
# Steam API does not have review connections yet, so I web scraped and pre-processed.

import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd


def get_reviews(appid, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+appid, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        return response.json()

def get_n_appids(n=100, filter_by='topsellers'):
    appids = []
    url = f'https://store.steampowered.com/search/?category1=998&filter={filter_by}&page='
    page = 0

    while page*25 < n:
        page += 1
        response = requests.get(url=url+str(page), headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        for row in soup.find_all(class_='search_result_row'):
            appids.append(row['data-ds-appid'])

    return appids[:n]

# collected_ids = get_n_appids(n=100,filter_by='topsellers')
# print(collected_ids)

def get_n_reviews(appid, n=100):
    reviews = []
    cursor = '*'
    params = {
            'json' : 1,
            'filter' : 'all',
            'language' : 'english',
            'day_range' : 9223372036854775807,
            'review_type' : 'all',
            'purchase_type' : 'all'
            }

    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100, n)
        n -= 100

        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews += response['reviews']

        if len(response['reviews']) < 100: break

    return reviews


# reviews_res = get_n_reviews(collected_ids)


reviews = []
appids = get_n_appids(100)
for appid in appids:
    reviews += get_n_reviews(appid, 100)
df = pd.DataFrame(reviews)[['review', 'voted_up']]
df.to_feather('steam_reviews.feather')
