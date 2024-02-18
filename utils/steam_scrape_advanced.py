# This code is a modified version from Andrew Mullers medium article:
# https://andrew-muller.medium.com/scraping-steam-user-reviews-9a43f9e38c92
# I modified his original version so that this scrapes for game name and app id, formats the reviews per game, process as csv. 
# I needed to collect the game names along with the reviews in a tuple format and collect as CSV rather than feather.
# Steam API does not have reviews available for data collection yet, so I web scraped and pre-processed.
# 2/17  updated script with the changes to include the review summary extraction as well.


import requests
from bs4 import BeautifulSoup
import pandas as pd
from html import unescape

def get_reviews(appid, params={'json': 1}):
    url = f'https://store.steampowered.com/appreviews/{appid}'
    try:
        response = requests.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching reviews for appid {appid}: {e}")
        return None

def get_n_appids(n=100, filter_by='topsellers'):
    appids = []
    game_names = []
    review_summaries = []
    url = f'https://store.steampowered.com/search/?category1=998&filter={filter_by}&page='
    
    page = 0
    while len(appids) < n:
        page += 1
        response = requests.get(url=url + str(page), headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')

        for row in soup.find_all(class_='search_result_row'):
            appid = row['data-ds-appid']
            game_name = row.find('span', class_='title').text
            review_summary_element = row.find('span', class_='search_review_summary')
            
            if review_summary_element and 'data-tooltip-html' in review_summary_element.attrs:
                review_summary_html = unescape(review_summary_element['data-tooltip-html'])
                review_summary_text = BeautifulSoup(review_summary_html, 'html.parser').text
            else:
                review_summary_text = "No user reviews"

            print(f"Review Summary for {game_name}: {review_summary_text}")  # Debugging print

            appids.append(appid)
            game_names.append(game_name)
            review_summaries.append(review_summary_text)

            if len(appids) >= n:
                break

    return list(zip(appids, game_names, review_summaries))[:n]

def get_n_reviews(appid_game_summary_tuple, n=50):
    appid, game_name, review_summary = appid_game_summary_tuple
    reviews = []
    cursor = '*'
    params = {
        'json': 1,
        'filter': 'all',
        'language': 'english',
        'day_range': 9223372036854775807,
        'review_type': 'all',
        'purchase_type': 'all'
    }

    while n > 0:
        params['cursor'] = cursor
        params['num_per_page'] = min(100, n)
        response = get_reviews(appid, params)
        if not response:
            break

        cursor = response.get('cursor', '*')
        batch_reviews = response.get('reviews', [])
        for review in batch_reviews:
            review['game_id'] = appid
            review['game_name'] = game_name
            review['review_summary'] = review_summary
        reviews += batch_reviews
        n -= len(batch_reviews)
        if len(batch_reviews) < params['num_per_page']:
            break

    return reviews

reviews = []
appids_and_names_and_reviews = get_n_appids(50)
for appid_game_summary_tuple in appids_and_names_and_reviews:
    reviews += get_n_reviews(appid_game_summary_tuple, 200)

extracted_reviews = []
for review in reviews:
    review_data = {
        'game_id': review['game_id'],
        'game_name': review['game_name'],
        'review_text': review.get('review', ''),
        'review_summary': review.get('review_summary', '')
    }
    extracted_reviews.append(review_data)

df = pd.DataFrame(extracted_reviews)
df.to_csv('data/steam_reviews.csv', index=False)
print("Finished scraping and saved to steam_reviews.csv")
