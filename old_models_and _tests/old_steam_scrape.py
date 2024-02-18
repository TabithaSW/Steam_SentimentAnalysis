
# This code is a modified version from Andrew Mullers medium article:
# https://andrew-muller.medium.com/scraping-steam-user-reviews-9a43f9e38c92
# I modified his original version so that this scrapes for game name and app id, formats the reviews per game, process as csv. 
# I needed to collect the game names along with the reviews in a tuple format and collect as CSV rather than feather.
# Steam API does not have reviews available for data collection yet, so I web scraped and pre-processed.
# 2/17  updated script with the changes to include the review summary extraction.

import requests # A Python HTTP library for sending all kinds of HTTP requests. 
# Used here to make GET requests to the Steam website.

import bs4 # Part of the Beautiful Soup library, used for parsing HTML and XML documents. It creates parse trees that is helpful to extract the data easily
from bs4 import BeautifulSoup
import pandas as pd # A data manipulation and analysis library.\
from html import unescape



def get_reviews(appid, params={'json': 1}):
    """
    Fetches reviews for a given Steam app/game using its appid.
    """
    url = f'https://store.steampowered.com/appreviews/{appid}'
    try:
        response = requests.get(url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()
    # added fail check
    except requests.exceptions.RequestException as e:
        print(f"Error fetching reviews for appid {appid}: {e}")
        return None




def get_n_appids(n=100, filter_by='topsellers'):
    """
    Fetches appids along with the game names and review summaries.
    """
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
            # Extract the review summary information
            review_summary_element = row.find('span', class_='search_review_summary')
            if review_summary_element and 'data-tooltip-html' in review_summary_element.attrs:
                # Unescape HTML entities in the tooltip text
                review_summary_html = unescape(review_summary_element['data-tooltip-html'])
                review_summary_text = BeautifulSoup(review_summary_html, 'html.parser').text
            else:
                review_summary_text = "No user reviews"

            appids.append(appid)
            game_names.append(game_name)
            review_summaries.append(review_summary_text)

            if len(appids) >= n:
                break

    return list(zip(appids, game_names, review_summaries))[:n]
# collected_ids = get_n_appids(n=100,filter_by='topsellers')
# print("APP IDS", collected_ids) TEST WORKED HERE
def get_n_reviews(appid_game_summary_tuple, n=50):
    """
    Collects up to 'n' reviews for a given Steam game.
    """
    appid, game_name, review_summary = appid_game_summary_tuple  # Corrected variable name here
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
            break  # If there's an error or no response, break the loop

        cursor = response.get('cursor', '*')
        batch_reviews = response.get('reviews', [])
        for review in batch_reviews:
            review['game_id'] = appid
            review['game_name'] = game_name
            # Optionally include 'review_summary' from 'appid_game_summary_tuple' if needed
        reviews += batch_reviews
        n -= len(batch_reviews)
        if len(batch_reviews) < params['num_per_page']:
            break

    return reviews

# Collect reviews for each game
reviews = []
appids_and_names_and_reviews = get_n_appids(50)  # Adjust the number to how many games you want to process
for appid_game_summary_tuple in appids_and_names_and_reviews:
    reviews += get_n_reviews(appid_game_summary_tuple, 50)  # Collect 50 reviews for each game


# Extract relevant data from each review, we want to csv format organized for tokenizing:
extracted_reviews = []
for review in reviews:
    review_data = {
        'game_id': review['game_id'],
        'game_name': review['game_name'],
        'review_text': review['review'],
        'review_summary': review.get('review_summary', '')  # Include the review summary if available
    }
    extracted_reviews.append(review_data)

# Create a DataFrame with specific columns and save to CSV (not feather file this time just for ease of access)
df = pd.DataFrame(extracted_reviews)
df.to_csv('steam_reviews.csv', index=False)

# i am going to start with 50  games and 50 reviews for each. we can increase as model evaluation continues.