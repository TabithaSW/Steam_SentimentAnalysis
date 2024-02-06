
# This code is a modified version from Andrew Mullers medium article:
# https://andrew-muller.medium.com/scraping-steam-user-reviews-9a43f9e38c92
# I modified his original version so that this scrapes for game name and app id, formats the reviews per game, process as csv. 
# I needed to collect the game names along with the reviews in a tuple format and collect as CSV rather than feather.
# Steam API does not have reviews available for data collection yet, so I web scraped and pre-processed.

import requests # A Python HTTP library for sending all kinds of HTTP requests. 
# Used here to make GET requests to the Steam website.

import bs4 # Part of the Beautiful Soup library, used for parsing HTML and XML documents. It creates parse trees that is helpful to extract the data easily
from bs4 import BeautifulSoup
import pandas as pd # A data manipulation and analysis library.


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
    Function get_n_appids, pulls appid which is unique for each Steam game.
    Takes two parameters: n (the number of app IDs and game names to retrieve, defaulting to 100) 
    and filter_by (the criterion for filtering games, defaulting to 'topsellers').
    Returns a list of tuples where each tuple contains an app ID and the corresponding game name.
    """
    appids = []  # Initializes an empty list to store the app IDs.
    game_names = []  # Initializes an empty list to store the game names.

    url = f'https://store.steampowered.com/search/?category1=998&filter={filter_by}&page='
    # Sets the base URL for searching games on Steam, including the filter criteria and a placeholder for the page number.
    
    page = 0  # Initializes the page number to 0.

    while page * 25 < n:
        # Begins a loop to paginate through the search results. Steam typically displays 25 results per page, so this continues until enough app IDs are gathered.
        page += 1  # Increments the page number.

        response = requests.get(url=url + str(page), headers={'User-Agent': 'Mozilla/5.0'})
        # Sends a GET request for each new page, again setting a browser-like user agent in the headers.
        soup = BeautifulSoup(response.text, 'html.parser')
        # Parses the HTML response using Beautiful Soup.

        for row in soup.find_all(class_='search_result_row'):
            # Iterates over each game listing found in the parsed HTML.
            appid = row['data-ds-appid']  # Extracts the app ID from each listing.
            game_name = row.find('span', class_='title').text  # Extracts the game name.
            appids.append(appid)
            game_names.append(game_name)

    return list(zip(appids, game_names))[:n]

# collected_ids = get_n_appids(n=100,filter_by='topsellers')
# print("APP IDS", collected_ids) TEST WORKED HERE
def get_n_reviews(appid_game_tuple, n=50):
    """
    Collects up to 'n' reviews for a given Steam game.
    """
    appid, game_name = appid_game_tuple
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
        reviews += batch_reviews
        n -= len(batch_reviews)
        if len(batch_reviews) < params['num_per_page']:
            break

    return reviews

# Collect reviews for each game
reviews = []
appids_and_names = get_n_appids(500)  # Adjust the number to how many games you want to process
for appid_game_tuple in appids_and_names:
    reviews += get_n_reviews(appid_game_tuple, 50)  # Collect 50 reviews for each game

# Extract relevant data from each review, we want to csv format organized for tokenizing:
extracted_reviews = []
for review in reviews:
    review_data = {
        'game_id': review['game_id'],
        'game_name': review['game_name'],
        'review_text': review['review']
    }
    extracted_reviews.append(review_data)

# Create a DataFrame with specific columns and save to CSV (not feather file this time just for ease of access)
df = pd.DataFrame(extracted_reviews)
df.to_csv('steam_reviews.csv', index=False)

# i am going to start with 500  games and 50 reviews for each. we can increase as model evaluation continues.