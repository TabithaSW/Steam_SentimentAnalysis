
# This code utility was derived from Andrew Mullers medium article:
# https://andrew-muller.medium.com/scraping-steam-user-reviews-9a43f9e38c92
# I modified his original version from the open source code he posted for scraping reviews. 
# I needed to collect the game names along with the reviews in a tuple format and collect as CSV rather than feather.
# Steam API does not have reviews available for data collection yet, so I web scraped and pre-processed.

import requests # A Python HTTP library for sending all kinds of HTTP requests. 
# Used here to make GET requests to the Steam website.

import bs4 # Part of the Beautiful Soup library, used for parsing HTML and XML documents. It creates parse trees that is helpful to extract the data easily
from bs4 import BeautifulSoup
import pandas as pd # A data manipulation and analysis library.


def get_reviews(appid_game_tuple, params={'json': 1}):
    """
    Defines a function get_reviews which takes two parameters: 
    appid_game_tuple (a tuple containing the ID and name of a Steam app/game) 
    and params (optional, defaulting to {'json': 1} to specify the format of the response).
    
    """
    appid = appid_game_tuple[0]  # Extracts the app ID from the tuple
    url = 'https://store.steampowered.com/appreviews/' + appid  # Constructs the complete URL for Steam app reviews.

    response = requests.get(url=url, params=params, headers={'User-Agent': 'Mozilla/5.0'}) 
    # Sends a GET request using the requests library.
    # The user agent in the headers mimics a browser request (some websites block requests that don't come from browsers).

    return response.json()  # Returns the JSON response containing the reviews for the specified Steam app.


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
