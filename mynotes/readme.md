# Run Guide

1. **Run the steam_scrape_advanced script** in the utilities folder. Produces the raw, unprocessed steam review data into a csv available in the data folder, called steamreviews.csv

2. **Run the tokenize script**, this script preprocesses the reviews by handling negations (like "not good", "very good", etc), replaces game specific abbreviations (like 'pvp") with full terms, checks for emphasized words, stopwords, then tokenizes the text into words and applies lemmatization (reducing words to their base to normalize). I added in more custom stop words along with the NLTK library specific to games. **Then it splits the data into train and test sets for the models and saves it as process_reviews.csv in the data folder.**

3. **Run the TD_IDF script**, this will list the most relevant terms across all reviews, we want to make sure the processed reviews are pulling actual key words before putting data into the models, no junk. Then **run Silhoutte Script to get the best number of clusters based off the processed data.**

4. 