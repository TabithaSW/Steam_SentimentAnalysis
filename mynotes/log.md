
**1/01 - 2/15**
- Made the steam scrape script & the tokenzing script (utils) to pre-process, lemmatize and pull out stop words and pick which words will help determine sentiment. Made an overall processed cvs and test/train. Fed it to the kmeans model and made visuals, labelled clusters to find primary identies. Making analysis web based using flask framework, tested html homepage from flask run.py. 


**2/16:**
- Added distinct stopwords along with the ones provided from the nltk library. These will be provided in the final report so as to not skew the perception of the analysis, no bias was used in the selection of the words.
- Originally kmeans had no labelling, I was analyzing the clusters myself with visuals. Clustering all reviews for each game into a few clusters (e.g., positive, negative, neutral) but this was proving mixed cluster results between the sentiment of them. Started feature engineering, fine tuning model, and remaking pre-processing to ensure higher accuracy.
- Made a secondary test kmeans model to see if I could fine tune to get the clusters correctly (labelled and distribution wise) distinct between games.


**2/17:**
- Advancing the tokenizing so the pre-processed data is better quality.  Emphasis Detection,  Handling for Slang and Abbreviations, Negation Handling, also want to add Emoji and Emoticon Analysis since this is important to these type of reviews, gamers use slang/emojis. Moved old tokenize file to new folder, old_models_and_tests.
- Adding new words and their associated sentiment intensity scores to the VADER lexicon dictionary. Game specific sentiment scores to the kmeans VADER because sometimes jargon in the reviews are viewed different than traditional reviews, since gamers don't use the same terminology as movie reviews or podcast reviews, etc. Stuff like gg, rpg. Also because words like complex, challenge, difficulty, those may be perceived as negative in regular reviews but for games that would not be perceived this way. 
- Changed the kmeans model to account for emphasis, slang, and negation from the pre-process.
- Added the overall review score from steam to my steam_scrape so I can compare my kmeans cluster results to actual steam game labels instead of training my unsupervised models on labelled data. For example, "Review Summary for Poppy Playtime: Very Positive81% of the 65,131 user reviews for this game are positive."

