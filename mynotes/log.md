***Viewers:***

I kindly request that individuals who clone this repository refrain from reproducing my work verbatim and presenting it as their own project. Please take the time to review the license; this repository is public and intended for educational purposes and career development. If you are interested in contributing to the project, please reach out to me via message.

**1/01 - 2/15**
- Made a steam scrape script & the tokenzing script (utils) to pre-process, lemmatize and pull out stop words and pick which words will help determine sentiment. Made an overall processed csv and test/train csv to feed to the models. Fed it to the kmeans model first, and made basic visuals (wordclouds, etc), to view clusters and find primary identies. Making analysis web based using flask framework, tested html homepage from flask run.py. 


**2/16:**
- Added distinct stopwords along with the ones provided from the nltk library, tokenize script. These will be provided in the final report so as to not skew the perception of the analysis, no bias was used in the selection of the words. Added some sentiment words to prioritize, these will also be added to the report.
- Originally kmeans had no labelling, I was analyzing the clusters myself with visuals. Clustering reviews for each game into a few clusters (e.g., no labels yet - positive, negative, neutral) but this was proving mixed cluster results between the sentiment of them when I analyzed the visuals. Started feature engineering, fine tuning the model, and remaking pre-processing to ensure higher accuracy. 
- Made changes to the kmeans model - Changed the thresholds for more granularity when labelling the clusters.


**2/17:**
- Added from sklearn.metrics import silhouette_score, davies_bouldin_score to the kmeans so I could get a better idea of the cluster accuracy.
- Advancing the tokenizing so the pre-processed data is better quality.  Emphasis Detection,  Handling for Slang and Abbreviations, Negation Handling, also want to add Emoji and Emoticon Analysis since this is important to these type of reviews, gamers use slang/emojis. Moved old tokenize file to new folder, old_models_and_tests. Changed the kmeans model to account for the emphasis, slang, and negation from the pre-process.
- Adding new words and their associated sentiment intensity scores to the VADER lexicon dictionary. Game specific sentiment scores to the kmeans VADER because sometimes jargon in the reviews are viewed different than traditional reviews, since gamers don't use the same terminology as movie reviews or podcast reviews, etc. Stuff like gg, rpg. Also because words like complex, challenge, difficulty, those may be perceived as negative in regular reviews but for games that would not be perceived this way. 
- Added the overall review score from steam to my steam_scrape so I can compare my kmeans cluster results to actual steam game labels instead of training my unsupervised models on labelled data. For example, "Review Summary for Poppy Playtime: Very Positive 81% of the 65,131 user reviews for this game are positive."

**2/17 More**
- Refined the algorithm, clusters per game more homogenous in terms of sentiment in each cluster. Made a utlity for checking what terms whave the highest TF-IDF scores (they indicate their importance in the documents) so I can make sure it's using the right vader-lexicon terms to identify cluster labelling.
- Made a Silhouette Method script so I can get a better idea of how many clusters to use. (Feature engineering/fine tuning)
- Made another utility test_cluster_distribution to make sure they are properly being generated.
- Adjusted stopwords to exclude all mentions of the game name.
- Changed the visuals to be PER GAME for the wordclouds by cluster visual, sentiment distribution across clusters visual, and sentiment scores between clusters (violin plot), and distribution of sentiment scores across all reviews - because the clusters are generated on all the (200) reviews PER GAME. So making a visual of all the clusters wouldn't correctly portray the distribution of sentiment (since each game sentiment distribution is different)
