
**1/01 - 2/15**
- Made the steam scrape script & the tokenzing script (utils) to pre-process, lemmatize and pull out stop words and pick which words will help determine sentiment. Made an overall processed cvs and text/train. Fed it to the kmeans model and made visuals, realized clusters are having mixed identies when they should be easily identified from each other. Making analysis web based using flask framework, tested html homepage from flask run.py. 


**2/16:**
- Added distinct stopwords along with the ones provided from the nltk library. These will be provided in the final report so as to not skew the perception of the analysis, no bias was used in the selection of the words.
- Originally kmeans had no labelling, I was analyzing the clusters myself with visuals. Clustering all reviews for each game into a few clusters (e.g., positive, negative, neutral) but this was proving mixed cluster results between the sentiment of them.
- Made a test kmeans model to see if I could fine tune to get the clusters correctly (labelled and sitrbution wise) distinct between games.
- Today I added a test kmeans model so I could correctly label the overall cluster sentiment based on the reviews for each game. This is to ensure the accuracy in my model so the visuals and sentiment are correct, but the labels are mixed for each cluster when each cluster should have an overall them (positive, negative, etc) it could be an issue with my labelling method or my pre-processing or the model features not being tuned enough. Fix this before moving onto the next two models.
