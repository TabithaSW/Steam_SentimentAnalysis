
**1/01 - 2/15**
- Made the steam scrape script & the tokenzing script (utils) to pre-process, lemmatize and pull out stop words and pick which words will help determine sentiment. Made an overall processed cvs and test/train. Fed it to the kmeans model and made visuals, labelled clusters to find primary identies. Making analysis web based using flask framework, tested html homepage from flask run.py. 


**2/16:**
- Added distinct stopwords along with the ones provided from the nltk library. These will be provided in the final report so as to not skew the perception of the analysis, no bias was used in the selection of the words.
- Originally kmeans results had no labelling, I was analyzing the clusters myself with visuals. Clustering all reviews for each game into a few clusters (e.g., positive, negative, neutral) but this was proving mixed cluster results between the sentiment of them. Fine tuned cluster count and features.
- Made a secondary test kmeans model to see if I could fine tune to get the clusters more precisely (labelled and distribution wise) distinct between games. This is to ensure the accuracy in my model so the visuals and sentiment are correct, each cluster should have an overall them (positive, negative, etc) it could be an issue with my labelling method or my pre-processing or the model features not being tuned enough.
- Working on this before moving onto the next two models.
