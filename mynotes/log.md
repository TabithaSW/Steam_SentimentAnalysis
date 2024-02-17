
1/01 - 2/15
- Made the steam scrapoe script, make the tokenziing script to lemmatize and pull out stop words and pick which words will help determine sentiment. Made an overall processed cvs and text/train. FAed it to the model and made visuals, realized clusters are having mixed identies when they should be easily identified from each other. Also tested html page from flask run.py 


2/16:
- Today I added a test kmeans model so I could correctly label the overall cluster sentiment based on the reviews for each game. This is to ensure the accuracy in my model so the visuals and sentiment are correct, but the labels are mixed for each cluster when each cluster should have an overall them (positive, negative, etc) it could be an issue with my labelling method or my pre-processing or the model features not being tuned enough. Fix this before moving onto the next two models.
