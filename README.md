**Overview**

The Steam Sentiment Analysis project is a comprehensive application designed to analyze user reviews on Steam, the popular digital distribution platform for video games. Utilizing a custom-built machine learning model, this project aims to accurately gauge public sentiment towards various video games based on user-generated reviews.

**Objectives**

- Sentiment Analysis: Implement a machine learning model to classify the sentiment of Steam reviews into categories such as positive, negative, or neutral.
  
- Data Collection: Fetch and process review data from Steam using the Steam API.
  
- User Interface: Develop a user-friendly web interface where users can view sentiment analysis results.
  
- Scalability: Ensure the application can handle a large number of reviews and can be scaled up to include more games and different types of analysis.


**Models:**

1. K-Means Clustering
- Description: A method that groups reviews into a specified number of clusters based on the similarity of their content.
- Application to Steam Reviews: K-Means can categorize reviews into distinct groups based on their text. Each cluster might represent reviews focusing on similar aspects of games, like graphics, gameplay, or customer service. However, it won't explicitly label these groups as positive or negative.

2. Hierarchical Clustering
- Description: Builds a tree-like structure of clusters, showing how each review is grouped at various levels of similarity.
- Application to Steam Reviews: This method allows you to see not just which reviews are similar, but also how they relate to each other in a multi-level hierarchy. It can reveal nuanced relationships between different reviews, such as grouping together all reviews that discuss a specific game feature, then further subdividing them based on sentiment or specific aspects of that feature.

3. Latent Dirichlet Allocation (LDA) for Topic Modeling
- Description: An advanced technique that discovers latent topics within the text data. Each review can contribute to multiple topics to varying degrees.
- Application to Steam Reviews: LDA can identify underlying themes or topics across your reviews. For instance, it might reveal common subjects like game difficulty, story depth, or technical issues. This method helps in understanding the predominant topics of discussion in the reviews, although it doesn’t classify sentiment directly.


**Toolset:**
- Flask
- Python 
- NLTK
- Scikit-Learn
- Gensim
- Pandas
- Transformers
- Feather/CSV Files
- Git and GitHub
