**Overview**

The Steam Sentiment Analysis project is a comprehensive application designed to analyze user reviews on Steam, the popular digital distribution platform for video games. Utilizing a custom-built machine learning model, this project aims to accurately gauge public sentiment towards various video games based on user-generated reviews.

**Data Collection & Preprocessing**

I have developed scripts to efficiently scrape Steam reviews for various games. The collected data undergoes a thorough preprocessing stage where I tokenize and lemmatize the review text. This process is crucial for cleaning and preparing the data, making it suitable for the subsequent analysis. By breaking down the text into its base forms, I can ensure that the analysis is based on the core content of the reviews, removing any noise that could skew the results.

**Machine Learning Models**

In my project, I employ unsupervised learning models to cluster the reviews, aiming to uncover common themes and sentiments expressed by gamers. I've chosen a mix of models to address different aspects of the analysis: K-Means clustering serves as the foundation for general grouping of reviews, Hierarchical clustering is used to delve into the nuanced relationships between different sentiments, and LDA (Latent Dirichlet Allocation) is utilized for sophisticated topic modeling. This multifaceted approach allows me to capture a wide array of insights from the review data.

**Analysis & Visualization**

Following the clustering, I plan to apply sentiment analysis using the VADER tool to determine the overall sentiment present within each cluster. This step will help quantify the positive, negative, or neutral sentiments in the reviews. To visually represent the findings, I will create word clouds that highlight the most frequent terms within each cluster, offering an immediate sense of the predominant themes. Additionally, I will use bar charts to present a comparative view of sentiments across different clusters, enabling a clear visualization of the data's emotional landscape.

**Flask Web Application**

To make the insights derived from my analysis accessible and understandable, I am developing a Flask web application. This web app will serve HTML pages that display the results of the sentiment analysis and clustering in an interactive and user-friendly manner. By presenting the data through a web interface, I aim to provide both gamers and game developers with valuable insights into the gaming community's feedback, potentially guiding future game development and enhancing the gaming experience.


**Models: Unsupervised**

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
