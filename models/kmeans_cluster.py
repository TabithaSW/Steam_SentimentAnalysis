# Importing necessary libraries
import pandas as pd  # For data manipulation
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to TF-IDF vectors
from sklearn.cluster import KMeans  # K-Means clustering algorithm

def main():
    # Load the training data
    df = pd.read_csv('../data/train_reviews.csv')

    # Initialize a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)  # Limiting the number of features to 1000 for simplicity

    # Convert the 'processed_review' column to TF-IDF vectors
    X = vectorizer.fit_transform(df['processed_review'])

    # Define the number of clusters for K-Means
    num_clusters = 5  # Example: 5 clusters

    # Initialize and fit the K-Means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # The clusters are now in kmeans.labels_
    df['cluster'] = kmeans.labels_

    # Further analysis of clusters can be done here

if __name__ == "__main__":
    main()
