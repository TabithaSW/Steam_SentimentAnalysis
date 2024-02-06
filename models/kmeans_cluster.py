# Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
def cluster_reviews_for_game(df, game_id, game_name, vectorizer, num_clusters, output_file=None):
    """
    Clusters reviews for a specific game and writes results to a file.
    """
    # Filter reviews for the specific game
    game_reviews = df[df['game_id'] == game_id]
    
    # Write game information to the file
    if output_file is not None:
        output_file.write(f"\n\n{'='*40}\nGame ID: {game_id}, Game Name: {game_name}\n{'='*40}\n")
    
    # Check if there are enough reviews to cluster
    if len(game_reviews) < num_clusters:
        msg = f"Not enough reviews for clustering Game ID: {game_id} - {game_name}"
        print(msg)
        if output_file is not None:
            output_file.write(msg + "\n")
        return game_reviews
    
    # Convert 'processed_review' to TF-IDF vectors
    X = vectorizer.fit_transform(game_reviews['processed_review'])
    
    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    
    # Assign cluster labels to the game's reviews
    game_reviews['cluster'] = kmeans.labels_
    
    # Optionally write cluster information to the file
    if output_file is not None:
        for i in range(num_clusters):
            output_file.write(f"Cluster {i} reviews sample:\n")
            cluster_size = game_reviews[game_reviews['cluster'] == i].shape[0]
            n_samples = min(3, cluster_size)  # Take 3 samples or less depending on the cluster size
            sample_reviews = game_reviews[game_reviews['cluster'] == i]['processed_review'].sample(n=n_samples, random_state=42).values
            for review in sample_reviews:
                output_file.write(f"- {review[:200]}\n")  # Write the first 200 characters of each review
            output_file.write("\n")
    
    return game_reviews


def main():
    # Load the training data
    df = pd.read_csv('data/train_reviews.csv')
    df['processed_review'] = df['processed_review'].fillna('')

    # Initialize a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
     # When you multiply TF by IDF, you get a value that increases with the number of times a word appears in a document but decreases if it's common across all documents. 
    # This helps to highlight words that are unique and important in individual reviews.
    
    num_clusters = 5  # Example: 5 clusters per game
    
    # Open a text file to write the output
    with open('cluster_analysis_results.txt', 'w', encoding='utf-8') as output_file:
        # Get unique games in the dataset
        unique_games = df[['game_id', 'game_name']].drop_duplicates().values
        
        for game_id, game_name in unique_games:
            cluster_reviews_for_game(df, game_id, game_name, vectorizer, num_clusters, output_file)
        
        print("Clustering completed and results saved to cluster_analysis_results.txt")

if __name__ == "__main__":
    main()

