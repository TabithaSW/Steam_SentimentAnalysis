import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer

def cluster_reviews_for_game(df, game_id, game_name, vectorizer, num_clusters):
    """
    Clusters reviews for a specific game.
    """
    # Filter reviews for the specific game
    game_reviews = df[df['game_id'] == game_id].copy()
    
    # Check if there are enough reviews to cluster
    if len(game_reviews) < num_clusters:
        print(f"Not enough reviews for clustering Game ID: {game_id} - {game_name}")
        return None
    
    # Convert 'processed_review' to TF-IDF vectors
    X = vectorizer.fit_transform(game_reviews['processed_review'])
    
    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    
    # Assign cluster labels to the game's reviews
    game_reviews['cluster'] = kmeans.labels_
    
    return game_reviews

def analyze_and_save_results(clustered_df, output_filepath):
    """
    Performs sentiment analysis on clustered reviews, calculates the average sentiment per cluster,
    assigns sentiment labels based on the cluster average, and saves results to a CSV file.
    """
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to each review
    clustered_df['sentiment_score'] = clustered_df['processed_review'].apply(
        lambda x: sia.polarity_scores(x)['compound'])

    # Calculate the average sentiment score for each cluster
    cluster_sentiment = clustered_df.groupby('cluster')['sentiment_score'].mean().to_dict()

    # Assign a sentiment label based on the average score of each cluster
    clustered_df['cluster_sentiment'] = clustered_df['cluster'].map(cluster_sentiment)
    clustered_df['cluster_label'] = clustered_df['cluster_sentiment'].apply(
        lambda score: 'very_positive' if score >= 0.5 else
                      'positive' if score >= 0.2 else
                      'neutral' if score >= -0.2 else
                      'negative' if score >= -0.5 else
                      'very_negative')

    # Define sentiment descriptions
    sentiment_labels = {
        'very_negative': 'Highly Negative Sentiments',
        'negative': 'Moderately Negative Sentiments',
        'neutral': 'Mixed Sentiments',
        'positive': 'Moderately Positive Sentiments',
        'very_positive': 'Highly Positive Sentiments'
    }

    # Assign sentiment descriptions based on cluster labels
    clustered_df['cluster_description'] = clustered_df['cluster_label'].map(sentiment_labels)

    # Save to CSV
    clustered_df.to_csv(output_filepath, index=False)

def main():
    # Load the entire processed dataset
    df = pd.read_csv('data/processed_reviews.csv')  # Adjusted to load the full processed dataset
    df['processed_review'] = df['processed_review'].fillna('')

    # Initialize a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=30,ngram_range=(1, 3))
    num_clusters = 4 # Example: 5 clusters per game
    
    all_clustered_reviews = []
    unique_games = df[['game_id', 'game_name']].drop_duplicates().values
    
    for game_id, game_name in unique_games:
        clustered_reviews = cluster_reviews_for_game(df, game_id, game_name, vectorizer, num_clusters)
        if clustered_reviews is not None:
            all_clustered_reviews.append(clustered_reviews)
    
    # Combine all clustered reviews into a single DataFrame
    final_clustered_df = pd.concat(all_clustered_reviews, ignore_index=True)
    
    # Analyze sentiment, assign labels to clusters, and save results
    analyze_and_save_results(final_clustered_df, 'data/KMEANS_cluster_TESTres.csv')
    
    print("Clustering, sentiment analysis, and label assignment completed. Results saved to 'data/KMEANS_cluster_res.csv'")

if __name__ == "__main__":
    main()
