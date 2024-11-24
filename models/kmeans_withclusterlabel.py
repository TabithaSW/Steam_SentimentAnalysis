import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from custom_lexicon import custom_lexicon


"""

K-Means is a centroid-based clustering method, which partitions the data into K clusters by minimizing the variance within each cluster. 
It requires specifying the number of clusters in advance and works well for spherical clusters.

This script performs unsupervised clustering and sentiment analysis on text reviews of video games.

Clustering: Reviews are clustered for each game using the K-Means algorithm, with metrics like Silhouette Score and Davies-Bouldin Index calculated to assess the quality of clustering.
Sentiment Analysis: Reviews are analyzed for sentiment, using a custom lexicon tailored to gaming terminology to improve accuracy.
Output: The processed reviews, with clustering and sentiment details, are saved to a CSV file.


"""

def cluster_reviews_for_game(df, game_id, game_name, vectorizer, num_clusters,cluster_range=(2,10)):
    """
    Inputs:
    df: The DataFrame containing the dataset.
    game_id: The unique identifier of a specific game.
    game_name: The name of the game.
    vectorizer: A TF-IDF vectorizer for transforming text data into numerical features.
    num_clusters: The number of clusters to divide the reviews into.
    

    Filters reviews specific to the given game. Vectorizes the reviews using TF-IDF, converting text into numerical features.
    Performs K-Means clustering on the vectorized data. Calculates Silhouette Score and Davies-Bouldin Index to evaluate the clustering quality.
    Adds cluster labels, Silhouette Score, and Davies-Bouldin Index to the reviews DataFrame.
    
    Outputs:
    Returns a DataFrame containing the clustered reviews and evaluation metrics for the specific game.
    """
    game_reviews = df[df['game_id'] == game_id].copy()

    """
    if len(game_reviews) < num_clusters:
        print(f"Not enough reviews for clustering Game ID: {game_id} - {game_name}")
        return None
    """

    X = vectorizer.fit_transform(game_reviews['processed_review'])
    # optimal_k = find_optimal_clusters(X, cluster_range) # Testing this out.
    X_dense = X.toarray()  # Convert sparse matrix to dense
    kmeans = KMeans(n_clusters=num_clusters, random_state=10)
    kmeans.fit(X_dense)

    game_reviews['cluster'] = kmeans.labels_
    
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    davies_bouldin_avg = davies_bouldin_score(X_dense, kmeans.labels_)
    
    # Store the scores in the DataFrame
    game_reviews['silhouette_score'] = silhouette_avg
    game_reviews['davies_bouldin_score'] = davies_bouldin_avg

    return game_reviews

def find_optimal_clusters(vectorized_data, cluster_range=(2,10)):
    """
    Finds the optimal number of clusters using silhouette score.
    """
    silhouette_scores = {}
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=10)
        kmeans.fit(vectorized_data)
        score = silhouette_score(vectorized_data, kmeans.labels_)
        silhouette_scores[k] = score
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print("OPTIMAL CLUSTERS TEST:",optimal_k)
    return optimal_k

def analyze_and_save_results(clustered_df, output_filepath):
    """
    Inputs:
    clustered_df: DataFrame containing reviews and their cluster labels.
    output_filepath: Filepath to save the analyzed and updated DataFrame.
        
    Extends the sentiment lexicon with a custom lexicon (custom_lexicon) tailored for gaming reviews.
    Applies sentiment analysis to the processed_review column using SentimentIntensityAnalyzer: Assigns a sentiment score (numeric value) to each review.
    Classifies reviews into sentiment categories (very_positive, positive, etc.) based on the score.
    Maps sentiment labels to descriptive names (e.g., "Highly Positive Sentiments").
    Saves the updated DataFrame (with sentiment scores, labels, and descriptions) to the specified file.
    """
    sia = SentimentIntensityAnalyzer()
    # Extending the custom sentiment lexicon with more words and scores. 
    # game review data is precise to how gamers talk. we need to help it identify without explicit labels.
    # these are common words that game reviews use
    sia.lexicon.update(custom_lexicon)
    
    # Directly apply sentiment analysis to each review
    clustered_df['sentiment_score'] = clustered_df['processed_review'].apply(
        lambda x: sia.polarity_scores(x)['compound'])

    # Apply individual review sentiment classification with adjusted thresholds
    # Apply individual review sentiment classification with new thresholds
    clustered_df['sentiment_label'] = clustered_df['sentiment_score'].apply(
        lambda score: 'very_positive' if score > 0.7 else  # Higher threshold for "very_positive"
                  'positive' if score > 0.3 else       # Adjust for more "positive"
                  'neutral' if score >= -0.3 else      # Broader range for "neutral"
                  'negative' if score > -0.7 else      # Adjust for more "negative"
                  'very_negative'                      # Lower threshold for "very_negative"
        )


    # Update sentiment descriptions to reflect individual sentiments
    sentiment_labels = {
        'very_negative': 'Highly Negative Sentiments',
        'negative': 'Negative Sentiments',
        'neutral': 'Neutral Sentiments',
        'positive': 'Positive Sentiments',
        'very_positive': 'Highly Positive Sentiments'
    }

    # Map sentiment descriptions based on individual labels
    clustered_df['sentiment_description'] = clustered_df['sentiment_label'].map(sentiment_labels)

    print("TESTING LABEL DISTRIBUTION:",clustered_df['sentiment_label'].value_counts())
    for label in ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']:
        print("Review a few examples from each sentiment category:",
              clustered_df[clustered_df['sentiment_label'] == label].sample(3))

    # Save the updated DataFrame to CSV
    clustered_df.to_csv(output_filepath, index=False)



def main():
    """
    Reads the input dataset (processed_reviews.csv).
    Prepares the text data using TF-IDF vectorization with specific parameters (max_features=100, ngram_range=(1, 2)).
    Specifies the number of clusters (num_clusters=7).
    Iterates over each unique game (identified by game_id and game_name) in the dataset:
    Calls cluster_reviews_for_game to cluster reviews for the game.
    Appends the clustered reviews to a list.
    Combines all clustered reviews into a single DataFrame.
    Calls analyze_and_save_results to perform sentiment analysis on the clustered data and save it to a CSV file.
    """
    df = pd.read_csv('data/processed_reviews.csv')
    df['processed_review'] = df['processed_review'].fillna('')

    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2)) 
    #min_df=3, max_df=0.8
    #  If min_df is set to 5, it means that a word must appear in at least 5 different game reviews to be included in the analysis
    #  If max_df is set to 0.7 (or 70%), it means that any word appearing in more than 70% of the game reviews will be excluded from the analysis
    num_clusters = 3
    
    all_clustered_reviews = []
    unique_games = df[['game_id', 'game_name']].drop_duplicates().values
    
    for game_id, game_name in unique_games:
        clustered_reviews = cluster_reviews_for_game(df, game_id, game_name, vectorizer, num_clusters)
        if clustered_reviews is not None:
            all_clustered_reviews.append(clustered_reviews)
    
    final_clustered_df = pd.concat(all_clustered_reviews, ignore_index=True)
    
    analyze_and_save_results(final_clustered_df, 'data/KMEANS_cluster.csv')
    
    print("Clustering, sentiment analysis, and label assignment completed. Results saved to 'data/KMEANS_cluster_TESTres.csv'")

if __name__ == "__main__":
    main()

