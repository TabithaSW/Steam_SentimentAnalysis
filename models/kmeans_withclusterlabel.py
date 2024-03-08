import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from custom_lexicon import custom_lexicon


"""

K-Means is a centroid-based clustering method, which partitions the data into K clusters by minimizing the variance within each cluster. 
It requires specifying the number of clusters in advance and works well for spherical clusters.

"""

def cluster_reviews_for_game(df, game_id, game_name, vectorizer, num_clusters):
    """
    Clusters reviews for a specific game and evaluates the clustering using Silhouette Score and Davies-Bouldin Index.
    """
    game_reviews = df[df['game_id'] == game_id].copy()
    if len(game_reviews) < num_clusters:
        print(f"Not enough reviews for clustering Game ID: {game_id} - {game_name}")
        return None

    X = vectorizer.fit_transform(game_reviews['processed_review'])
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

def analyze_and_save_results(clustered_df, output_filepath):
    sia = SentimentIntensityAnalyzer()


    # Extending the custom sentiment lexicon with more words and scores. 
    # game review data is precise to how gamers talk. we need to help it identify without explicit labels.
    # these are common words that game reviews use
    sia.lexicon.update(custom_lexicon)
    
    # Directly apply sentiment analysis to each review
    clustered_df['sentiment_score'] = clustered_df['processed_review'].apply(
        lambda x: sia.polarity_scores(x)['compound'])

    # Apply individual review sentiment classification with adjusted thresholds
    clustered_df['sentiment_label'] = clustered_df['sentiment_score'].apply(
        lambda score: 'very_positive' if score > 0.5 else  # Adjusting thresholds for more granularity
                      'positive' if score > 0.2 else
                      'neutral' if score >= -0.2 else
                      'negative' if score > -0.5 else
                      'very_negative')

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

    # Save the updated DataFrame to CSV
    clustered_df.to_csv(output_filepath, index=False)



def main():
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
    
    analyze_and_save_results(final_clustered_df, 'data/KMEANS_cluster_TESTres.csv')
    
    print("Clustering, sentiment analysis, and label assignment completed. Results saved to 'data/KMEANS_cluster_TESTres.csv'")

if __name__ == "__main__":
    main()

