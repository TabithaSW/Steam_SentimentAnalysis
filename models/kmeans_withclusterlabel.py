import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import silhouette_score, davies_bouldin_score

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
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
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

    # Extending the custom sentiment lexicon with more words and scores. game review data is precise to how gamers talk. we need to help it identify without explicit labels.
    sia.lexicon.update( {
        # Positive sentiment words (scores range from +1 to +4)
        'good': 2.0,
        'great': 3.0,
        'awesome': 3.5,
        'perfect': 4.0,
        'fun': 3.0,
        'enjoyable': 3.0,
        'excellent': 4.0,
        'amazing': 3.5,
        'fantastic': 3.5,
        'best': 4.0,
        'incredible': 3.5,
        'epic': 3.0,
        'satisfying': 3.0,
        'engrossing': 3.0,
        'masterpiece': 4.0,
        'addictive': 2.5,
        'innovative': 3.0,
        'superb': 3.5,
        'thrilling': 3.0,
        'charming': 3.0,
        'delightful': 3.0,
        'top-notch': 3.5,
        'stellar': 3.5,
        'captivating': 3.0,
        'polished': 3.0,
        
        # Negative sentiment words (scores range from -1 to -4)
        'bad': -2.0,
        'terrible': -3.0,
        'awful': -3.5,
        'worst': -4.0,
        'boring': -3.0,
        'dull': -2.5,
        'disappointing': -3.0,
        'poor': -2.0,
        'buggy': -3.0,
        'frustrating': -3.0,
        'uninspired': -2.0,
        'lackluster': -2.5,
        'mediocre': -2.0,
        'tedious': -3.0,
        'clunky': -2.5,
        'unbalanced': -2.0,
        'repetitive': -2.5,
        'grindy': -2.0,
        'unpolished': -2.5,
        'monotonous': -2.0,
        'broken': -3.5,
        'pay-to-win': -4.0,
        'underwhelming': -2.0,
        'glitchy': -3.0,
        'forgettable': -2.0,
        'overpriced': -2.5,
        
        # General sentiment words
        'beautiful': 3.0,
        'hard': -1.0,
        'easy': 1.0,
        'challenging': 1.0,
        'difficult': -1.0,
        'simple': 1.0,
        'complex': 0
    })
    # Directly apply sentiment analysis to each review
    clustered_df['sentiment_score'] = clustered_df['processed_review'].apply(
        lambda x: sia.polarity_scores(x)['compound'])

    # Apply individual review sentiment classification with adjusted thresholds
    clustered_df['sentiment_label'] = clustered_df['sentiment_score'].apply(
        lambda score: 'very_positive' if score > 0.6 else  # Adjusting thresholds for more granularity
                      'positive' if score > 0.2 else
                      'neutral' if score >= -0.2 else
                      'negative' if score > -0.6 else
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

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3), min_df=5, max_df=0.7)
    num_clusters = 5
    
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

