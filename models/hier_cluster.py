"""

Hierarchical Clustering builds a tree of clusters without requiring the number of clusters to be specified beforehand. 
It's more informative for understanding the data structure and can reveal how clusters are related at different levels of granularity.

For hierarchical clustering, since we're working with a dendrogram, the decision on the number of clusters might be more subjective and based on the dendrogram's inspection.
However, we can still compute the Silhouette Score post-hoc after deciding on the number of clusters.


"""

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

def cluster_reviews_for_game_hierarchical(df, game_id, game_name, vectorizer, n_clusters):
    """
    Clusters reviews for a specific game using hierarchical clustering and evaluates the clustering.
    """
    game_reviews = df[df['game_id'] == game_id].copy()
    if len(game_reviews) < n_clusters:
        print(f"Not enough reviews for clustering Game ID: {game_id} - {game_name}")
        return None

    X = vectorizer.fit_transform(game_reviews['processed_review'])
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')

    game_reviews['cluster'] = cluster_model.fit_predict(X.toarray())

    # Evaluate clustering performance
    silhouette_avg = silhouette_score(X, game_reviews['cluster'])
    print(f"Game ID: {game_id} - {game_name}, Silhouette Score: {silhouette_avg}")
    
    return game_reviews

def main():
    df = pd.read_csv('data/processed_reviews.csv')
    df['processed_review'] = df['processed_review'].fillna('')
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    unique_games = df[['game_id', 'game_name']].drop_duplicates().values

    all_clustered_reviews = []
    for game_id, game_name in unique_games:
        clustered_reviews = cluster_reviews_for_game_hierarchical(df, game_id, game_name, vectorizer, n_clusters=3)
        if clustered_reviews is not None:
            all_clustered_reviews.append(clustered_reviews)

    final_clustered_df = pd.concat(all_clustered_reviews, ignore_index=True)
    final_clustered_df.to_csv('data/HIERARCHICAL_cluster_results.csv', index=False)
    
    print("Hierarchical clustering and sentiment analysis completed. Results saved to 'data/HIERARCHICAL_cluster_results.csv'")

if __name__ == "__main__":
    main()
