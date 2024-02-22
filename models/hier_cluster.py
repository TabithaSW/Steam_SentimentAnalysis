"""

Hierarchical Clustering builds a tree of clusters without requiring the number of clusters to be specified beforehand. 
It's more informative for understanding the data structure and can reveal how clusters are related at different levels of granularity.

For hierarchical clustering, since we're working with a dendrogram, the decision on the number of clusters might be more subjective and based on the dendrogram's inspection.
However, we can still compute the Silhouette Score post-hoc after deciding on the number of clusters.


"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from nltk.sentiment import SentimentIntensityAnalyzer

def cluster_reviews_for_game_hierarchical(df, game_id, game_name, vectorizer, n_clusters, sia):
    """
    Clusters reviews for a specific game using hierarchical clustering, evaluates the clustering,
    calculates sentiment scores, and includes the silhouette score for the clustering.
    """
    game_reviews = df[df['game_id'] == game_id].copy()
    if len(game_reviews) < n_clusters:
        print(f"Not enough reviews for clustering Game ID: {game_id} - {game_name}")
        return None

    X = vectorizer.fit_transform(game_reviews['processed_review'])
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
    game_reviews['cluster'] = cluster_model.fit_predict(X.toarray())

    # Calculate sentiment scores for each review
    game_reviews['sentiment_score'] = game_reviews['processed_review'].apply(lambda review: sia.polarity_scores(review)['compound'])

    # Evaluate clustering performance and append the silhouette score for each review
    silhouette_avg = silhouette_score(X, game_reviews['cluster'])
    game_reviews['silhouette_score'] = silhouette_avg  # Append silhouette score to each row

    print(f"Game ID: {game_id} - {game_name}, Silhouette Score: {silhouette_avg}")
    
    return game_reviews

def main():
    df = pd.read_csv('data/processed_reviews.csv')
    df['processed_review'] = df['processed_review'].fillna('')
    vectorizer = TfidfVectorizer(max_features=30, ngram_range=(1, 3))

    # Initialize Sentiment Intensity Analyzer and update lexicon with gaming-specific terms
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update({
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
        'disconnect': -2.5, # being disconnected from a game is negative
        'paywall': -3.0, # restrictions behind payments are seen as negative
        'grindfest': -2.0, # games that require excessive grinding are viewed negatively
        'cashgrab': -3.5, # games designed just to make money are seen as negative
        'clone': -1.5, # unoriginal games that copy others are seen negatively
        'bugfest': -2.5, # games with many bugs are viewed negatively
        'pay-to-progress': -3.0, # negative view on pay for progression systems
        'RNG': -1.0, # randomness can be negative if it affects progression
        'toxic': -2.5, # toxic gaming communities are seen negatively
        'cheater': -3.0, # cheating is a highly negative aspect
        'hacker': -3.0, # hacking negatively affects gameplay
        'broken': -3.0, #
        'beautiful': 3.0,
        'hard': -1.0,
        'easy': 1.0,
        'challenging': 1.0,
        'difficult': -1.0,
        'simple': 1.0,
        'complex': 0,
        'nerfed': -2.0, # often seen as negative if a favorite feature is weakened
        'buffed': 2.5, # improvements or enhancements are usually seen as positive
        'grinding': -1.0, # can be seen as negative due to repetitiveness
        'noob-friendly': 1.5, # accessibility can be positive
        'nerf': -1.5, # similar to 'nerfed'
        'buff': 2.0, # similar to 'buffed'
        'PvP': 1.5, # player versus player, can be exciting and positive
        'PvE': 1.5, # player versus environment, also generally positive
        'raid': 2.0, # can be a positive experience in games
        'loot': 2.5, # acquiring items is positive
        'crafting': 2.0, # creating items is often a rewarding experience
        'quest': 1.5, # can indicate engaging content
        'skill-tree': 1.5, # customization is generally seen as positive
        'modding': 2.0, # the ability to modify a game is often positive
        'co-op': 2.5, # cooperative play is usually a positive aspect
        'DLC': 1.0, # downloadable content can be positive but context matters
        'expansion': 2.5, # additional content is usually seen as positive
        'AAA': 1.0, # high-quality, high-budget games
        'indie': 1.5, # independent games can be seen positively for creativity
        'early access': 1.0, # can be positive for anticipation but context matters
        'sandbox': 2.0, # games with freedom are often viewed positively
        'freemium': -1.5, # often seen as negative due to microtransactions
        'microtransaction': -3.0, # generally seen as a negative aspect
        'multiplayer': 2.0, # playing with others can be positive
        'singleplayer': 1.5, # a solid single-player experience is often praised
        'lag': -2.5, # performance issues are negative
        'fps': 1.0, # frames per second, high is good, but context is key
        'ping': -1.0, # low ping is good, high ping is bad, context is important
        'replayability': 2.5, # the ability to play a game many times is positive
        'controller support': 1.5, # support for different controllers can be positive
        'crossplay': 2.0, # the ability to play across different platforms is positive
        'procedural generation': 1.0, # can be positive for game variety
        'roguelike': 1.5, # a genre that can be seen positively
        'metroidvania': 1.5, # a genre that's often well-received
        'turn-based': 1.0, # a type of gameplay that can be positive
        'real-time': 1.0, # another type of gameplay that can be positive
        'immersive': 3.0, # a highly desired trait in games
        'open-world': 2.5, # open-world games are often seen positively
        })

    all_clustered_reviews = []
    unique_games = df[['game_id', 'game_name']].drop_duplicates().values
    for game_id, game_name in unique_games:
        clustered_reviews = cluster_reviews_for_game_hierarchical(df, game_id, game_name, vectorizer, n_clusters=3, sia=sia)
        if clustered_reviews is not None:
            all_clustered_reviews.append(clustered_reviews)

    final_clustered_df = pd.concat(all_clustered_reviews, ignore_index=True)
    final_clustered_df.to_csv('data/Hcluster_res_silscore.csv', index=False)
    
    print("Hierarchical clustering, sentiment analysis, and silhouette scores completed. Results saved to 'data/HIERARCHICAL_cluster_results_with_silhouette.csv'")

if __name__ == "__main__":
    main()
