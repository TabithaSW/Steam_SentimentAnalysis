import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your dataset
df = pd.read_csv('data/processed_reviews.csv')

# Replace NaN values with an empty string
df['processed_review'].fillna('', inplace=True)

# Preprocess and vectorize your text data
vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as necessary
X = vectorizer.fit_transform(df['processed_review'])

# Range of clusters to try
cluster_range = range(2, 11)  # For example, trying from 2 to 10 clusters

silhouette_scores = []

# Compute the silhouette scores for each k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For k={k}, the average silhouette score is: {silhouette_avg}")

# Plot the silhouette scores
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, silhouette_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method For Optimal k')
plt.show()

# Determine the optimal number of clusters
optimal_k = cluster_range[np.argmax(silhouette_scores)]
print(f"The optimal number of clusters is: {optimal_k}")
