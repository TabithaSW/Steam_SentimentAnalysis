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
# The TF-IDF Vectorizer transforms text data into numerical features suitable for clustering.
# The max_features=100 limits the vocabulary size, making clustering more efficient and avoiding overfitting on infrequent terms.



# Range of clusters to try
cluster_range = range(2, 10)  
# 2 to 10, which is a typical range to explore in clustering.

silhouette_scores = []

# Compute the silhouette scores for each k
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For k={k}, the average silhouette score is: {silhouette_avg}")


# Determine the optimal number of clusters
optimal_k = cluster_range[np.argmax(silhouette_scores)]
print(f"The optimal number of clusters is: {optimal_k}")

# Print the highest silhouette score
highest_score = max(silhouette_scores)
print(f"The highest silhouette score is: {highest_score}, achieved at k={optimal_k}")
"""
Silhouette Score measures how well-separated the clusters are:
A score close to +1 means clusters are well-separated and distinct.
A score close to 0 means clusters overlap significantly.
A score close to -1 means data points are assigned to the wrong clusters.

"""

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

# By testing multiple k values, we ensure that the clusters chosen are the most appropriate for the data/reviews
"""
Sample Output:
For k=2, the average silhouette score is: 0.02584101455284097
For k=3, the average silhouette score is: 0.033431033207707524
For k=4, the average silhouette score is: 0.03899271631970048
For k=5, the average silhouette score is: 0.04424167733013159
For k=6, the average silhouette score is: 0.04366900754649064
For k=7, the average silhouette score is: 0.045631878634992164
For k=8, the average silhouette score is: 0.05477541035139925
For k=9, the average silhouette score is: 0.056440804774007876 (Highest score, best number of clusters to use)
For k=10, the average silhouette score is: 0.055666342520301094
"""