
# Sentiment Distribution Across Clusters
# Visualize the distribution of sentiment scores within each cluster to see how sentiments vary across different themes or topics identified by KMeans clustering.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your dataset
df = pd.read_csv('data/KMEANS_cluster_res.csv')

# Plotting sentiment distribution across clusters
plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster', y='sentiment_score', data=df)
plt.title('Sentiment Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Sentiment Score')
plt.show()

# Calculating average sentiment score per game
avg_sentiment_per_game = df.groupby('game_name')['sentiment_score'].mean().sort_values()

# Plotting
plt.figure(figsize=(12, 8))
avg_sentiment_per_game.plot(kind='barh')
plt.title('Average Sentiment Score per Game')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Game')
plt.tight_layout()  # Adjust layout to make room for the game names
plt.show()

from wordcloud import WordCloud

# Generating a word cloud for a single cluster as an example
cluster_reviews = ' '.join(df[df['cluster'] == 0]['processed_review'])  # Adjust cluster number as needed
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(cluster_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axis
plt.title('Word Cloud for Cluster 0')
plt.show()

# Example: Visual comparison using a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='cluster', y='sentiment_score', data=df)
plt.title('Comparison of Sentiment Scores Between Clusters')
plt.xlabel('Cluster')
plt.ylabel('Sentiment Score')
plt.show()
