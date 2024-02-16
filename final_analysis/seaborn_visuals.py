import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

# Load your dataset
df = pd.read_csv('data/KMEANS_cluster_res.csv')

# Setting up the color palette for a red/black gaming theme
sns.set(style="darkgrid", palette="dark")  # Adjust Seaborn style and palette

# Initialize the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(18, 12))

# Sentiment Distribution Across Clusters
sns.boxplot(x='cluster', y='sentiment_score', data=df, ax=axes[0, 0], palette="RdGy")
axes[0, 0].set_title('Sentiment Distribution Across Clusters')
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Sentiment Score')

# Calculating average sentiment score per game and plotting
avg_sentiment_per_game = df.groupby('game_name')['sentiment_score'].mean().sort_values()
barplot = sns.barplot(x=avg_sentiment_per_game.values, y=avg_sentiment_per_game.index, ax=axes[0, 1], palette="RdGy", edgecolor="w")
axes[0, 1].set_title('Average Sentiment Score per Game')
axes[0, 1].set_xlabel('Average Sentiment Score')
axes[0, 1].set_ylabel('Game')

# Adjusting the y-axis labels for better readability
for label in axes[0, 1].get_yticklabels():
    label.set_size(8)
    label.set_style('italic')
    label.set_color('gray')
# Word Cloud for Cluster 0
cluster_reviews = ' '.join(df[df['cluster'] == 0]['processed_review'].astype(str))  # Ensure all entries are strings
wordcloud = WordCloud(background_color='black', width=800, height=400, colormap="Reds").generate(cluster_reviews)
axes[1, 0].imshow(wordcloud, interpolation='bilinear')
axes[1, 0].axis('off')  # Hide axis
axes[1, 0].set_title('Word Cloud for Cluster 0')

# Comparison of Sentiment Scores Between Clusters using a violin plot
sns.violinplot(x='cluster', y='sentiment_score', data=df, ax=axes[1, 1], palette=red_black_palette)
axes[1, 1].set_title('Comparison of Sentiment Scores Between Clusters')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Sentiment Score')

# Adjust layout to not overlap
plt.tight_layout()
plt.show()
