import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your dataset
# df = pd.read_csv('data/KMEANS_cluster_res.csv')
df = pd.read_csv('data/KMEANS_cluster_TESTres.csv')

# Setting up the color palette for a red/black gaming theme
sns.set(style="darkgrid", palette="dark")  # Adjust Seaborn style and palette

# Initialize the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(18, 12))

# Sentiment Distribution Across Clusters
sns.boxplot(x='cluster', y='sentiment_score', data=df, ax=axes[0, 0], palette="RdGy")
axes[0, 0].set_title('Sentiment Distribution Across Clusters')
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Sentiment Score')

# Comparison of Sentiment Scores Between Clusters using a violin plot
sns.violinplot(x='cluster', y='sentiment_score', data=df, ax=axes[0, 1], palette="RdGy")
axes[0, 1].set_title('Comparison of Sentiment Scores Between Clusters')
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Sentiment Score')

# Adjusting the y-axis labels for better readability
for label in axes[0, 1].get_yticklabels():
    label.set_size(8)
    label.set_style('italic')
    label.set_color('gray')

# Distribution of sentiment scores across all reviews
sns.histplot(df['sentiment_score'], ax=axes[1, 1], color='red', kde=True)
axes[1, 1].set_title('Distribution of Sentiment Scores')
axes[1, 1].set_xlabel('Sentiment Score')
axes[1, 1].set_ylabel('Frequency')

# Calculating average sentiment score per game and plotting
avg_sentiment_per_game = df.groupby('game_name')['sentiment_score'].mean().sort_values()
barplot = sns.barplot(x=avg_sentiment_per_game.values, y=avg_sentiment_per_game.index, ax=axes[1, 0], palette="RdGy", edgecolor="w")
axes[1, 0].set_title('Average Sentiment Score per Game')
axes[1, 0].set_xlabel('Average Sentiment Score')
axes[1, 0].set_ylabel('Game')

# Adjusting the y-axis labels for better readability and preventing overlap
axes[1, 0].set_yticklabels([text[:15] + ('...' if len(text) > 15 else '') for text in avg_sentiment_per_game.index], fontsize=8, style='italic', color='gray')
axes[1, 0].tick_params(axis='y', rotation=0)

# Adjust layout to not overlap
plt.tight_layout()
plt.savefig('seaborn_plots.jpg', format='jpg')
plt.show()

