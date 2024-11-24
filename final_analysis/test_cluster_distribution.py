import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/KMEANS_cluster.csv')

# Select only the top 5 games (by alphabetical order or another criterion)
top_5_games = df['game_name'].unique()[:5]  # Adjust selection criterion as needed
df_filtered = df[df['game_name'].isin(top_5_games)]

# Group the data by game and cluster, then summarize sentiment scores
distribution_summary = df_filtered.groupby(['game_name', 'cluster'])['sentiment_score'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()

# Export the summary to a CSV (optional)
distribution_summary.to_csv('data/cluster_sentiment_distribution_summary.csv', index=False)

# Set the style and color palette
sns.set(style="whitegrid")
color_palette = sns.color_palette("tab10")

# Get the unique clusters
clusters = distribution_summary['cluster'].unique()
num_clusters = len(clusters)

# Create subplots for each cluster
fig, axes = plt.subplots(nrows=1, ncols=num_clusters, figsize=(16, 6), sharey=True)

for idx, cluster in enumerate(clusters):
    ax = axes[idx]
    cluster_data = distribution_summary[distribution_summary['cluster'] == cluster]
    sns.barplot(
        x='game_name', 
        y='mean', 
        data=cluster_data, 
        ax=ax, 
        palette=color_palette, 
        edgecolor="black"
    )
    ax.set_title(f'Cluster {cluster}', fontsize=12)
    ax.set_xlabel('Game Name', fontsize=10)
    ax.set_ylabel('Mean Sentiment' if idx == 0 else '', fontsize=10)
    ax.tick_params(axis='x', labelsize=8, rotation=90)  # Vertical x-axis labels

# Add a global title
fig.suptitle('Mean Sentiment by Cluster and Game', fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
plt.savefig('new_figures/mean_sentiment_by_cluster_top5.png', dpi=300)

# Show the figure
plt.show()

