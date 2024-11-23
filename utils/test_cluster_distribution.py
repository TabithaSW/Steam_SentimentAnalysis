import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('data/KMEANS_cluster_TESTres.csv')

# Group the data by game and cluster, then summarize sentiment scores
distribution_summary = df.groupby(['game_name', 'cluster'])['sentiment_score'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()

# View the distribution summary for each game and cluster
print(distribution_summary)

# If you want to export this summary to a CSV
distribution_summary.to_csv('data/cluster_sentiment_distribution_summary.csv', index=False)

sns.barplot(x='cluster', y='mean', hue='game_name', data=distribution_summary)
plt.title('Mean Sentiment by Cluster and Game')
plt.show()
