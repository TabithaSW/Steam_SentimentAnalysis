import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your dataset
df = pd.read_csv('data/KMEANS_cluster_res.csv')

# Setting up the color palette for a red/black gaming theme
sns.set(style="darkgrid", palette="dark")  # Adjust Seaborn style and palette

# Get unique game names
unique_games = df['game_name'].unique()

# Initialize the matplotlib figure
for game_name in unique_games:
    game_df = df[df['game_name'] == game_name]

    # Initialize the matplotlib figure for each game
    f, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Sentiment Distribution Across Clusters
    sns.boxplot(x='cluster', y='sentiment_score', data=game_df, ax=axes[0, 0], palette="RdGy")
    axes[0, 0].set_title(f'Sentiment Distribution Across Clusters - {game_name}')
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Sentiment Score')

    # Calculating average sentiment score per game and plotting
    avg_sentiment_per_game = game_df.groupby('game_name')['sentiment_score'].mean().sort_values()
    barplot = sns.barplot(x=avg_sentiment_per_game.values, y=avg_sentiment_per_game.index, ax=axes[0, 1], palette="RdGy", edgecolor="w")
    axes[0, 1].set_title(f'Average Sentiment Score per Game - {game_name}')
    axes[0, 1].set_xlabel('Average Sentiment Score')
    axes[0, 1].set_ylabel('Game')

    # Adjusting the y-axis labels for better readability
    for label in axes[0, 1].get_yticklabels():
        label.set_size(8)
        label.set_style('italic')
        label.set_color('gray')

    # Comparison of Sentiment Scores Between Clusters using a violin plot
    sns.violinplot(x='cluster', y='sentiment_score', data=game_df, ax=axes[1, 0], palette="RdGy")
    axes[1, 0].set_title(f'Comparison of Sentiment Scores Between Clusters - {game_name}')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Sentiment Score')

    # Adjust layout to not overlap
    plt.tight_layout()
    plt.show()
