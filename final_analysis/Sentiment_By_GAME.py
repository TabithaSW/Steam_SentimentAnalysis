import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_clusters_by_game(df, game_name):
    """
    Analyzes the distribution of clusters for a specific game.
    """
    game_df = df[df['game_name'] == game_name]

    # Initialize the matplotlib figure
    f, axes = plt.subplots(3, 1, figsize=(12, 15))  # Adjust figure size

    # Setting up the color palette for a red/black gaming theme
    sns.set(style="darkgrid", palette="dark")  # Adjust Seaborn style and palette

    # Sentiment Distribution Across Clusters
    sns.boxplot(x='cluster', y='sentiment_score', data=game_df, ax=axes[0], palette="RdGy")
    axes[0].set_title(f'Sentiment Distribution - {game_name}')  # Shorten title
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Sentiment Score')

    # Comparison of Sentiment Scores Between Clusters using a violin plot
    sns.violinplot(x='cluster', y='sentiment_score', data=game_df, ax=axes[1], palette="RdGy")
    axes[1].set_title(f'Comparison of Sentiment Scores - {game_name}')  # Shorten title
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Sentiment Score')

    # Distribution of sentiment scores across all reviews
    sns.histplot(game_df['sentiment_score'], ax=axes[2], color='red', kde=True)
    axes[2].set_title(f'Sentiment Score Distribution - {game_name}')  # Shorten title
    axes[2].set_xlabel('Sentiment Score')
    axes[2].set_ylabel('Frequency')

    # Adjust layout to not overlap
    plt.tight_layout()
    plt.show()

def main():
    # Load your dataset
    df = pd.read_csv('data/KMEANS_cluster_TESTres.csv')

    # Get unique game names
    unique_games = df['game_name'].unique()

    # Analyze clusters for each game separately
    for game_name in unique_games:
        analyze_clusters_by_game(df, game_name)

if __name__ == "__main__":
    main()
