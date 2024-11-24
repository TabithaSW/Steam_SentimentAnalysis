import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_clusters_by_game(df, game_name):
    """
    Analyzes the distribution of clusters for a specific game, arranging plots in a 2x2 grid for improved readability.
    """
    game_df = df[df['game_name'] == game_name]

    # Initialize the matplotlib figure for a 2x2 grid
    f, axes = plt.subplots(2, 2, figsize=(18, 16))  # Adjust figure size for better fit and readability

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    # Adjust Seaborn style for a professional and clean look
    sns.set(style="darkgrid")
    sns.set_context("talk")  # Enhance text readability

    # Use a sophisticated color palette
    palette = sns.color_palette("coolwarm")

    # Sentiment Distribution Across Clusters
    sns.boxplot(x='cluster', y='sentiment_score', data=game_df, ax=axes[0], palette=palette, linewidth=2.5)
    axes[0].set_title(f'Sentiment Distribution - {game_name}')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Sentiment Score')
    axes[0].grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.5)

    # Comparison of Sentiment Scores Between Clusters
    sns.violinplot(x='cluster', y='sentiment_score', data=game_df, ax=axes[1], palette=palette, linewidth=2.5)
    axes[1].set_title(f'Comparison of Sentiment Scores - {game_name}')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Sentiment Score')
    axes[1].grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.5)

    # Distribution of sentiment scores across all reviews
    sns.histplot(game_df['sentiment_score'], ax=axes[2], color="slateblue", kde=True)
    axes[2].set_title(f'Sentiment Score Distribution - {game_name}')
    axes[2].set_xlabel('Sentiment Score')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.5)

    # Count sentiment scores per cluster
    sentiment_counts = game_df.groupby('cluster')['sentiment_score'].count().reset_index()
    sns.barplot(x='cluster', y='sentiment_score', data=sentiment_counts, ax=axes[3], palette=palette, linewidth=2.5)
    axes[3].set_title(f'Sentiment Score Count by Cluster - {game_name}')
    axes[3].set_xlabel('Cluster')
    axes[3].set_ylabel('Sentiment Score Count')
    axes[3].grid(True, linestyle='--', linewidth=1, color='gray', alpha=0.5)

    # Adjust layout to not overlap
    plt.tight_layout()
    plt.show()

def main():
    # Load your dataset
    df = pd.read_csv('data/KMEANS_cluster.csv')

    # Get unique game names
    unique_games = df['game_name'].unique()

    # Analyze clusters for each game separately
    for game_name in unique_games:
        analyze_clusters_by_game(df, game_name)

if __name__ == "__main__":
    main()
