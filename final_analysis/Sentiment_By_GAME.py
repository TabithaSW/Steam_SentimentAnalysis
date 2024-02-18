import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_clusters_by_game(df, game_name):
    """
    Analyzes the distribution of clusters for a specific game.
    """
    game_df = df[df['game_name'] == game_name]

    # Initialize the matplotlib figure
    f, axes = plt.subplots(4, 1, figsize=(12, 18))  # Adjust figure size

    # Setting up the color palette for a rainbow theme
    sns.set(style="whitegrid", palette="rainbow")  # Adjust Seaborn style and palette

    # Sentiment Distribution Across Clusters
    sns.boxplot(x='cluster', y='sentiment_score', data=game_df, ax=axes[0], palette="rainbow", linewidth=1)
    axes[0].set_title(f'Sentiment Distribution - {game_name}')  # Shorten title
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Sentiment Score')
    axes[0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # Comparison of Sentiment Scores Between Clusters using a violin plot
    sns.violinplot(x='cluster', y='sentiment_score', data=game_df, ax=axes[1], palette="rainbow", linewidth=1)
    axes[1].set_title(f'Comparison of Sentiment Scores - {game_name}')  # Shorten title
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Sentiment Score')
    axes[1].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # Distribution of sentiment scores across all reviews
    sns.histplot(game_df['sentiment_score'], ax=axes[2], color='red', kde=True)
    axes[2].set_title(f'Sentiment Score Distribution - {game_name}')  # Shorten title
    axes[2].set_xlabel('Sentiment Score')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # Count sentiment scores per cluster
    sentiment_counts = game_df.groupby('cluster')['sentiment_score'].count().reset_index()

    # Plot sentiment score counts per cluster
    sns.barplot(x='cluster', y='sentiment_score', data=sentiment_counts, ax=axes[3], palette="rainbow", linewidth=1)
    axes[3].set_title(f'Sentiment Score Count by Cluster - {game_name}')  # Shorten title
    axes[3].set_xlabel('Cluster')
    axes[3].set_ylabel('Sentiment Score Count')
    axes[3].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # Annotate each bar with the count of sentiment scores
    for p in axes[3].patches:
        axes[3].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                         textcoords='offset points')

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
