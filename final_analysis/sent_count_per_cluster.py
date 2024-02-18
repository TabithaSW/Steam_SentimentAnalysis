import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_clusters_by_game(df, game_name):
    """
    Analyzes the distribution of clusters for a specific game.
    """
    game_df = df[df['game_name'] == game_name]

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size

    # Setting up the color palette for a red/black gaming theme
    sns.set(style="whitegrid", palette="Set2")  # Adjust Seaborn style and palette

    # Count sentiment scores per cluster
    sentiment_counts = game_df.groupby('cluster')['sentiment_score'].count().reset_index()

    # Plot sentiment score counts per cluster
    sns.barplot(x='cluster', y='sentiment_score', data=sentiment_counts, ax=ax, palette="Set2", edgecolor='black')
    ax.set_title(f'Sentiment Score Count by Cluster - {game_name}')  # Shorten title
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Sentiment Score Count')

    # Annotate each bar with the count of sentiment scores
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    # Add grid in the background
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

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


