import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_avg_sentiment_per_game(df):
    """
    Calculates and plots the average sentiment score per game with specified visual enhancements.
    """
    avg_sentiment_per_game = df.groupby('game_name')['sentiment_score'].mean().sort_values()
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x=avg_sentiment_per_game.values, y=avg_sentiment_per_game.index, palette="coolwarm", edgecolor="black")
    
    # Add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Italicize the game titles
    plt.yticks(fontstyle='italic')
    
    plt.title('Average Sentiment Score per Game')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Game')
    plt.tight_layout()
    plt.show()

def main():
    # Load your dataset
    df = pd.read_csv('data/Hcluster_res_silscore.csv')

    # Plot average sentiment score per game
    plot_avg_sentiment_per_game(df)

if __name__ == "__main__":
    main()
