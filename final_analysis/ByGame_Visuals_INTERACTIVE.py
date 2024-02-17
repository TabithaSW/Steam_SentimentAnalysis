import pandas as pd
import plotly.graph_objects as go
from ipywidgets import interact, Dropdown

# Load your dataset
df = pd.read_csv('data/KMEANS_cluster_res.csv')

# Define a function to update plots based on selected game
def update_plots(selected_game):
    # Filter data for the selected game
    filtered_df = df[df['game_name'] == selected_game]
    
    # Sentiment Distribution Across Clusters
    fig1 = go.Figure(go.Box(x=filtered_df['cluster'], y=filtered_df['sentiment_score'], 
                             title='Sentiment Distribution Across Clusters'))
    
    # Comparison of Sentiment Scores Between Clusters using a violin plot
    fig2 = go.Figure(go.Violin(x=filtered_df['cluster'], y=filtered_df['sentiment_score'],
                                title='Comparison of Sentiment Scores Between Clusters'))
    
    # Distribution of sentiment scores across all reviews
    fig3 = go.Figure(go.Histogram(x=filtered_df['sentiment_score'], 
                                  title='Distribution of Sentiment Scores'))
    
    # Calculating average sentiment score per game and plotting
    avg_sentiment_per_game = filtered_df.groupby('game_name')['sentiment_score'].mean().sort_values()
    fig4 = go.Figure(go.Bar(x=avg_sentiment_per_game.values, y=avg_sentiment_per_game.index, 
                             orientation='h', title='Average Sentiment Score per Game'))
    
    # Update layout
    for fig in [fig1, fig2, fig3, fig4]:
        fig.update_layout(xaxis_title='', yaxis_title='Sentiment Score')
    
    return fig1, fig2, fig3, fig4

# Get unique game names for dropdown options
game_names = df['game_name'].unique()

# Define dropdown widget for game selection
game_dropdown = Dropdown(options=game_names, description='Select a Game:')

# Define callback function for dropdown selection
def dropdown_callback(selected_game):
    fig1, fig2, fig3, fig4 = update_plots(selected_game)
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()

# Display the dropdown widget with callback
interact(dropdown_callback, selected_game=game_dropdown)
