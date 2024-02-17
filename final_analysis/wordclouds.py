import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from matplotlib.gridspec import GridSpec

# Load your dataset
df = pd.read_csv('data/KMEANS_cluster_res.csv')

# Additional stopwords that for whatever reason didn't get removed when tokenized
additional_stopwords = {
    "player", "need", "stop", "thing", "setting", "game", "playing", "hour", "year", "month", "overall",
    "sometimes", "thing", "get", "experience", "feel", "time", "dev", "get", "got", "money", "almost", "job",
    "td tr", "tldr", "etc", "feature", "right", "little", "match", "main", "item", "getting", "account", "reason",
    "everyone", "everything", "person", "place", "work", "server", "full", "b", "ever", "nan", "current", "sure",
    "step", "content", "story", "enough", "review", "two", "one","td","team","devs","u","list","mod","second","seen","said",
    "added","mean","felt","make","looking","map","look"
    }

# Initialize the matplotlib figure with dynamic sizing based on the number of clusters
num_clusters = df['cluster'].nunique()
cols = 2  # Define how many columns you want in your subplot grid
rows = -(-num_clusters // cols)  # Ceiling division to determine the number of rows needed
f = plt.figure(figsize=(20, 6 * rows))
gs = GridSpec(rows, cols, figure=f)

for cluster in range(num_clusters):
    # Combine reviews for each cluster and convert all entries to strings
    cluster_reviews = ' '.join(df[df['cluster'] == cluster]['processed_review'].astype(str))
    # Initialize WordCloud with additional stopwords
    wordcloud = WordCloud(background_color='black', width=500, height=500, colormap="hot",
                          stopwords=additional_stopwords).generate(cluster_reviews)
    ax = f.add_subplot(gs[cluster // cols, cluster % cols])
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud for Cluster {cluster}')

plt.tight_layout()
plt.show()