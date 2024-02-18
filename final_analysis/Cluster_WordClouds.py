import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('data/KMEANS_cluster_TESTres.csv')

# Additional stopwords
# so many stopwords ugh
additional_stopwords = {
    "actually", "also", "although", "always", "am", "among", "amount", "another", "anyone", "anything",
    "around", "away", "back", "because", "become", "becomes", "becoming", "been", "before", "begin",
    "behind", "being", "below", "between", "beyond", "both", "but", "by", "came", "can", "cannot", "come",
    "could", "day", "did", "do", "does", "done", "down", "each", "either", "end", "even", "every", "find",
    "first", "for", "from", "further", "game", "get", "give", "go", "going", "gone", "got", "had", "has",
    "have", "having", "he", "her", "here", "herself", "him", "himself", "his", "how", "however", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "keep", "kept", "know", "later", "least", "less",
    "let", "like", "likely", "lot", "made", "make", "many", "may", "me", "might", "more", "most", "much",
    "must", "my", "myself", "never", "new", "no", "nobody", "none", "nor", "not", "nothing", "now", "of",
    "off", "often", "on", "once", "one", "only", "or", "other", "our", "ours", "out", "over", "own", "part",
    "people", "place", "player", "put", "say", "see", "seem", "seemed", "seems", "several", "she",
    "should", "show", "since", "some", "something", "sometime", "somewhere", "still", "such", "take", "than",
    "that", "the", "their", "them", "then", "there", "therefore", "these", "they", "thing", "this", "those",
    "though", "through", "thus", "time", "to", "together", "too", "toward", "try", "under", "until", "up",
    "upon", "us", "use", "used", "using", "very", "want", "was", "way", "we", "well", "went", "were", "what",
    "when", "where", "whether", "which", "while", "who", "whole", "whose", "why", "will", "with", "within",
    "without", "would", "yet", "you", "your", "yours", "yourself","hour","player","buy","review","game","system",
    "feel","edit","instead","really","playing","baby","character","making","update","think","take",
    "absolutely","point","feel","etc","hit","fact","date","month","item","area","better","run","crouch","jump","h1","quite","free",
    "thing","played","thinking","ability","start","end","begin","steam","level","gameplay","change","everything","buy","sell","money","quest",
    "player", "need", "stop", "thing", "year", "month", "overall",
    "sometimes", "thing", "get", "experience", "feel", "time", "dev", "get", "got", "money", "almost", "job",
    "td tr", "tldr", "etc", "feature", "right", "little", "match", "main", "item", "getting", "account", "reason",
    "everyone", "everything", "person", "place", "work", "server", "full", "b", "ever", "nan", "current", "sure",
    "step", "content", "story", "enough", "review", "two", "one","td","team","devs","u","list","mod","second","seen","said",
    "added","mean","felt","make","looking","filtered","ac"
}
additional_stopwords.update({
    "player","need", "stop", "thing", "hour", "year", "month", "overall",
    "sometimes", "thing", "get", "experience", "feel", "time", "dev", "get", "got", "money", "almost", "job",
    "td tr", "tldr", "etc", "feature", "right", "little", "match", "main", "item", "getting", "account", "reason",
    "everyone", "everything", "person", "place", "work", "server", "full", "b", "ever", "nan", "current", "sure",
    "step", "content", "story", "enough", "review", "two", "one","td","team","devs","u","list","mod","second","seen","said",
    "added","mean","felt","make","looking","map","look","ubisoft","city","steampowered","http","dlc","grow","island","season pass",
    "store","tell","various","outside","coming","case","head","offline","true","guy","girl","sorry","thank","fromsoftware","bethesda",
    "option","opinion","return","15h","entire","part","create","fully","simulation","mission","drift","previous","before"
})

additional_stopwords.update({
    "furthermore", "thus", "nevertheless", "moreover", "nonetheless", "regardless", "consequently", "hence",
    "therefore", "otherwise", "likewise", "similarly", "surprisingly", "ultimately", "meanwhile", "additionally",
    "accordingly", "specifically", "subsequently", "notwithstanding", "altogether", "nevertheless", "moreover",
    "nevertheless", "conversely", "therefore", "furthermore", "otherwise", "additionally", "simultaneously",
    "similarly", "meanwhile", "respectively", "consequently", "accordingly", "likewise", "subsequently",
    "ultimately", "particularly", "notwithstanding", "moreover", "therefore", "consequently", "additionally",
    "similarly", "consequently", "nevertheless", "therefore", "additionally", "furthermore"
})

# Group by game and cluster, and apply word cloud generation
for game_name, game_group in df.groupby('game_name'):
    print(f"Generating word clouds for {game_name}...")
    unique_clusters = game_group['cluster'].unique()

    # Check if there are clusters for the current game
    if len(unique_clusters) == 0:
        print(f"No clusters found for the game: {game_name}")
    else:
        num_clusters = len(unique_clusters)
        cols = 2  # Number of columns for subplots
        rows = (num_clusters + cols - 1) // cols  # Calculate number of rows needed

        fig, axs = plt.subplots(rows, cols, figsize=(15, 6 * rows))

        # Generate word clouds for each cluster
        for i, cluster in enumerate(unique_clusters):
            # Filter dataframe for the current cluster
            cluster_df = game_group[game_group['cluster'] == cluster]

            # Concatenate all processed reviews for the current cluster
            cluster_reviews = ' '.join(cluster_df['processed_review'].astype(str))

            # Generate word cloud for the current cluster
            wordcloud = WordCloud(background_color='black', width=800, height=400, colormap="hot",
                                  stopwords=additional_stopwords).generate(cluster_reviews)

            # Plot word cloud
            ax = axs[i // cols, i % cols] if num_clusters > 1 else axs
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Cluster {cluster}')
            ax.axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

