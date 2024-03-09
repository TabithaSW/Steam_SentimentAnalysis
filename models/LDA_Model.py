import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Load data
df = pd.read_csv('data/processed_reviews.csv')

# Convert tokenized reviews to list of lists
processed_reviews = []

# Check each element in the 'game_name' column
for idx, game_name in enumerate(df['game_name']):
    # Check if the game_name is "HELLDIVERS™ 2"
    if game_name == "HELLDIVERS™ 2":
        # Process the review
        review = df['processed_review'][idx]
        if isinstance(review, str):
            processed_reviews.append(review.split())

# Create dictionary
dictionary = Dictionary(processed_reviews)

# Filter out extremes
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Create bag-of-words corpus
bow_corpus = [dictionary.doc2bow(text) for text in processed_reviews]

# Train the LDA model
num_topics = 5
ldamodel = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=20)

# Get the topics and words
topics = ldamodel.show_topics(num_topics=num_topics, num_words=10, log=False, formatted=False)

# Prepare data for saving
data = []
for idx, game_name in enumerate(df['game_name']):
    if game_name == "HELLDIVERS™ 2" and idx < len(bow_corpus):
        topic_distribution = ldamodel.get_document_topics(bow_corpus[idx])
        topic_idx = max(topic_distribution, key=lambda item: item[1])[0]
        words = [word for word, _ in topics[topic_idx][1]]
        data.append({'Game Name': game_name, 'Processed Review': df['processed_review'][idx], 'Words Produced': words})

# Create DataFrame
result_df = pd.DataFrame(data)

# Save to CSV file
result_df.to_csv('data/lda_results.csv', index=False)

