import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming 'processed_review' is your preprocessed text column in the DataFrame
df = pd.read_csv('data/processed_reviews.csv')

# Replace NaN values with an empty string
df['processed_review'].fillna('', inplace=True)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3), min_df=5, max_df=0.7)

# Fit and transform the processed reviews
X = vectorizer.fit_transform(df['processed_review'])

# Get feature names
feature_names = vectorizer.get_feature_names_out()  # Updated method for getting feature names

# Sum tf-idf scores for each term across all documents
summed_tfidf = X.sum(axis=0)

# Map feature names to their summed tf-idf scores
tfidf_scores = {feature_names[col]: summed_tfidf[0, col] for col in range(X.shape[1])}

# Sort terms by their scores
sorted_terms = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)

# Review the top terms
print("Top terms by TF-IDF score:")
for term, score in sorted_terms[:20]:  # Change 20 to review more or fewer terms
    print(f"{term}: {score}")

