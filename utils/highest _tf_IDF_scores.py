import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Read and clean data:
#  preprocessed text column in DataFrame
df = pd.read_csv('data/processed_reviews.csv')
df['processed_review'].fillna('', inplace=True) # Replace NaN values with an empty string


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3), min_df=5, max_df=0.7)

'''
max_features=500: Limits the vocabulary size to the top 500 terms with the highest TF-IDF scores.
ngram_range=(1, 3): Considers unigrams (single words), bigrams (two-word phrases), and trigrams (three-word phrases).
min_df=5: Ignores terms that appear in fewer than 5 reviews (documents).
max_df=0.7: Excludes terms that appear in more than 70% of the reviews, as these are likely stopwords or overly common terms (e.g., "the", "and").
'''

# Fit and transform the processed reviews
X = vectorizer.fit_transform(df['processed_review'])

# Get feature names (Extracts the vocabulary (terms) learned by the vectorizer.)
feature_names = vectorizer.get_feature_names_out()  # Updated method for getting feature names

# Sum tf-idf scores for each term across all documents
summed_tfidf = X.sum(axis=0)
# This gives an overall score for how "important" each term is across the dataset.


# Map feature names to their summed tf-idf scores - Keys: Terms from feature_names (e.g., "great graphics", "poor design").
tfidf_scores = {feature_names[col]: summed_tfidf[0, col] for col in range(X.shape[1])}

# Sort terms by their scores  - this ensures the most "important" terms (i.e., those with the highest scores) appear first.
sorted_terms = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)

# Review the top terms - The output will list the most relevant terms across all reviews. 
print("Top terms by TF-IDF score:")
for term, score in sorted_terms[:20]:  # Change 20 to review more or fewer terms
    print(f"{term}: {score}")

