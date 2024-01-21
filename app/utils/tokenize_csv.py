# pre-processing the data to prepare it for feeding to the model, train and test sets.
# we will use the nltk library for this portion. including lemmatization.
# Lemmatization reduces words to their base form, but unlike stemming, it considers the context and converts the word to its meaningful base form. 

# nltk: Natural Language Toolkit, a suite of libraries for natural language processing (NLP).
import nltk
# nltk.corpus: Provides access to a variety of linguistic data, including stopwords.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd
from sklearn.model_selection import train_test_split
# re: Regular expression library for string searching and manipulation.
import re

# os: Library for interacting with the operating system, not used in this revised script.
import os

# Download necessary NLTK resources if not already present.
nltk.download('punkt')  # Tokenizers for splitting text into tokens (words).
nltk.download('stopwords')  # Common words (like 'the', 'is', 'in', etc.) that are usually removed in NLP tasks.

from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(filtered_tokens)


def main():
    # Read the CSV file from the 'data' directory.
    df = pd.read_csv('C:/Users/Tabitha/Desktop/Py_Projects/Steam_SentimentAnalysis/data/steam_reviews.csv')

    # Preprocess and tokenize the 'review_text' column of the DataFrame.
    df['processed_review'] = df['review_text'].apply(preprocess_text)

    # Split the dataset into train (80%) and test (20%) sets.
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Save the processed training data to a CSV file.
    train.to_csv('train_reviews.csv', index=False)

    # Save the processed testing data to a CSV file.
    test.to_csv('test_reviews.csv', index=False)

    # Print a preview of the processed data.
    print("Data preprocessing and splitting completed.")
    print("Preview of processed data:")
    print(df.head())

if __name__ == "__main__":
    main()
