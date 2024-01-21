

import pandas as pd
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import random

# Function to convert a tokenized sentence into NLTK's format
def format_sentence(sent):
    return {word: True for word in word_tokenize(sent)}

def main():
    # Load the preprocessed training data
    # Make sure to replace the path with the correct one
    df = pd.read_csv('C:/Users/Tabitha/Desktop/Py_Projects/Steam_SentimentAnalysis/data/train_reviews.csv')

    # Assuming the DataFrame has a 'processed_review' column and a 'label' column
    # 'label' column should have values like 'pos' for positive and 'neg' for negative
    # Convert the DataFrame into a format suitable for training
    data = [(format_sentence(review), label) for review, label in zip(df['processed_review'], df['label'])]

    # Shuffle the data
    random.shuffle(data)

    # Train the Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(data)

    # Save the trained model - NLTK doesn't provide a direct way, but you can use pickle
    import pickle
    with open('../models/sentiment_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    print("Model training completed and saved.")

if __name__ == "__main__":
    main()
