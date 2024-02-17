# pre-processing the data to prepare it for feeding to the model, train and test sets.
# we will use the nltk library for this portion. including lemmatization.
# Lemmatization reduces words to their base form, but unlike stemming, it considers the context and converts the word to its meaningful base form. 

# nltk: Natural Language Toolkit, a suite of libraries for natural language processing (NLP).
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import re
# nltk.corpus: Provides access to a variety of linguistic data, including stopwords.
import pandas as pd
from sklearn.model_selection import train_test_split
# re: Regular expression library for string searching and manipulation.

# os: Library for interacting with the operating system, not used in this revised script.
import os

# Download necessary NLTK resources if not already present.
#nltk.download('punkt')  # Tokenizers for splitting text into tokens (words).
#nltk.download('stopwords')  # Common words (like 'the', 'is', 'in', etc.) that are usually removed in NLP tasks.\
#nltk.download('wordnet')  # WordNet Lemmatizer resource.



# Expanded list of additional stopwords
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
    "people", "place", "player", "play","play er","player", "put", "say", "see", "seem", "seemed", "seems", "several", "she",
    "should", "show", "since", "some", "something", "sometime", "somewhere", "still", "such", "take", "than",
    "that", "the", "their", "them", "then", "there", "therefore", "these", "they", "thing", "this", "those",
    "though", "through", "thus", "time", "to", "together", "too", "toward", "try", "under", "until", "up",
    "upon", "us", "use", "used", "using", "very", "want", "was", "way", "we", "well", "went", "were", "what",
    "when", "where", "whether", "which", "while", "who", "whole", "whose", "why", "will", "with", "within",
    "without", "would", "yet", "you", "your", "yours", "yourself","hour","player","buy","review","game","system",
    "feel","edit","instead","really","playing","hour","baby","character","npc","mmo","rpg","making","update","think","take",
    "absolutely","point","feel","etc","hit","fact","date","month","item","area","better","run","crouch","jump","h1","quite","free",
    "thing","played","thinking","ability","start","end","begin","steam","level","gameplay","change","everything","buy","sell","money","quest",
    "player", "need", "stop", "thing", "setting", "game", "playing", "hour", "year", "month", "overall",
    "sometimes", "thing", "get", "experience", "feel", "time", "dev", "get", "got", "money", "almost", "job",
    "td tr", "tldr", "etc", "feature", "right", "little", "match", "main", "item", "getting", "account", "reason",
    "everyone", "everything", "person", "place", "work", "server", "full", "b", "ever", "nan", "current", "sure",
    "step", "content", "story", "enough", "review", "two", "one","td","team","devs","u","list","mod","second","seen","said",
    "added","mean","felt","make","looking"
}

# Expanded list of sentiment words to prioritize in analysis (game related so we can get a good feel of the consensus)
sentiment_words = {
    "bad", "good", "sad", "happy", "funny", "laughing", "pissed", "angry", "pretty", 
    "expensive", "cheap", "ugly", "beautiful", "enjoyable", "boring", "exciting", 
    "amazing", "terrible", "awful", "disappointing", "satisfying", "thrilling", 
    "poor", "great", "disgusting", "impressive", "fantastic", "horrific", "creepy", 
    "scary", "intense", "relaxing", "stressful", "rewarding", "dull", "fantastic", 
    "perfect", "flawed", "favorite", "hate", "love", "worst", "best", "fail", "win", 
    "loser", "winner", "bored", "excited", "fear", "joy", "tears", "smiles", "unhappy", 
    "content", "joyful", "gloomy", "bright", "dark", "vivid", "blurry", "clear", "fuzzy",
    "sharp", "colorful", "drab", "shiny", "gloomy", "luminous", "radiant", "dismal", 
    "gleaming", "dazzling", "sleek", "nifty", "elegant", "graceful", "clumsy", "awkward"
    # Add additional sentiment words here
}
# Combine the default NLTK stopwords with your additional stopwords
all_stopwords = set(stopwords.words('english')).union(additional_stopwords)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    # Keep words if they are not in all_stopwords or if they are in sentiment_words
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in all_stopwords or word in sentiment_words]
    return ' '.join(filtered_tokens)

def main():
    # Load the steam reviews dataset
    df = pd.read_csv('C:/Users/Tabitha/Desktop/Py_Projects/Steam_SentimentAnalysis/data/steam_reviews.csv')
    
    # Apply the preprocessing function to the review text
    df['processed_review'] = df['review_text'].apply(lambda x: preprocess_text(x))
    
    # Save the entire processed dataset before splitting
    df.to_csv('C:/Users/Tabitha/Desktop/Py_Projects/Steam_SentimentAnalysis/data/processed_reviews.csv', index=False)
    
    # Split the processed dataset into training and testing sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the training and testing datasets
    train.to_csv('C:/Users/Tabitha/Desktop/Py_Projects/Steam_SentimentAnalysis/data/train_reviews.csv', index=False)
    test.to_csv('C:/Users/Tabitha/Desktop/Py_Projects/Steam_SentimentAnalysis/data/test_reviews.csv', index=False)
    
    print("Data preprocessing and splitting completed. Check the specified directories for the processed files.")

if __name__ == "__main__":
    main()
