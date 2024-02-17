from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import re

import pandas as pd
import re
# nltk.corpus: Provides access to a variety of linguistic data, including stopwords.
import pandas as pd
from sklearn.model_selection import train_test_split
# re: Regular expression library for string searching and manipulation.

# os: Library for interacting with the operating system, not used in this revised script.
import os

# Slang and Abbreviations Dictionary
slang_abbrev_dict = {
    "gg": "good game",
    "op": "overpowered",
    "npc": "non-player character",
    "mmo": "massively multiplayer online",
    "rpg": "role-playing game",
    "fps": "first-person shooter",
    # Add more slang and abbreviations here as needed.
}

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
    "feel","edit","instead","really","playing","hour","baby","character","making","update","think","take",
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
    # Existing words...
    
    # Added gaming-specific sentiment words
    "immersive", "grinding", "laggy", "responsive", "unresponsive", "buggy", "glitchy",
    "optimized", "unoptimized", "engaging", "monotonous", "repetitive", "innovative",
    "clunky", "smooth", "balanced", "unbalanced", "fair", "unfair", "rewarding", "punishing",
    "accessible", "inaccessible", "deep", "shallow", "dynamic", "static", "rich", "poor",
    "varied", "generic", "original", "cliched", "predictable", "unpredictable", "challenging",
    "easy", "difficult", "hard", "simple", "complex", "strategic", "braindead", "mindless",
    "thoughtful", "meaningful", "pointless", "tedious", "fascinating", "captivating", "bland",
    "vibrant", "detailed", "sparse", "lush", "barren", "realistic", "unrealistic", "authentic",
    "believable", "immersion-breaking", "atmospheric", "non-atmospheric", "stylish", "drab",
    "charming", "lifeless", "soulful", "soulless", "innovative", "dated", "fresh", "stale",
    "intuitive", "convoluted", "user-friendly", "frustrating", "satisfactory", "unsatisfactory",
    "groundbreaking", "derivative", "polished", "unfinished", "complete", "incomplete",
    "overpriced", "underpriced", "valuable", "valueless", "overhyped", "underhyped",
    "overrated", "underrated", "hyped", "disregarded", "praised", "criticized", "celebrated",
    "condemned", "welcomed", "rejected", "anticipated", "ignored", "innovative", "traditional",
    "fast-paced", "slow-paced", "adventurous", "daring", "cautious", "epic", "mundane",
    "thrilling", "tedious", "enriching", "depleting", "rewarding", "frustrating",
    "motivating", "demotivating", "empowering", "disempowering", "skill-based", "luck-based",
    "tactical", "strategic", "casual", "hardcore", "niched", "mainstream", "accessible",
    "elitist", "inclusive", "exclusive", "welcoming", "alienating", "friendly", "hostile",
    "supportive", "toxic", "enjoyable", "miserable", "fun", "unfun", "entertaining", "dull",
    "inspiring", "depressing", "uplifting", "heartbreaking", "humorous", "serious",
    "light-hearted", "grim", "optimistic", "pessimistic", "realistic", "fantasy",
    "sci-fi", "historical", "modern", "futuristic", "retro", "timeless", "outdated",
    "trendy", "fashionable", "unfashionable", "cool", "uncool", "appealing", "unappealing",
    "addictive", "unengaging", "engrossing", "forgettable", "memorable", "unmemorable",
    "lasting", "ephemeral", "replayable", "one-time", "endless", "finite", "expansive",
    "limited", "open-world", "linear", "sandbox", "scripted", "player-driven", "story-driven",
    "narrative-rich", "plot-thin", "character-driven", "plot-driven", "action-packed",
    "slow-burn", "fast-moving", "steady", "evolving", "static", "revolutionary", "traditional"
}

# Function to handle negations (e.g., "not good" becomes "not_good")
def handle_negations(input_text):
    negation_words = ['not', 'no', 'never', 'neither', 'nor', 'cannot']
    negated_tokens = []
    tokens = input_text.split()
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in negation_words and i < len(tokens) - 1:
            negated_tokens.append(token + '_' + tokens[i + 1])
            skip_next = True
        else:
            negated_tokens.append(token)
    return ' '.join(negated_tokens)

# Function to replace slang and abbreviations with their expanded form
def expand_slang_abbreviations(text, slang_dict):
    regex = re.compile(r'\b(' + '|'.join(map(re.escape, slang_dict.keys())) + r')\b')
    return regex.sub(lambda match: slang_dict[match.group(0)], text)

# Function to detect and emphasize emphasized words
def emphasize_detection(text):
    emphasized_text = re.sub(r'(\b\w+\b)(\s+\1)+', r'\1_emphasized', text)
    return emphasized_text

# Combine the default NLTK stopwords with your additional stopwords
all_stopwords = set(stopwords.words('english')).union(additional_stopwords)
sentiment_words = set(sentiment_words)  # Ensure this is a set for efficient look-up

def preprocess_text(text):
    # Decode text to UTF-8 to handle special characters
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = expand_slang_abbreviations(text, slang_abbrev_dict)  # Expand slang and abbreviations
    text = re.sub(r'\W+', ' ', text)  # Replace non-word characters with spaces
    text = handle_negations(text)  # Handle negations
    text = emphasize_detection(text)  # Detect and emphasize emphasized words
    tokens = word_tokenize(text)
    
    # Filter out stopwords and keep sentiment words
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in all_stopwords or word in sentiment_words
    ]
    
    return ' '.join(filtered_tokens)


def main():
    # Load the steam reviews dataset with UTF-8 encoding to handle special characters like "é" in "Pokémon"
    df = pd.read_csv('data/steam_reviews.csv', encoding='utf-8')
    
    # Apply the preprocessing function to the review text
    df['processed_review'] = df['review_text'].apply(lambda x: preprocess_text(x))
    
    # Save the entire processed dataset with UTF-8 encoding
    df.to_csv('data/processed_reviews.csv', index=False, encoding='utf-8')
    
    # Split the processed dataset into training and testing sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the training and testing datasets with UTF-8 encoding
    train.to_csv('data/train_reviews.csv', index=False, encoding='utf-8')
    test.to_csv('data/test_reviews.csv', index=False, encoding='utf-8')
    
    print("Data preprocessing and splitting completed. Check the specified directories for the processed files.")


if __name__ == "__main__":
    main()
