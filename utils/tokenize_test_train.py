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

# Slang and Abbreviations Dictionary (specific for game reviews)
slang_abbrev_dict = {
    "pve": "player versus environment",
    "pvp": "player versus player",
    "mmo": "massively multiplayer online",
    "mmorpg": "massively multiplayer online role-playing game",
    "fps": "first-person shooter",
    "rts": "real-time strategy",
    "rpg": "role-playing game",
    "mob": "mobile object",
    "mob": "monster or enemy",
    "aggro": "aggression or aggressiveness",
    "agg": "aggressive",
    "dps": "damage per second",
    "aoe": "area of effect",
    "buff": "a beneficial status effect",
    "nerf": "a detrimental status effect",
    "proc": "process or activation of a special ability",
    "raid": "a large group of players tackling a difficult challenge",
    "loot": "items or rewards dropped by defeated enemies",
    "respawn": "the reappearance of a player or enemy after being defeated",
    "spawn": "the point where players or enemies enter the game world",
    "tp": "teleport or teleportation",
    "gg": "good game",
    "wp": "well played",
    "glhf": "good luck, have fun",
    "ez": "easy",
    "afk": "away from keyboard",
    "b2p": "buy to play",
    "f2p": "free to play",
    "p2p": "pay to play",
    "gank": "ambush or surprise attack",
    "farming": "repeatedly defeating enemies or completing tasks to gain resources or experience",
    "grinding": "repeatedly performing the same action to gain experience or resources",
    "twink": "a low-level character that has been heavily equipped or supported by a higher-level player",
    "whale": "a player who spends a large amount of money on microtransactions",
    "camping": "waiting in a strategic location to ambush or defeat other players or enemies",
    "cheese": "exploiting a game mechanic or strategy to gain an unfair advantage",
    "noob": "a inexperienced or unskilled player",
    "pro": "a highly skilled or experienced player",
    "griefing": "intentionally disrupting or annoying other players",
    "nerf": "to weaken or reduce the effectiveness of a character, item, or ability",
    "buff": "to strengthen or improve the effectiveness of a character, item, or ability",
    "pwn": "to defeat or dominate an opponent",
    "rekt": "to be decisively defeated or destroyed",
    "tryhard": "a player who tries excessively hard to win or succeed",
    "wombo combo": "a combination of abilities or attacks that results in a devastating outcome",
    "zerg": "a strategy involving overwhelming an opponent with sheer numbers",
    "spawn camping": "repeatedly attacking players as they respawn in a multiplayer game",
    "gg ez": "a sarcastic or taunting message implying that the game was easy",
    "toxic": "behaving in a disruptive, offensive, or unsportsmanlike manner",
    "trolling": "deliberately provoking or antagonizing other players for amusement",
    "gank": "to ambush or surprise attack an opponent in a multiplayer game",
    "lag": "a delay or interruption in the gameplay due to slow or inconsistent network connection",
    "hacker": "a player who uses unauthorized or illegal methods to gain an advantage in a game",
    "exploit": "a software or gameplay loophole that allows players to gain an unfair advantage",
    "meta": "the most effective or popular strategies, characters, or builds in a game",
    "easter egg": "a hidden feature, message, or reference in a game",
    "speedrun": "completing a game as quickly as possible",
    "glitch": "a software or programming error that causes unexpected behavior in a game",
    "mod": "a modification or alteration made to a game by players or fans",
    "beta": "a pre-release version of a game that is made available for testing",
    "alpha": "an early development version of a game that is not yet feature-complete",
    "early access": "a system where players can purchase and play a game before its official release",
    "dlc": "downloadable content, additional game content released after the initial launch",
    "expansion": "a major addition to a game that adds new features, content, or gameplay mechanics",
    "patch": "a software update that fixes bugs, balances gameplay, or adds new features to a game",
    "server": "a computer or system that hosts multiplayer game sessions and facilitates communication between players",
    "multiplayer": "a mode of gameplay where multiple players can interact and compete with each other in real-time",
    "singleplayer": "a mode of gameplay where a player can experience the game alone, without interaction from other players",
    "co-op": "a mode of gameplay where multiple players can collaborate and work together towards a common goal",
    "campaign": "a series of connected missions or levels that form the main storyline of a game",
    "quest": "a task or objective given to a player by a non-player character in a game",
    "respawn": "the act of a player or enemy reappearing in the game world after being defeated",
    "npc": "non-player character, a character controlled by the game's artificial"
}

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
    "added","mean","felt","make","looking","player","need", "stop", "thing", "hour", "year", "month", "overall",
    "sometimes", "thing", "get", "experience", "feel", "time", "dev", "get", "got", "money", "almost", "job",
    "td tr", "tldr", "etc", "feature", "right", "little", "match", "main", "item", "getting", "account", "reason",
    "everyone", "everything", "person", "place", "work", "server", "full", "b", "ever", "nan", "current", "sure",
    "step", "content", "story", "enough", "review", "two", "one","td","team","devs","u","list","mod","second","seen","said",
    "added","mean","felt","make","looking","map","look","ubisoft","city","steampowered","http","dlc","grow","island","season pass",
    "store","tell","various","outside","coming","case","head","offline","true","guy","girl","sorry","thank","fromsoftware","bethesda",
    "option","opinion","return","15h","entire","part","create","fully","simulation","mission","drift","previous","before",
    "furthermore", "thus", "thing", "nevertheless", "moreover", "nonetheless", "regardless", "consequently", "hence",
    "therefore", "otherwise", "likewise", "similarly", "surprisingly", "ultimately", "meanwhile", "additionally",
    "accordingly", "specifically", "subsequently", "notwithstanding", "altogether", "nevertheless", "moreover",
    "nevertheless", "conversely", "therefore", "furthermore", "otherwise", "additionally", "simultaneously",
    "similarly", "meanwhile", "respectively", "consequently", "accordingly", "likewise", "subsequently",
    "ultimately", "particularly", "notwithstanding", "moreover", "therefore", "consequently", "additionally",
    "similarly", "consequently", "nevertheless", "therefore", "additionally", "furthermore",
    "you", "in", "this", "that", "but", "for", "with", "on", "or", "are", "have", "can", "my", "if", "like", "they"}



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

## Function to handle negations (e.g., "not good" becomes "not_good")
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

def emphasize_detection(text):
    """
    Detect and emphasize repeated full words while ignoring single characters or numbers.
    Regex Explanation:
    \b: Word boundary.
    \w{2,}: Matches words with 2 or more characters.
    (\s+\1)+: Detects repetitions of the matched word with spaces in between.
    Only matches valid words with sufficient length to be meaningful.

    """
    emphasized_text = re.sub(r'\b(\w{2,})(\s+\1)+\b', r'\1_emphasized', text)  # Match words with 2+ characters
    return emphasized_text


def preprocess_text(text, game_names):
    # Handle non-string inputs
    if not isinstance(text, str):
        text = str(text)

    # Decode text to UTF-8 to handle special characters
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = expand_slang_abbreviations(text, slang_abbrev_dict)  # Expand slang and abbreviations
    text = re.sub(r'\W+', ' ', text)  # Replace non-word characters with spaces
    text = handle_negations(text)  # Handle negations
    text = emphasize_detection(text)  # Detect and emphasize emphasized words
    
    tokens = word_tokenize(text)
    
    # Combine the default NLTK stopwords with additional stopwords
    all_stopwords = set(stopwords.words('english')).union(additional_stopwords)
    
    # Add game names to the stopwords set
    all_stopwords.update(game_names.lower().split())
    
    # Filter out stopwords, single characters, and numeric tokens
    filtered_tokens = [
    lemmatizer.lemmatize(word) for word in tokens
    if word not in all_stopwords  # Exclude stopwords
    and len(word) > 1  # Exclude single characters
    and not word.isnumeric()  # Exclude numbers
    or word in sentiment_words  # Keep only valid sentiment words
    ]   
    
    return ' '.join(filtered_tokens)



def main():
    # Load the steam reviews dataset with UTF-8 encoding to handle special characters like "é" in "Pokémon"
    df = pd.read_csv('data/steam_reviews.csv', encoding='utf-8')

    # Ensure text columns are strings
    df['review_text'] = df['review_text'].fillna('').astype(str)
    df['game_name'] = df['game_name'].fillna('').astype(str)
    
    # Apply the preprocessing function to the review text
    df['processed_review'] = df.apply(lambda row: preprocess_text(row['review_text'], row['game_name']), axis=1)
    
    # Save the entire processed dataset with UTF-8 encoding
    df.to_csv('data/processed_reviews.csv', index=False, encoding='utf-8')

    print(df['processed_review'].head()) # Test that the data is validated when changed post process.
    
    # Split the processed dataset into training and testing sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the training and testing datasets with UTF-8 encoding
    train.to_csv('data/train_reviews.csv', index=False, encoding='utf-8')
    test.to_csv('data/test_reviews.csv', index=False, encoding='utf-8')
    
    print("Data preprocessing and splitting completed. Check the specified directories for the processed files.")


if __name__ == "__main__":
    main()

"""
Quick test:

TEST 1 - REMOVE SINGLE DIGITS AND LETTERS
TEST 2 - REMOVE STOPWORDS
"""
sample_text = "This game is great great but 1 1 1 is not cool a a a at all."
game_names = "example_game"
processed_text = preprocess_text(sample_text, game_names)
print("TEST 1",processed_text)
# RESULT -> this game is great_emphasized but is not_cool at all (This is perfect!)

sample_text = "You play this game for fun, but it is not that great in my opinion."
game_names = "example_game"
processed_text = preprocess_text(sample_text, game_names)
print("TEST 2",processed_text)
