import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

# Download NLTK resources if not already downloaded
# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords, stemmer, and lemmatizer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s,]', '', text)

    # Tokenize text
    tokens = text.split(',')

    # Define a list of prepositions POS tags
    preposition_tags = ["IN", "TO", "RB", "CC", "DT"]

    # Remove stopwords and prepositions based on POS tags
    filtered_tokens = []
    for token in tokens:
        # Tokenize each word in the token
        words = word_tokenize(token)
        # Perform part-of-speech tagging for each word
        tagged_words = pos_tag(words)
        # Remove stopwords and prepositions
        filtered_words = [word for word, pos in tagged_words if word.lower() not in stop_words and pos not in preposition_tags]
        # Join the remaining words into a token
        filtered_token = ' '.join(filtered_words)
        filtered_tokens.append(filtered_token)

    # Perform lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Perform stemming
    #stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]

    # Join tokens back into a string
    processed_text2 = ', '.join(lemmatized_tokens)

    #print(processed_text2)

    # Split the input string by commas
    parts = processed_text2.split(', ')

    # Process each part
    processed_parts = []
    for part in parts:
        # Split each part by space and remove 'ing' from words
        processed_words = [word[:-3] if word.endswith("ing") else word for word in part.split()]
        processed_words = [word[:-3] if word.endswith("ish") else word for word in processed_words]
        # Join the words back together
        processed_parts.append(' '.join(processed_words))

    # Join the processed parts back together with commas
    processed_text3 = ", ".join(processed_parts)

    #print(processed_text3)
    processed_text = output_text = ", ".join([word.replace(" ", "_") for word in processed_text3.split(", ")])



    return processed_text
