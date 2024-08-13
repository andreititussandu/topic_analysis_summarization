import re
import spacy
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

additional_stopwords = [
    "-", "_", "'", "would", "could", "should", "also", "us", "said", "error",
    "please", "ad", "blocker", "site", "always", "however"
]

def preprocess_text(text):
    stop_words = set(stopwords.words('english')).union(set(additional_stopwords))

    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Process text with spaCy
    doc = nlp(text)

    # Filter out named entities and unwanted POS tags
    filtered_words = []
    for token in doc:
        if token.text.lower() not in stop_words and not token.ent_type_ and token.pos_ in {'NOUN', 'ADJ'}:
            filtered_words.append(token.lemma_)

    return ' '.join(filtered_words)
