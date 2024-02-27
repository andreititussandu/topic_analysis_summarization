import pymongo
import pandas as pd
import re
import nltk
import string
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
bootstrap = Bootstrap(app)

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["website_data"]
collection = db["websites"]


def preprocess_text(text):
    """Preprocess text data by removing HTML tags, punctuation, stopwords, and converting to lowercase."""
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = cleaned_text.translate(translator)
    tokens = word_tokenize(cleaned_text.lower())
    stop_words = set(stopwords.words('english')).union({"-", "_", "'"})
    relevant_words = [word for word in tokens if word not in stop_words]
    return relevant_words


def get_text_content(url):
    """Retrieve text content from a given URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = " ".join(paragraph.get_text() for paragraph in paragraphs)
        return text_content
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None


def create_lda_model(dataframe):
    """Create an LDA model and its corresponding dictionary from text data."""
    documents = []
    for url in dataframe['website_url']:
        text_content = get_text_content(url)
        if text_content:
            relevant_words = preprocess_text(text_content)
            documents.append(relevant_words)

    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=len(dataframe['Category'].unique()), eval_every=1,
                         per_word_topics=True)
    return lda_model, dictionary


def insert_data_to_mongodb(dataframe, lda_model, dictionary):
    """Insert data into MongoDB collection."""
    for index, row in dataframe.iterrows():
        text_content = get_text_content(row['website_url'])
        if text_content:
            relevant_words = preprocess_text(text_content)
            bow = dictionary.doc2bow(relevant_words)
            topics = lda_model.get_document_topics(bow)
            topics.sort(key=lambda x: x[1], reverse=True)
            top_topic = topics[0]
            top_topic_terms = lda_model.print_topic(top_topic[0], len(dictionary))
            top_topic_strength = float(top_topic[1])
            url_category = row['Category']
            document = {
                "website_url": row['website_url'],
                "Category": url_category,
                "top_topic_terms": top_topic_terms,
                "top_topic_strength": top_topic_strength
            }
            collection.insert_one(document)


# @app.route('/', methods=['GET', 'POST'])  - this works
# def index():
#     """Main route to process webpage and return LDA topics."""
#     if request.method == 'POST':
#         url = request.form.get('url')
#         if url:
#             dataframe = pd.read_csv('website_classification_copy.csv')
#             lda_model, dictionary = create_lda_model(dataframe)
#             webpage_topic = process_webpage(url, dataframe, lda_model, dictionary)
#             return jsonify(webpage_topic)
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route to process webpage and return LDA topics."""
    if request.method == 'POST':
        url = request.form.get('url')
        if url:
            dataframe = pd.read_csv('website_classification_copy.csv')
            lda_model, dictionary = create_lda_model(dataframe)
            existing_data = collection.find_one({"website_url": url})
            if existing_data:
                webpage_topic = {
                    "top_topic_terms": existing_data["top_topic_terms"],
                    "top_topic_strength": existing_data["top_topic_strength"],
                    "category": existing_data["Category"]
                }
            else:
                webpage_topic = process_webpage(url, lda_model, dictionary)
            return jsonify(webpage_topic)
    return render_template('index.html')


def process_webpage(url, dataframe, lda_model, dictionary):
    """Process webpage and return LDA topics."""
    try:
        text_content = get_text_content(url)
        if text_content:
            relevant_words = preprocess_text(text_content)
            bow = dictionary.doc2bow(relevant_words)
            topics = lda_model.get_document_topics(bow)
            topics.sort(key=lambda x: x[1], reverse=True)
            top_topic = topics[0]
            top_topic_terms = lda_model.print_topic(top_topic[0], len(dictionary))
            top_topic_strength = float(top_topic[1])
            #url_category = dataframe[dataframe['website_url'] == url]['Category'].values[0]
            return {
                "top_topic_terms": top_topic_terms,
                "top_topic_strength": top_topic_strength,
                #"category": url_category
            }
    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    # Insert data into MongoDB collection
    dataframe = pd.read_csv('website_classification_copy.csv')
    lda_model, dictionary = create_lda_model(dataframe)
    insert_data_to_mongodb(dataframe, lda_model, dictionary)

    # Run the Flask application
    app.run(debug=False)

#  gunicorn --config gunicorn_config.py main:app -> to start gunicorn server