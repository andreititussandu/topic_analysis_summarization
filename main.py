import os
import pickle
import time
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

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["website_data"]
collection = db["websites"]

def preprocess_text(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = cleaned_text.translate(translator)
    tokens = word_tokenize(cleaned_text.lower())
    stop_words = set(stopwords.words('english')).union({"-", "_", "'","would","could","should","also", "us",
                *map(str, range(10))})
    relevant_words = [word for word in tokens if word not in stop_words and all(ord(char) < 128 for char in word)]
    return relevant_words

def get_text_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = " ".join(paragraph.get_text() for paragraph in paragraphs)
        return text_content
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None

def train_lda_model(dataframe):
    lda_models = {}
    dictionaries = {}
    for link in dataframe['link']:
        text_content = get_text_content(link)
        if text_content:
            relevant_words = preprocess_text(text_content)
            dictionary = Dictionary([relevant_words])
            corpus = [dictionary.doc2bow(relevant_words)]
            lda_model = LdaModel(corpus, id2word=dictionary, num_topics=1, chunksize=100,
                                 passes=50, eval_every=1, per_word_topics=True)
            lda_models[link] = lda_model
            dictionaries[link] = dictionary
    return lda_models, dictionaries

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def retrain_lda_model(dataframe, filename):
    if os.path.exists(filename):
        lda_models, dictionaries = load_model(filename)
    else:
        lda_models, dictionaries = train_lda_model(dataframe)
        save_model((lda_models, dictionaries), filename)
    new_links = [link for link in dataframe['link'] if link not in lda_models]
    if new_links:
        new_lda_models, new_dictionaries = train_lda_model(dataframe[dataframe['link'].isin(new_links)])
        lda_models.update(new_lda_models)
        dictionaries.update(new_dictionaries)
        save_model((lda_models, dictionaries), filename)

def insert_data_to_mongodb(dataframe, lda_models, dictionaries):
    for index, row in dataframe.iterrows():
        text_content = get_text_content(row['link'])
        if text_content:
            relevant_words = preprocess_text(text_content)
            dictionary = dictionaries.get(row['link'])
            lda_model = lda_models.get(row['link'])
            if dictionary and lda_model:
                bow = dictionary.doc2bow(relevant_words)
                topics = lda_model.get_document_topics(bow)
                topics.sort(key=lambda x: x[1], reverse=True)
                top_topic = topics[0]
                top_topic_terms = lda_model.print_topic(top_topic[0], len(dictionary))
                top_topic_strength = float(top_topic[1])
                document = {
                    "topic": row['topic'],
                    "link": row['link'],
                    "top_topic_terms": top_topic_terms,
                    "top_topic_strength": top_topic_strength
                }
                collection.insert_one(document)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url')
        if url:
            existing_data = collection.find_one({"link": url})
            if existing_data:
                existing_data['_id'] = str(existing_data['_id'])
                return jsonify(existing_data)
            else:
                return jsonify({"error": "URL not found in database"})
    return render_template('index.html')

if __name__ == '__main__':
    dataframe = pd.read_csv('test2.csv', delimiter=',')
    retrain_lda_model(dataframe, 'lda_model.pkl')
    lda_models, dictionaries = load_model('lda_model.pkl')
    insert_data_to_mongodb(dataframe, lda_models, dictionaries)
    app.run(debug=False)



#  gunicorn --config gunicorn_config.py main:app -> to start gunicorn server