from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np
import requests
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def preprocess_text(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = cleaned_text.translate(translator)
    tokens = word_tokenize(cleaned_text.lower())
    relevant_words = [word for word in tokens if word not in stop_words]
    return relevant_words

def create_lda_model(dataframe):
    your_documents = []
    for url in dataframe['website_url']:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text_content = ""
            for paragraph in paragraphs:
                text_content += paragraph.get_text() + " "
            relevant_words = preprocess_text(text_content)
            your_documents.append(relevant_words)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")

    dictionary = Dictionary(your_documents)
    corpus = [dictionary.doc2bow(doc) for doc in your_documents]
    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=len(dataframe['Category'].unique()))
    return lda_model, dictionary

stop_words = set(stopwords.words('english')).union({"-", "_", "'"})
custom_stop_words = set(["-", "'", "_"])

dataset = pd.read_csv('website_classification_copy.csv')
df = dataset[['website_url', 'cleaned_website_text', 'Category']].copy()

lda_model, dictionary = create_lda_model(df)

def get_topics(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ""
        for paragraph in paragraphs:
            text_content += paragraph.get_text() + " "
        relevant_words = preprocess_text(text_content)
        bow = dictionary.doc2bow(relevant_words)
        topics = lda_model.get_document_topics(bow)
        topics.sort(key=lambda x: x[1], reverse=True)
        top_topic = topics[0]
        top_topic_terms = lda_model.print_topic(top_topic[0], len(dictionary))
        top_topic_strength = float(top_topic[1])
        url_category = df[df['website_url'] == url]['Category'].values[0]
        return {
            "top_topic_terms": top_topic_terms,
            "top_topic_strength": top_topic_strength,
            "category": url_category
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/', methods=['GET', 'POST'])
def get_urls():
    try:
        if request.method == 'POST':
            print('it is a POST method')
            url = request.form.get('url')
            print(f"Extracted URL: {url}")
            if url:
                webpage_topic = get_topics(url)
                return jsonify(webpage_topic)
            else:
                return render_template('index.html')
        else:
            return render_template('index.html')
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=False)