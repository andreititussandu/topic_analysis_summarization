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
import tensorflow as tf

print(tf.__version__)
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

urls = [
    "https://www.mayoclinic.org/drugs-supplements-creatine/art-20347591",
    "https://www.infoworld.com/article/3204016/what-is-python-powerful-intuitive-programming.html",
    "https://www.football365.com/news/summer-transfer-window-2023-most-expensive-players-biggest-deals"
]

your_documents = []

stop_words = set(stopwords.words('english')).union({"-", "\'"})
custom_stop_words = set(["-", "\'"])


# Function to clean and preprocess text
def preprocess_text(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = cleaned_text.translate(translator)
    tokens = word_tokenize(cleaned_text.lower())
    relevant_words = [word for word in tokens if word not in stop_words]
    return relevant_words


# Collect web page contents and preprocess
def scraping(urls):
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')  # Split the text into paragraphs
            text_content = ""
            for paragraph in paragraphs:
                text_content += paragraph.get_text() + " "

            # Preprocess the text
            relevant_words = preprocess_text(text_content)

            # Append the relevant words to your_documents
            your_documents.append(relevant_words)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")


scraping(urls)

# Create a Dictionary object from your documents
dictionary = Dictionary(your_documents)

# Convert the documents to bag-of-words format
corpus = [dictionary.doc2bow(doc) for doc in your_documents]

# Train the LDA model
lda_model = LdaModel(corpus, id2word=dictionary, num_topics=5)


# Function to get topics and summary for a URL
def get_topics_and_summary(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')  # Split the text into paragraphs
        text_content = ""
        for paragraph in paragraphs:
            text_content += paragraph.get_text() + " "

        # Preprocess the text
        relevant_words = preprocess_text(text_content)

        # Create a bag-of-words representation
        bow = dictionary.doc2bow(relevant_words)

        # Get topics from the LDA model
        topics = lda_model.get_document_topics(bow)

        # Sort topics by strength
        topics.sort(key=lambda x: x[1], reverse=True)

        # Extract the top topic and its strength
        top_topic = topics[0]

        # Convert top_topic strength to a serializable data type (e.g., float64)
        top_topic_strength = float(top_topic[1])

        # Get the terms associated with the top topic
        # top_topic_terms = lda_model.print_topic(top_topic)

        # Generate a summary
        summary = ' '.join(relevant_words[:100])

        return {
            "top_topic": top_topic[0],
            "top_topic_strength": top_topic_strength,
            "summary": summary
        }
    except Exception as e:
        return {"error": str(e)}


@app.route('/analyze', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json()
        url = data.get('url')
        result = get_topics_and_summary(url)
        # return render_template('result.html', top_topic=result["top_topic"],
        # top_topic_strength=result["top_topic_strength"], summary=result["summary"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=False)
