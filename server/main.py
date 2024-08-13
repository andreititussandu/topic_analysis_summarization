from flask import request, jsonify
from flask_cors import cross_origin
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
import pickle
from server.web_server import app, db, collection
from scripts.web_scraper import scrape_text_from_url
#from scripts.summarizer import summarize_text

# Function to read CSV file
def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print("CSV read successfully")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

# Function to validate required columns
def validate_columns(df, required_columns):
    for column in required_columns:
        if column not in df.columns:
            print(f"Error: Required column '{column}' not found in CSV")
            return False
    return True

# Function to scrape text from URLs
def scrape_documents(links):
    documents = []
    for link in links:
        try:
            print(f"Scraping text from URL: {link}")
            text = scrape_text_from_url(link)
            documents.append(text)
        except Exception as e:
            print(f"Failed to scrape {link}: {e}")
            documents.append('')
    return documents

# Function to vectorize documents
def vectorize_documents(documents):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2))
    dtm = vectorizer.fit_transform(documents)
    return vectorizer, dtm

# Function to apply LDA
def apply_lda(dtm, n_components=7):
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(dtm)
    return lda

# Function to store data in MongoDB
def store_in_mongodb(links, topics, documents, lda, vectorizer):
    for idx, link in enumerate(links):
        collection.insert_one({
            'url': link,
            'topic': topics[idx],
            'lda': lda.transform(vectorizer.transform([documents[idx]]))[0].tolist()
        })

# Function to train the Naive Bayes model
def train_predictive_model(dtm, topics):
    nb = MultinomialNB()
    nb.fit(dtm, topics)
    return nb

# Function to save models to disk
def save_model_and_vectorizer(model, vectorizer):
    with open('../models/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('../models/vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

# Main function to process the CSV file
def process_csv(file_path):
    df = read_csv(file_path)
    if df is None or not validate_columns(df, ['topic', 'link']):
        return

    topics = df['topic'].tolist()
    links = df['link'].tolist()

    documents = scrape_documents(links)
    vectorizer, dtm = vectorize_documents(documents)
    lda = apply_lda(dtm)

    store_in_mongodb(links, topics, documents, lda, vectorizer)
    nb_model = train_predictive_model(dtm, topics)
    save_model_and_vectorizer(nb_model, vectorizer)
    print("CSV processed and data stored in MongoDB")

@app.route('/upload_csv', methods=['POST'])
@cross_origin()
def upload_csv():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        file_path = 'uploaded_file.csv'
        file.save(file_path)
        process_csv(file_path)
        return "CSV processed and data stored in MongoDB", 200
    else:
        return "No file uploaded", 400

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    url = request.json.get('url')
    if not url:
        return "No URL provided", 400

    try:
        text = scrape_text_from_url(url)
    except Exception as e:
        return f"Failed to scrape {url}: {e}", 500

    with open('./models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('./models/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)

    return jsonify({'predicted_topic': prediction[0]}), 200

#@app.route('/summarize', methods=['POST'])
#@cross_origin()
#def summarize():
#    url = request.json.get('url')
#    if not url:
#        return "No URL provided", 400

#    try:
#        text = scrape_text_from_url(url)
#        summary = summarize_text(text)
#    except Exception as e:
#        return f"Failed to scrape or summarize {url}: {e}", 500

#    return jsonify({'summary': summary}), 200

if __name__ == '__main__':
    app.run(debug=False)

# gunicorn -c gunicorn_config.py server.main:app