import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from .preprocess import preprocess_text

def scrape_text_from_url(url, timeout = 60):
    try:
        response = requests.get(url, timeout=timeout)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return preprocess_text(text)
    except Timeout:
        print(f"Timeout occurred while trying to scrape {url}")
        return ''