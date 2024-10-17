import pickle
import re
import json
import unicodedata

from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from gensim.models.phrases import Phraser

with open("/Users/bimac/Documents/CTU/CT294/CT294-PAEBMR/config.json", 'r') as file:
    config = json.load(file)
# Đọc mô hình bigram từ file
bigram_model = Phraser.load(config['bigram_model'])

# Đọc mô hình trigram từ file
trigram_model = Phraser.load(config['trigram_model'])

# Đọc stopwords từ file
with open(config['eng_stopwords'], 'rb') as f:
    eng_stopwords = pickle.load(f)

# Khởi tạo WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def unicode_normalize(text):
    return unicodedata.normalize('NFC', text)

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_illegal_chars(text):
    ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010\013\014\016-\037]')
    return ILLEGAL_CHARACTERS_RE.sub('', text)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    return text.replace(',', ' ').replace('.', ' ')

def to_lower(text):
    return text.lower()

def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word.lower() not in eng_stopwords]
    return ' '.join(words)

def lemmatize_words(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def apply_bigrams_trigrams_to_text(text, bigram_model, trigram_model):
    words = text.split()
    text_with_bigrams = bigram_model[words]
    text_with_trigrams = trigram_model[text_with_bigrams]
    return ' '.join(text_with_trigrams)

def preprocess_review(review):
    review = unicode_normalize(review)
    review = remove_html_tags(review)
    review = remove_illegal_chars(review)
    review = clean_text(review)
    review = to_lower(review)
    review = remove_stopwords(review)
    review = lemmatize_words(review)
    review = apply_bigrams_trigrams_to_text(review, bigram_model, trigram_model)
    return review
