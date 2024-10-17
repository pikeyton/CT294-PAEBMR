import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB


with open("/Users/bimac/Documents/CTU/CT294/CT294-PAEBMR/config.json", 'r') as file:
    config = json.load(file)

# Tải các model đã lưu
tfidf = joblib.load(config['tfidf'])
chi2_selector = joblib.load(config['chi2_selector'])
mnb = joblib.load(config['mnb'])

map_label = {
    0: 'negative',
    1: 'positive'
}

def feature_extraction(text: str):
    text = [text]
    text = tfidf.transform(text)
    text = chi2_selector.transform(text)
    return text.toarray()

def classify_review(text: str):
    features = feature_extraction(text)
    prediction = mnb.predict(features)
    return map_label[prediction[0]]


