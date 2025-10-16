import random
import json
import nltk
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

with open("intents.json") as file:
    data = json.load(file)

corpus = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern)
        tags.append(intent["tag"])

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)  #break the sentence into words
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words]
    print(filtered)
    
    return " ".join(filtered)

corpus_clean = [clean_text(text) for text in corpus]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus_clean)

def get_response(user_input):
    cleaned_input = clean_text(user_input)
    vec_input = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(vec_input, X)
    index = np.argmax(similarity)
    if similarity[0][index] > 0.5:
        tag = tags[index]
        for intent in data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."
