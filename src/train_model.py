import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Parse train_data.txt
train_records = []
with open("data/train_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ::: ")
        if len(parts) == 4:
            _, _, genre, description = parts
            train_records.append({"genre": genre, "plot": description})

df_train = pd.DataFrame(train_records)

# Preprocess text
def preprocess(text):
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

df_train['clean_plot'] = df_train['plot'].apply(preprocess)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_train['clean_plot'])
y = df_train['genre']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/genre_nb_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")