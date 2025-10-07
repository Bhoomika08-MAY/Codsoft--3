import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download stopwords if not already present
nltk.download('stopwords')

# Load model and vectorizer
model_path = os.path.join('model', 'genre_nb_model.pkl')
vectorizer_path = os.path.join('model', 'tfidf_vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Load test data
with open(os.path.join('data', 'test_data.txt'), 'r', encoding='utf-8') as f:
    test_plots = [line.strip() for line in f if line.strip()]

# Preprocessing function
def preprocess(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Preprocess all plots
cleaned_plots = [preprocess(plot) for plot in test_plots]

# Transform and predict
X_test = vectorizer.transform(cleaned_plots)
predictions = model.predict(X_test)

# Print results
for plot, genre in zip(test_plots, predictions):
    print(f"\n🎬 Plot: {plot}\n📌 Predicted Genre: {genre}")