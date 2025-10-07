import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import os

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load(os.path.join('model', 'genre_nb_model.pkl'))
vectorizer = joblib.load(os.path.join('model', 'tfidf_vectorizer.pkl'))

# Preprocessing function
def preprocess(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Verified movie samples
verified_samples = {
    "💘 Crazy Rich Asians": "A woman discovers her boyfriend is one of Asia’s wealthiest bachelors and must navigate his extravagant world and disapproving family.",
    "💘 Notting Hill": "A shy London bookstore owner falls in love with a famous American actress, but their worlds clash as they try to make the relationship work.",
    "🧠 Inception": "A skilled thief enters people’s dreams to steal secrets, but is tasked with planting an idea instead — a mission that blurs reality and illusion.",
    "🧠 The Matrix": "A hacker discovers reality is a simulation and joins a rebellion to fight the machines controlling humanity.",
    "📜 The King's Speech": "King George VI struggles with a speech impediment and seeks help from an unorthodox therapist to lead Britain through war.",
    "📜 12 Years a Slave": "A free Black man is kidnapped and sold into slavery, enduring years of brutality before fighting for his freedom.",
    "🧙 Harry Potter and the Sorcerer’s Stone": "A boy discovers he’s a wizard and attends a magical school, where he uncovers secrets about his past and a dark force rising.",
    "🧙 The Lord of the Rings: The Fellowship of the Ring": "A hobbit sets out on a quest to destroy a powerful ring that could enslave the world.",
    "👻 It (2017)": "A group of kids in a small town face their worst fears when a shape-shifting clown named Pennywise begins hunting them.",
    "👻 The Conjuring": "Paranormal investigators help a family terrorized by a dark presence in their farmhouse.",
    "🎯 Mission: Impossible – Fallout": "Ethan Hunt and his team race against time to prevent nuclear disaster after a mission goes wrong.",
    "🎯 Skyfall": "James Bond investigates an attack on MI6 and faces a former agent seeking revenge.",
    "♟️ Lady Bird": "A teenage girl navigates her final year of high school, grappling with identity, family tension, and dreams of escaping her hometown.",
    "♟️ The Perks of Being a Wallflower": "A shy teenager navigates friendship, trauma, and self-discovery during his freshman year of high school.",
    "🕵️ Gone Girl": "A man becomes the prime suspect when his wife disappears, but the truth behind her vanishing is darker than anyone imagined.",
    "🕵️ Prisoners": "A father takes matters into his own hands when his daughter and her friend go missing, while a detective follows a trail of disturbing clues."
}

# UI Layout
st.title("🎬 Verified Movie Genre Classifier")
st.write("Choose a verified movie plot or write your own:")

sample_choice = st.selectbox("🎥 Verified Movie Samples", list(verified_samples.keys()) + ["✍️ Custom Input"])

# Input area (always editable)
if sample_choice == "✍️ Custom Input":
    user_input = st.text_area("📝 Enter Your Movie Plot", height=200)
else:
    plot_text = verified_samples[sample_choice]
    st.markdown("**📋 Selected Plot (copy below):**")
    st.code(plot_text)
    user_input = st.text_area("📥 Paste or Edit Plot Here", value=plot_text, height=200)

# Prediction
if st.button("Predict Genre"):
    if user_input.strip():
        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        st.success(f"📌 Predicted Genre: **{prediction}**")
    else:
        st.warning("Please enter or paste a movie plot to classify.")