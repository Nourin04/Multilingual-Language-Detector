import streamlit as st
import joblib
import re

# Load model and utilities
model = joblib.load("language_detector_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict function
def predict_language(text):
    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    pred = model.predict(vector)
    return label_encoder.inverse_transform(pred)[0]

# Streamlit UI
st.title("🌍 Multilingual Language Detector")
st.write("Type or paste a sentence and I'll detect the language (17 supported languages).")

user_input = st.text_area("Enter your text here:", height=100)

if st.button("Detect Language"):
    if user_input.strip() != "":
        lang = predict_language(user_input)
        st.success(f"🧠 Detected Language: **{lang}**")
    else:
        st.warning("Please enter some text to detect.")
