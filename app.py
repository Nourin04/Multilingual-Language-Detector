import streamlit as st
import joblib
import re

# Load model and tools
model = joblib.load("language_detector_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ✨ Inject Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');

* {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background-image: url('https://images.unsplash.com/photo-1614107151491-6876eecbff89?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
    background-size: cover;
    background-attachment: fixed;
}

.glass {
    background: rgba(255, 255, 255, 0.15);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    max-width: 700px;
    margin: 3rem auto;
}

h1, h2, h3 {
    color: #ffffff;
    text-shadow: 1px 1px 2px #00000060;
}

textarea, .stButton>button {
    font-size: 1rem;
}

.stButton>button {
    background-color: #04C38E;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    border: none;
    transition: all 0.3s ease-in-out;
}

.stButton>button:hover {
    background-color: #02a97a;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

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
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    return label_encoder.inverse_transform(pred)[0]

# 🌐 UI with Glassmorphism Container
with st.container():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.markdown("### 🌍 Multilingual Language Detector")
    st.markdown("Detect the language of your text from **17 global languages** 🌐")

    user_input = st.text_area("✍️ Enter text here", height=120, placeholder="Eg: Bonjour, comment ça va?")

    if st.button("🔍 Detect Language"):
        if user_input.strip():
            lang = predict_language(user_input)
            st.success(f"✅ Detected Language: **{lang}**")
        else:
            st.warning("⚠️ Please enter some text.")

    st.markdown("</div>", unsafe_allow_html=True)

