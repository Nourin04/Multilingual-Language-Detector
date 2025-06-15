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
.stApp {
    background-image: url('https://images.unsplash.com/photo-1519750157634-b6d493a0f77b?auto=format&fit=crop&w=1470&q=80');
    background-size: cover;
    background-attachment: fixed;
}

.glass {
    background: rgba(255, 255, 255, 0.15);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
    max-width: 700px;
    margin: 3rem auto;
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

