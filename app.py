import streamlit as st
import joblib
import re

# Load model and tools
model = joblib.load("language_detector_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 🌐 Background CSS
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1529070538774-1843cb3265df?auto=format&fit=crop&w=1470&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🧹 Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 🔍 Predict function
def predict_language(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    return label_encoder.inverse_transform(pred)[0]

# 🎯 App Content
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)

    st.title("🌍 Multilingual Language Detector")
    st.markdown("**Detect the language of your text instantly. Supports 17 languages!**")

    st.markdown("📥 *Type a sentence in any language:*")

    user_input = st.text_area("💬 Your Text", height=150, placeholder="Eg: Bonjour, comment ça va?")

    if st.button("🔍 Detect Language"):
        if user_input.strip():
            lang = predict_language(user_input)
            st.success(f"🎉 Detected Language: **{lang}**")
        else:
            st.warning("⚠️ Please enter some text.")

    st.markdown("</div>", unsafe_allow_html=True)
