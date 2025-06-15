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
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1614107151491-6876eecbff89?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    .main-container {
        background-color: rgba(255, 255, 255, 0.08);
        padding: 2rem;
        border-radius: 1rem;
        margin: 2rem auto;
        box-shadow: 0 0 15px rgba(0,0,0,0.3);
        max-width: 800px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 📌 Sidebar: Supported Languages
st.sidebar.title("🌐 Supported Languages")
languages = [
    "🇬🇧 English", "🇮🇳 Malayalam", "🇮🇳 Hindi", "🇮🇳 Tamil", "🇮🇳 Kannada",
    "🇫🇷 French", "🇪🇸 Spanish", "🇵🇹 Portuguese", "🇮🇹 Italian", "🇷🇺 Russian",
    "🇸🇪 Swedish", "🇳🇱 Dutch", "🇸🇦 Arabic", "🇹🇷 Turkish", "🇩🇪 German",
    "🇩🇰 Danish", "🇬🇷 Greek"
]
for lang in languages:
    st.sidebar.markdown(f"- {lang}")

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
            st.markdown(f"""
            <div style="
                background-color: rgba(255, 255, 255, 0.7);
                padding: 1.5rem;
                border-radius: 1rem;
                margin-top: 2rem;
                text-align: center;
                font-size: 1.6rem;
                font-weight: 600;
                color: #1e3a8a;
                box-shadow: 0 8px 25px rgba(0,0,0,0.25);">
                🎉 Detected Language: <span style="color:#059669;">{lang}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text.")

    st.markdown("</div>", unsafe_allow_html=True)
