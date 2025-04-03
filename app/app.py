import streamlit as st
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import re
import hydralit_components as hc

# Set page config
st.set_page_config(
    page_title="Text Analysis Dashboard",
    page_icon="📊",
    # layout="wide",
    # initial_sidebar_state="expanded"

)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem;
        height:100vh;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
    }
    .result-box {
        background-color: #ffffff;
        # padding: 1.5rem;
        border-radius: 15px;
        # margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    .model-title {
        color: #FF4B4B;
        font-weight: bold;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .result-value {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .sentiment-positive {
        background-color: #e6f4ea;
        color: #1e7e34;
    }
    .sentiment-negative {
        background-color: #fce8e6;
        color: #dc3545;
    }
    .sentiment-neutral {
        background-color: #f8f9fa;
        color: #6c757d;
    }
    .emotion-joy {
        background-color: #fff3cd;
        color: #856404;
    }
    .emotion-sad {
        background-color: #cce5ff;
        color: #004085;
    }
    .emotion-anger {
        background-color: #f8d7da;
        color: #721c24;
    }
    .emotion-love {
        background-color: #f8d7da;
        color: #721c24;
    }
    .emotion-surprise {
        background-color: #d4edda;
        color: #155724;
    }
    .emotion-fear {
        background-color: #e2e3e5;
        color: #383d41;
    }
    .emotion-others {
        background-color: #f8f9fa;
        color: #6c757d;
    }
    .mbti-box {
        background-color: #e3f2fd;
        color: #0d47a1;
    }
    .country-box {
        background-color: #f3e5f5;
        color: #4a148c;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar: Instructions and Text Input
st.sidebar.title("Welcome to the **Text Analysis Dashboard**!")
st.sidebar.markdown("""

**How to use:**
1. Enter your text in the text area below.
2. Click on **Analyze Text**.
3. The analysis results will be displayed on the right as a conversation-like output.

*For best results, please enter at least 50 characters of text.*
""")

text_input = st.sidebar.text_area(
    label="Your Text Input",
    height=250,
    placeholder="Type or paste your text here..."
    # on_change=update_results
)

if st.sidebar.button("Submit Now"):
    st.sidebar.write("Your task is generated!")
    if text_input:
        # Input validation
        text_length = len(text_input.strip())
        if text_length < 10:
            st.sidebar.warning("⚠️ Please enter at least 10 characters.")
        elif text_length < 20:
            st.sidebar.info("ℹ️ For better results, consider entering more text (at least 20 characters).")
        elif text_length < 50:
            st.sidebar.info("ℹ️ For most accurate predictions, consider entering at least 50 characters.")
        
        # Proceed with analysis only if text exists
        with st.spinner("Analyzing text..."):
            # Store the text input in session_state to simulate a conversation dialogue
            st.session_state.user_text = text_input

            def clean_text(text):
                text = text.lower()  
                text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url', text).strip() # replace url
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
                text = re.sub(r'\s+', ' ', text).strip() 
                return text

            @st.cache_resource
            def load_models():
                try:
                    # Load sentiment model and its components
                    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                        "./models/sentiment_model",
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    sentiment_tokenizer = AutoTokenizer.from_pretrained(
                        "./models/sentiment_model",
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    sentiment_model.eval()
                    sentiment_label_encoder = joblib.load("./models/sentiment_model/sentiment_label_encoder.pkl")
                    
                    # Load emotion model
                    emotion_model = joblib.load("./models/emotion_model/linear_svm_tfidf_model.joblib")
                    
                    # Load MBTI models
                    mbti_models = {
                        'I/E': joblib.load("./models/mbti_model/mbti_I_E_SVC.joblib"),
                        'N/S': joblib.load("./models/mbti_model/mbti_N_S_SVC.joblib"),
                        'T/F': joblib.load("./models/mbti_model/mbti_T_F_SVC.joblib"),
                        'J/P': joblib.load("./models/mbti_model/mbti_J_P_SVC.joblib")
                    }
                    mbti_dimension_mapping = {
                        'I/E': {1: 'I', 0: 'E'},
                        'N/S': {1: 'N', 0: 'S'},
                        'T/F': {1: 'T', 0: 'F'},
                        'J/P': {1: 'J', 0: 'P'}
                    }
                    
                    # Load country model and its components
                    country_model = AutoModelForSequenceClassification.from_pretrained(
                        "./models/country_model",
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    country_tokenizer = AutoTokenizer.from_pretrained(
                        "./models/country_model",
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    with open("./models/country_model/label_mapping.json", "r") as f:
                        country_label_mapping = json.load(f)
                    country_id2label = {v: k for k, v in country_label_mapping.items()}
                    
                    return (sentiment_model, sentiment_tokenizer, sentiment_label_encoder, emotion_model, 
                            mbti_models, mbti_dimension_mapping, country_model, country_tokenizer, country_id2label)
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
                    st.error("Please ensure all model files are present in their respective directories.")
                    st.stop()

            try:
                sentiment_model, sentiment_tokenizer, sentiment_label_encoder, emotion_model, \
                mbti_models, mbti_dimension_mapping, country_model, country_tokenizer, country_id2label = load_models()
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                st.stop()

            def predict_mbti(text):
                cleaned_text = clean_text(text)
                mbti = []
                for dim, model in mbti_models.items():
                    pred = model.predict([cleaned_text])[0]
                    mbti.append(mbti_dimension_mapping[dim][pred])
                return ''.join(mbti)

            # Sentiment Analysis
            sentiment_inputs = sentiment_tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                sentiment_outputs = sentiment_model(**sentiment_inputs)
            sentiment_pred = torch.argmax(sentiment_outputs.logits, dim=1).item()
            sentiment_result = sentiment_label_encoder.inverse_transform([sentiment_pred])[0]

            # Emotion Detection
            emotion_result = emotion_model.predict([text_input])[0]

            # MBTI Prediction
            mbti_result = predict_mbti(text_input)

            # Country Classification
            country_inputs = country_tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                country_outputs = country_model(**country_inputs)
            country_pred = torch.argmax(country_outputs.logits, dim=1).item()
            country_result = country_id2label[country_pred]

            st.session_state.results = {
                "sentiment": sentiment_result,
                "emotion": emotion_result,
                "mbti": mbti_result,
                "country": country_result,
                "user_input": text_input
            }
    else:
        st.sidebar.warning("Please enter some text to analyze.")

# Main area: Display conversation-like dialog
st.title("📊 Analysis Results")
cc = []
for i in range(4):
    col1, col2 = st.columns(2)
    cc.append((col1, col2))

if "results" in st.session_state:
    user_text = st.session_state.results.get("user_input", "")
    sentiment_result = st.session_state.results.get("sentiment", "")
    emotion_result = st.session_state.results.get("emotion", "")
    mbti_result = st.session_state.results.get("mbti", "")
    country_result = st.session_state.results.get("country", "")
    

    sentiment_icons = "😊" if sentiment_result.lower() == "positive" else "fa-solid fa-face-laugh-squint" if sentiment_result.lower() == "negative" else "😐"
    emotion_icons = {
        'joy': '😊', 'sad': '😢', 'anger': '😠', 'love': '❤️',
        'surprise': '😮', 'fear': '😨', 'others': '🤔'
    }
    mbti_images = {
    'INTJ': 'app/assets/intj.png',
    'INTP': 'app/assets/intp.png',
    'ENTJ': 'app/assets/entj.png',
    'ENTP': 'app/assets/entp.png',
    'INFJ': 'app/assets/infj.png',
    'INFP': 'app/assets/infp.png',
    'ENFJ': 'app/assets/enfj.png',
    'ENFP': 'app/assets/enfp.png',
    'ISTJ': 'app/assets/istj.png',
    'ISFJ': 'app/assets/isfj.png',
    'ESTJ': 'app/assets/estj.png',
    'ESFJ': 'app/assets/esfj.png',
    'ISTP': 'app/assets/istp.png',
    'ISFP': 'app/assets/isfp.png',
    'ESTP': 'app/assets/estp.png',
    'ESFP': 'app/assets/esfp.png'
}

    theme_bad = {'bgcolor': '#FFF0F0','title_color': 'red','content_color': 'red','icon_color': 'transparent',  # 让图标不可见
    'icon': None}
    theme_neutral = {'bgcolor': '#f9f9f9','title_color': 'orange','content_color': 'orange','icon_color': 'transparent',  # 让图标不可见
    'icon': None}
    theme_good = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'transparent',  # 让图标不可见
    'icon': None}

    mbti_theme = {
        'bgcolor': '#e3f2fd',
        'title_color': '#0d47a1',
        'content_color': '#0d47a1',
    }

    country_theme = {
        'bgcolor': '#f3e5f5',
        'title_color': '#4a148c',
        'content_color': '#4a148c',
    }

    emotion_theme_mapping = {
        'joy': theme_good,
        'love': theme_good,
        'surprise': theme_good,
        'sad': theme_bad,
        'anger': theme_bad,
        'fear': theme_bad,
        'others': theme_neutral
    }
    with cc[0][0]:
    # can just use 'good', 'bad', 'neutral' sentiment to auto color the card
        hc.info_card(title='Sentiment', content=sentiment_result+": "+sentiment_icons,
                     bar_value=0,
                     theme_override=theme_good if sentiment_result.lower() == 'positive' else (theme_bad if sentiment_result.lower() == 'negative' else theme_neutral)
                    )

    with cc[0][1]:
        hc.info_card(title='Emotion', content=emotion_result+": "+emotion_icons[emotion_result],bar_value=0,
                             theme_override=emotion_theme_mapping[emotion_result],
)

    with cc[1][1]:
        hc.info_card(title='MBTI', content=mbti_result, bar_value=0,theme_override=mbti_theme)

    with cc[1][0]:
        st.image(mbti_images[mbti_result], width=180)

    with cc[2][0]:
        hc.info_card(title='Country',content=country_result,key='sec',bar_value=0,theme_override=country_theme)
else:
    st.info("Enter text in the sidebar and click **Analyze Text** to see the analysis results.")
