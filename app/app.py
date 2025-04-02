import streamlit as st
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import numpy as np

# Set page config
st.set_page_config(
    page_title="Text Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
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
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
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

# Title and description
st.title("📊 Text Analysis Dashboard")
st.markdown("""
    This dashboard analyzes your text using four different models:
    - Sentiment Analysis: Determines if the text expresses positive, negative, or neutral sentiment
    - Emotion Detection: Identifies the primary emotion (joy, sadness, anger, love, surprise, fear, or others)
    - MBTI Personality Prediction: Predicts the writer's MBTI personality type
    - Country Classification: Identifies the country of origin based on writing style
""")

# Load models
@st.cache_resource
def load_models():
    # Load sentiment model and its components
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("../models/sentiment_model")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("../models/sentiment_model")
    sentiment_model.eval()
    sentiment_label_encoder = joblib.load("../models/sentiment_model/sentiment_label_encoder.pkl")
    
    # Load emotion model
    emotion_model = joblib.load("../models/emotion_model/linear_svm_tfidf_model.joblib")
    
    # Load MBTI model
    mbti_model = joblib.load("../models/mbti_model/mbti-vectorized-SVC.joblib")
    
    # Load country model and its components
    country_model = AutoModelForSequenceClassification.from_pretrained("../models/country_model")
    country_tokenizer = AutoTokenizer.from_pretrained("../models/country_model")
    with open("../models/country_model/label_mapping.json", "r") as f:
        country_label_mapping = json.load(f)
    country_id2label = {v: k for k, v in country_label_mapping.items()}
    
    return (sentiment_model, sentiment_tokenizer, sentiment_label_encoder, emotion_model, mbti_model, 
            country_model, country_tokenizer, country_id2label)

# Load models
try:
    sentiment_model, sentiment_tokenizer, sentiment_label_encoder, emotion_model, mbti_model, country_model, country_tokenizer, country_id2label = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Create two columns for input and results
col1, col2 = st.columns([1, 1])

with col1:
    # Text input
    st.subheader("Enter Your Text")
    text_input = st.text_area(
        label="Text to Analyze",
        label_visibility="visible",
        height=150,
        placeholder="Type or paste your text here...",
        key="text_input",
        help="Enter the text you want to analyze. The text will be processed by all four models simultaneously."
    )
    
    # Analyze button
    if st.button("Analyze Text", key="analyze", help="Click to analyze the entered text using all four models"):
        if text_input:
            with st.spinner("Analyzing text..."):
                # Sentiment Analysis
                sentiment_inputs = sentiment_tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    sentiment_outputs = sentiment_model(**sentiment_inputs)
                sentiment_pred = torch.argmax(sentiment_outputs.logits, dim=1).item()
                sentiment_result = sentiment_label_encoder.inverse_transform([sentiment_pred])[0]
                
                # Emotion Detection
                emotion_result = emotion_model.predict([text_input])[0]
                
                # MBTI Prediction
                mbti_result = mbti_model.predict([text_input])[0]
                
                # Country Classification
                country_inputs = country_tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    country_outputs = country_model(**country_inputs)
                country_pred = torch.argmax(country_outputs.logits, dim=1).item()
                country_result = country_id2label[country_pred]
                
                # Store results in session state
                st.session_state.results = {
                    "sentiment": sentiment_result,
                    "emotion": emotion_result,
                    "mbti": mbti_result,
                    "country": country_result
                }
        else:
            st.warning("Please enter some text to analyze.")

with col2:
    st.subheader("Analysis Results")
    if hasattr(st.session_state, 'results'):
        # Sentiment Result
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        sentiment_icon = "😊" if st.session_state.results["sentiment"] == "Positive" else "😢" if st.session_state.results["sentiment"] == "Negative" else "😐"
        st.markdown(f'<div class="model-title">{sentiment_icon} Sentiment Analysis</div>', unsafe_allow_html=True)
        sentiment_class = f"sentiment-{st.session_state.results['sentiment'].lower()}"
        st.markdown(f'<div class="result-value {sentiment_class}">{st.session_state.results["sentiment"]}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">Overall emotional tone of the text</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Emotion Result
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        emotion_icons = {
            'joy': '😊', 'sad': '😢', 'anger': '😠', 'love': '❤️',
            'surprise': '😮', 'fear': '😨', 'others': '🤔'
        }
        emotion_icon = emotion_icons.get(st.session_state.results["emotion"].lower(), '🤔')
        st.markdown(f'<div class="model-title">{emotion_icon} Emotion Detection</div>', unsafe_allow_html=True)
        emotion_class = f"emotion-{st.session_state.results['emotion'].lower()}"
        st.markdown(f'<div class="result-value {emotion_class}">{st.session_state.results["emotion"].title()}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">Primary emotion expressed in the text</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # MBTI Result
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<div class="model-title">🧠 MBTI Personality</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value mbti-box">{st.session_state.results["mbti"]}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">Predicted personality type based on writing style</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Country Result
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<div class="model-title">🌍 Country Classification</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value country-box">{st.session_state.results["country"]}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">Predicted country of origin based on writing style</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Enter text and click 'Analyze Text' to see results.")


