# MindReader: Multi-Model Text Analysis Dashboard

MindReader is a powerful text analysis dashboard that delves into the depths of written content using multiple machine learning models. It provides comprehensive insights into sentiment, emotions, personality type (MBTI), and country of origin based on writing style.

## Features

- **Sentiment Analysis**: Determines if the text expresses positive, negative, or neutral sentiment
- **Emotion Detection**: Identifies the primary emotion (joy, sadness, anger, love, surprise, fear, or others)
- **MBTI Personality Prediction**: Predicts the writer's MBTI personality type
- **Country Classification**: Identifies if the text is from the United States or United Kingdom based on writing style

## Installation

1. Install Git LFS (Large File Storage):
   - On macOS with Homebrew:
     ```bash
     brew install git-lfs
     ```
   - On Windows with Chocolatey:
     ```bash
     choco install git-lfs
     ```
   - On Linux:
     ```bash
     curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
     sudo apt-get install git-lfs
     ```

2. Clone the repository:
```bash
git clone https://github.com/tianyuanzoe/MindReader.git
cd MindReader
```

3. Pull the large model files using Git LFS:
```bash
git lfs install
git lfs pull
```

4. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

5. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The project requires the following main dependencies:

```
streamlit>=1.24.0
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.0.0
joblib>=1.3.0
numpy>=1.24.0
pandas>=1.5.0
```

## Project Structure

```
.
├── app/
│   └── app.py              # Main Streamlit application
├── models/
│   ├── sentiment_model/    # Sentiment analysis model files
│   ├── emotion_model/      # Emotion detection model files
│   ├── mbti_model/        # MBTI personality model files
│   └── country_model/     # Country classification model files
├── requirements.txt       # Project dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app/app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter your text in the input area and click "Analyze Text"

4. View the results from all four models:
   - Sentiment Analysis (Positive, Negative, or Neutral)
   - Emotion Detection (joy, sadness, anger, love, surprise, fear, or others)
   - MBTI Personality Prediction (one of 16 personality types)
   - Country Classification (United States or United Kingdom)

## Model Details

### Sentiment Analysis
- Uses a pre-trained transformer model
- Classifies text as Positive, Negative, or Neutral
- Includes label encoding for proper class mapping

### Emotion Detection
- Uses a Linear SVM model with TF-IDF vectorization
- Detects seven emotions: joy, sadness, anger, love, surprise, fear, and others

### MBTI Personality Prediction
- Uses a Support Vector Classification model
- Predicts one of the 16 MBTI personality types based on writing style

### Country Classification
- Uses a pre-trained transformer model
- Currently supports classification between:
  - United States (US)
  - United Kingdom (UK)
- Identifies country of origin based on writing patterns and language usage
- Includes label mapping for US/UK classification

## Limitations

- Country classification is currently limited to distinguishing between US and UK writing styles
- The model may not accurately classify text from other countries
- Results are based on writing style patterns and may not be 100% accurate

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- Thanks to all the model contributors and maintainers
- Built with Streamlit 