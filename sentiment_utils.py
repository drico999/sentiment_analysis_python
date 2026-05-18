import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Ensure the VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize the analyzer
sia = SentimentIntensityAnalyzer()

def analyze_single_text(text: str) -> dict:
    """
    Analyzes a single string of text and returns the sentiment scores and label.
    """
    if not text or text.strip() == "":
        return {"compound": 0, "label": "Neutral", "pos": 0, "neu": 1.0, "neg": 0}

    scores = sia.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        label = "Positive"
    elif scores['compound'] <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
        
    scores['label'] = label
    return scores

def analyze_batch(texts: list) -> pd.DataFrame:
    """
    Analyzes a list of texts and returns a DataFrame with the texts, scores, and labels.
    """
    results = []
    for text in texts:
        scores = analyze_single_text(str(text))
        results.append({
            "Text": text,
            "Label": scores["label"],
            "Compound": scores["compound"],
            "Positive": scores["pos"],
            "Neutral": scores["neu"],
            "Negative": scores["neg"]
        })
        
    return pd.DataFrame(results)