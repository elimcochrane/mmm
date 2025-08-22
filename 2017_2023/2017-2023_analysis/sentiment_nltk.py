"""
Sentiment Analysis Module
Handles sentiment analysis using NLTK's VADER sentiment analyzer
"""

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import ssl

def setup_nltk():
    """Setup NLTK with SSL workaround"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(df, text_column='text_clean'):
    """
    Perform sentiment analysis on text data
    
    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Name of the column containing text to analyze
    
    Returns:
        pd.DataFrame: DataFrame with sentiment scores added
    """
    print("Performing sentiment analysis...")
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis
    sentiment = df[text_column].apply(lambda x: sia.polarity_scores(x[:1000]))
    
    # Add sentiment scores to dataframe
    sentiment_df = pd.json_normalize(sentiment).add_prefix('sentiment_')
    result_df = pd.concat([df, sentiment_df], axis=1)
    
    return result_df