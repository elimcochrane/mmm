"""
Virality Analysis Module
Handles calculation of virality scores based on engagement metrics
"""

import pandas as pd
import numpy as np

def calculate_virality_score(df, virality_columns=['score', 'num_comments']):
    """
    Calculate virality score based on engagement metrics
    
    Args:
        df (pd.DataFrame): DataFrame containing engagement data
        virality_columns (list): List of column names for virality calculation
        
    Returns:
        pd.DataFrame: DataFrame with virality scores added
    """
    print("Calculating virality scores...")
    
    # Check if required columns exist
    if not all(col in df.columns for col in virality_columns):
        missing_cols = [col for col in virality_columns if col not in df.columns]
        print(f"Warning: Virality columns {missing_cols} not found in data")
        df['virality_score'] = 0
        return df
    
    try:
        # Calculate virality score (comments weighted more heavily)
        score_col = virality_columns[0]  # 'score'
        comments_col = virality_columns[1]  # 'num_comments'
        
        df['virality_score'] = (
            pd.to_numeric(df[comments_col], errors='coerce').fillna(0) * 2 +
            pd.to_numeric(df[score_col], errors='coerce').fillna(0)
        )
        
        # Print virality statistics
        virality_stats = df['virality_score'].describe()
        print("Virality score statistics:")
        print(f"  Mean: {virality_stats['mean']:.2f}")
        print(f"  Std: {virality_stats['std']:.2f}")
        print(f"  Max: {virality_stats['max']:.2f}")
        print(f"  High virality (>90th percentile): {(df['virality_score'] > df['virality_score'].quantile(0.9)).sum()} posts")
        
        return df
        
    except Exception as e:
        print(f"Error calculating virality scores: {str(e)}")
        df['virality_score'] = 0
        return df

def analyze_virality_patterns(df):
    """
    Analyze patterns in virality data
    
    Args:
        df (pd.DataFrame): DataFrame with virality scores
        
    Returns:
        dict: Dictionary containing virality analysis results
    """
    try:
        if 'virality_score' not in df.columns:
            print("No virality scores found in data")
            return {}
        
        print("Analyzing virality patterns...")
        
        # Basic virality statistics
        virality_analysis = {
            'total_posts': len(df),
            'mean_virality': df['virality_score'].mean(),
            'median_virality': df['virality_score'].median(),
            'top_10_percent_threshold': df['virality_score'].quantile(0.9),
            'viral_posts_count': (df['virality_score'] > df['virality_score'].quantile(0.9)).sum()
        }
        
        # Virality by topic (if topics exist)
        if 'topic_name' in df.columns:
            topic_virality = df.groupby('topic_name')['virality_score'].agg(['mean', 'count', 'max']).round(2)
            virality_analysis['virality_by_topic'] = topic_virality.to_dict()
        
        # Virality by sentiment (if sentiment exists)
        if 'sentiment_compound' in df.columns:
            # Categorize sentiment
            df['sentiment_category'] = pd.cut(
                df['sentiment_compound'], 
                bins=[-1, -0.1, 0.1, 1], 
                labels=['Negative', 'Neutral', 'Positive']
            )
            sentiment_virality = df.groupby('sentiment_category')['virality_score'].agg(['mean', 'count']).round(2)
            virality_analysis['virality_by_sentiment'] = sentiment_virality.to_dict()
        
        return virality_analysis
        
    except Exception as e:
        print(f"Error in virality pattern analysis: {str(e)}")
        return {}

def identify_viral_content(df, threshold_percentile=0.95):
    """
    Identify viral content based on virality threshold
    
    Args:
        df (pd.DataFrame): DataFrame with virality scores
        threshold_percentile (float): Percentile threshold for viral content
        
    Returns:
        pd.DataFrame: DataFrame containing only viral content
    """
    try:
        if 'virality_score' not in df.columns:
            print("No virality scores found in data")
            return pd.DataFrame()
        
        threshold = df['virality_score'].quantile(threshold_percentile)
        viral_content = df[df['virality_score'] >= threshold].copy()
        
        print(f"Identified {len(viral_content)} viral posts (>{threshold_percentile*100}th percentile)")
        print(f"Virality threshold: {threshold:.2f}")
        
        # Sort by virality score
        viral_content = viral_content.sort_values('virality_score', ascending=False)
        
        return viral_content
        
    except Exception as e:
        print(f"Error identifying viral content: {str(e)}")
        return pd.DataFrame()