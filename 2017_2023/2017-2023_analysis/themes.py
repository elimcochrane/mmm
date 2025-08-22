"""
Thematic Analysis Module
Handles topic modeling using predefined themes with semantic similarity matching
Modified to return top 2 themes per post
"""

# import pandas as pd
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict

class ThematicAnalyzer:
    def __init__(self, custom_themes=None):
        """
        Initialize the thematic analyzer with custom themes
        
        Args:
            custom_themes (dict): Dictionary where keys are theme names and values are lists of keywords/phrases                        
        """
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.custom_themes = custom_themes
        
        # Precompute theme embeddings for semantic matching
        self.theme_embeddings = {}
        self._compute_theme_embeddings()
    
    def _compute_theme_embeddings(self):
        """Compute embeddings for each theme based on their keywords"""
        for theme_name, keywords in self.custom_themes.items():
            # Create a representative text for the theme
            theme_text = " ".join(keywords)
            embedding = self.embedding_model.encode([theme_text])[0]
            self.theme_embeddings[theme_name] = embedding
    
    def preprocess_texts(self, texts):
        """
        Preprocess texts for topic modeling
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: Cleaned and validated texts
        """
        clean_texts = []
        for i, text in enumerate(texts):
            if isinstance(text, str) and len(text.strip()) > 0 and text.lower() not in ['nan', 'none']:
                clean_texts.append(text.strip().lower())
            else:
                placeholder = f"empty post {i}"
                clean_texts.append(placeholder)
                print(f"Fixed invalid text at index {i}: {repr(text)} -> '{placeholder}'")
        
        return clean_texts
    
    def assign_themes_keyword_based(self, texts):
        """
        Assign top 2 themes based on keyword matching
        
        Args:
            texts (list): List of text strings
            
        Returns:
            tuple: (primary_themes, secondary_themes, primary_scores, secondary_scores)
        """
        primary_themes = []
        secondary_themes = []
        primary_scores = []
        secondary_scores = []
        
        for text in texts:
            scores = defaultdict(int)
            
            # Score each theme based on keyword presence
            for theme_name, keywords in self.custom_themes.items():
                for keyword in keywords:
                    # Simple keyword matching (case-insensitive)
                    if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text.lower()):
                        scores[theme_name] += 1
            
            # Get top 2 themes
            if scores:
                # Sort themes by score (descending)
                sorted_themes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                # Primary theme
                primary_themes.append(sorted_themes[0][0])
                primary_scores.append(sorted_themes[0][1])
                
                # Secondary theme
                if len(sorted_themes) > 1 and sorted_themes[1][1] > 0:
                    secondary_themes.append(sorted_themes[1][0])
                    secondary_scores.append(sorted_themes[1][1])
                else:
                    secondary_themes.append(None)
                    secondary_scores.append(0)
            else:
                primary_themes.append("Other")
                secondary_themes.append(None)
                primary_scores.append(0)
                secondary_scores.append(0)
        
        return primary_themes, secondary_themes, primary_scores, secondary_scores
    
    def assign_themes_semantic_based(self, texts):
        """
        Assign top 2 themes based on semantic similarity using embeddings
        
        Args:
            texts (list): List of text strings
            
        Returns:
            tuple: (primary_themes, secondary_themes, primary_scores, secondary_scores)
        """
        print(f"Computing embeddings for {len(texts)} texts...")
        text_embeddings = self.embedding_model.encode(texts)
        
        primary_themes = []
        secondary_themes = []
        primary_scores = []
        secondary_scores = []
        
        theme_list = list(self.custom_themes.keys())
        theme_embedding_matrix = np.array([self.theme_embeddings[theme] for theme in theme_list])
        
        for text_embedding in text_embeddings:
            # Calculate cosine similarity with all themes
            similarities = cosine_similarity([text_embedding], theme_embedding_matrix)[0]
            
            # Get indices of top 2 similarities
            top_2_indices = np.argsort(similarities)[-2:][::-1]  # Descending order
            
            # Primary theme
            primary_idx = top_2_indices[0]
            primary_sim = similarities[primary_idx]
            
            if primary_sim > 0.3:  # Threshold for similarity
                primary_themes.append(theme_list[primary_idx])
                primary_scores.append(round(primary_sim, 3))
            else:
                primary_themes.append("Other")
                primary_scores.append(0)
            
            # Secondary theme
            if len(top_2_indices) > 1:
                secondary_idx = top_2_indices[1]
                secondary_sim = similarities[secondary_idx]
                
                if secondary_sim > 0.3 and secondary_idx != primary_idx:
                    secondary_themes.append(theme_list[secondary_idx])
                    secondary_scores.append(round(secondary_sim, 3))
                else:
                    secondary_themes.append(None)
                    secondary_scores.append(0)
            else:
                secondary_themes.append(None)
                secondary_scores.append(0)
        
        return primary_themes, secondary_themes, primary_scores, secondary_scores
    
    def assign_themes_hybrid(self, texts):
        """
        Assign top 2 themes using both keyword and semantic matching
        
        Args:
            texts (list): List of text strings
            
        Returns:
            tuple: (primary_themes, secondary_themes, primary_scores, secondary_scores)
        """
        # Get results from both methods
        kw_primary, kw_secondary, kw_p_scores, kw_s_scores = self.assign_themes_keyword_based(texts)
        sem_primary, sem_secondary, sem_p_scores, sem_s_scores = self.assign_themes_semantic_based(texts)
        
        final_primary = []
        final_secondary = []
        final_p_scores = []
        final_s_scores = []
        
        for i in range(len(texts)):
            # Combine scores from both methods
            combined_scores = defaultdict(float)
            
            # Add keyword scores (normalized to 0-1 range for combination)
            if kw_primary[i] != "Other":
                combined_scores[kw_primary[i]] += kw_p_scores[i] * 0.1  # Weight keyword matches
            if kw_secondary[i] is not None:
                combined_scores[kw_secondary[i]] += kw_s_scores[i] * 0.1
            
            # Add semantic scores
            if sem_primary[i] != "Other":
                combined_scores[sem_primary[i]] += sem_p_scores[i]
            if sem_secondary[i] is not None:
                combined_scores[sem_secondary[i]] += sem_s_scores[i]
            
            # Get top 2 from combined scores
            if combined_scores:
                sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Primary
                final_primary.append(sorted_combined[0][0])
                final_p_scores.append(round(sorted_combined[0][1], 3))
                
                # Secondary
                if len(sorted_combined) > 1 and sorted_combined[1][1] > 0:
                    final_secondary.append(sorted_combined[1][0])
                    final_s_scores.append(round(sorted_combined[1][1], 3))
                else:
                    final_secondary.append(None)
                    final_s_scores.append(0)
            else:
                final_primary.append("Other")
                final_secondary.append(None)
                final_p_scores.append(0)
                final_s_scores.append(0)
        
        return final_primary, final_secondary, final_p_scores, final_s_scores
    
    def run_custom_theming(self, texts, method="hybrid"):
        """
        Run custom theme assignment on texts, returning top 2 themes
        
        Args:
            texts (list): List of text strings
            method (str): Method to use - "keyword", "semantic", or "hybrid"
            
        Returns:
            tuple: (primary_themes, secondary_themes, primary_scores, secondary_scores)
        """
        try:
            print(f"Running custom theming on {len(texts)} texts using {method} method...")
            
            if method == "keyword":
                return self.assign_themes_keyword_based(texts)
            elif method == "semantic":
                return self.assign_themes_semantic_based(texts)
            elif method == "hybrid":
                return self.assign_themes_hybrid(texts)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            print(f"Custom theming failed: {str(e)}")
            # Fallback to assigning everything as "Other"
            return (["Other"] * len(texts), [None] * len(texts), 
                   [0] * len(texts), [0] * len(texts))
    
    def get_theme_distribution(self, primary_themes, secondary_themes=None):
        """Get distribution of themes including secondary themes"""
        from collections import Counter
        
        primary_dist = Counter(primary_themes)
        print("Primary theme distribution:")
        for theme, count in primary_dist.items():
            print(f"  {theme}: {count}")
        
        if secondary_themes:
            # Count non-None secondary themes
            secondary_clean = [theme for theme in secondary_themes if theme is not None]
            if secondary_clean:
                secondary_dist = Counter(secondary_clean)
                print("\nSecondary theme distribution:")
                for theme, count in secondary_dist.items():
                    print(f"  {theme}: {count}")
        
        return primary_dist
    
    def add_theme(self, theme_name, keywords):
        """
        Add a new theme to the analyzer
        
        Args:
            theme_name (str): Name of the new theme
            keywords (list): List of keywords for the theme
        """
        self.custom_themes[theme_name] = keywords
        # Recompute embeddings
        theme_text = " ".join(keywords)
        embedding = self.embedding_model.encode([theme_text])[0]
        self.theme_embeddings[theme_name] = embedding
        print(f"Added theme '{theme_name}' with keywords: {keywords}")

def analyze_topics(df, text_column='text_clean', custom_themes=None, method="hybrid"):
    """
    Perform topic analysis on text data using custom themes, returning top 2 themes
    
    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Name of the column containing text to analyze
        custom_themes (dict): Dictionary of custom themes and their keywords
        method (str): Method to use - "keyword", "semantic", or "hybrid"
    
    Returns:
        pd.DataFrame: DataFrame with topic information added
    """
    analyzer = ThematicAnalyzer(custom_themes=custom_themes)
    
    texts = df[text_column].tolist()
    clean_texts = analyzer.preprocess_texts(texts)
    
    if len(clean_texts) != len(df):
        raise ValueError(f"Text count mismatch: {len(clean_texts)} texts vs {len(df)} dataframe rows")
    
    primary_themes, secondary_themes, primary_scores, secondary_scores = analyzer.run_custom_theming(clean_texts, method=method)
    
    # Add columns for top 2 themes
    df['primary_theme'] = primary_themes
    df['secondary_theme'] = secondary_themes
    df['primary_theme_score'] = primary_scores
    df['secondary_theme_score'] = secondary_scores
    
    # Print theme distribution
    analyzer.get_theme_distribution(primary_themes, secondary_themes)
    
    return df