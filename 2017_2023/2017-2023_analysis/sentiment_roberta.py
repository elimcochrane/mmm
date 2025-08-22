"""
Reddit Post Stance Analyzer
Reads Excel file of Reddit posts and analyzes stance toward primary themes
Creates n=2000 sample if data has more than 2000 rows
"""

import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import re
import os
from collections import defaultdict
import argparse

class ThemeStanceAnalyzer:
    def __init__(self):
        """Initialize the theme stance analyzer"""
        self.llm_pipeline = None
        self.setup_llm_pipeline()
        
        # Define support/opposition indicators for each theme
        self.stance_indicators = {
            'masculinity': {
                'support': ['strong men', 'masculine', 'traditional roles', 'provider', 'protector', 
                          'male leadership', 'fathers', 'responsibility', 'discipline', 'strength'],
                'oppose': ['toxic masculinity', 'patriarchy', 'oppressive', 'outdated', 'harmful',
                          'misogyny', 'sexist', 'domination', 'controlling', 'problematic']
            },
            'politics_events': {
                'support': ['great policy', 'good decision', 'support', 'agree', 'smart move',
                          'necessary', 'finally', 'about time', 'excellent', 'approve'],
                'oppose': ['terrible', 'disaster', 'wrong', 'oppose', 'disagree', 'stupid',
                          'harmful', 'dangerous', 'corrupt', 'awful', 'condemn']
            },
            'politics_culture': {
                'support': ['agree', 'exactly right', 'truth', 'correct', 'spot on',
                          'based', 'makes sense', 'logical', 'reasonable', 'valid point'],
                'oppose': ['wrong', 'disagree', 'nonsense', 'ridiculous', 'absurd',
                          'biased', 'ignorant', 'extreme', 'radical', 'propaganda']
            },
            'peterson': {
                'support': ['brilliant', 'genius', 'wise', 'helped me', 'changed my life',
                          'truth', 'insightful', 'intelligent', 'profound', 'respect'],
                'oppose': ['grifter', 'fraud', 'wrong', 'harmful', 'pseudoscience',
                          'charlatan', 'misleading', 'dangerous', 'cult', 'problematic']
            },
            'humanities_philosophy': {
                'support': ['important', 'valuable', 'wisdom', 'enlightening', 'profound',
                          'meaningful', 'truth', 'insightful', 'necessary', 'fundamental'],
                'oppose': ['useless', 'pointless', 'pretentious', 'waste of time', 'irrelevant',
                          'outdated', 'elitist', 'impractical', 'abstract nonsense']
            },
            'humanities_psychology': {
                'support': ['scientific', 'evidence-based', 'helpful', 'insightful', 'valid',
                          'therapeutic', 'healing', 'growth', 'understanding', 'beneficial'],
                'oppose': ['pseudoscience', 'unproven', 'quackery', 'harmful', 'manipulative',
                          'biased', 'outdated', 'dangerous', 'misleading', 'wrong']
            },
            'women': {
                'support': ['respect women', 'equality', 'empowerment', 'rights', 'strong women',
                          'support', 'appreciate', 'value', 'important role', 'deserve'],
                'oppose': ['don\'t understand', 'complicated', 'emotional', 'irrational',
                          'hypergamy', 'manipulative', 'entitled', 'unreasonable']
            },
            'lgbtq': {
                'support': ['accept', 'support', 'equality', 'rights', 'respect', 'valid',
                          'human dignity', 'inclusion', 'tolerance', 'love'],
                'oppose': ['against', 'unnatural', 'wrong', 'mental illness', 'ideology',
                          'pushing agenda', 'confusing', 'harmful', 'indoctrination']
            },
            'self-help': {
                'support': ['helpful', 'changed my life', 'motivated', 'inspiring', 'practical',
                          'works', 'effective', 'valuable', 'growth', 'improvement'],
                'oppose': ['doesn\'t work', 'scam', 'oversimplified', 'blame victim',
                          'unrealistic', 'privileged', 'toxic positivity', 'harmful']
            }
        }
    
    def setup_llm_pipeline(self):
        """Setup LLM pipeline for stance detection"""
        try:
            self.llm_pipeline = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print("LLM pipeline initialized for stance detection")
        except Exception as e:
            print(f"Warning: Could not initialize LLM pipeline. Error: {str(e)}")
            print("Falling back to lexicon-based analysis only")
            self.llm_pipeline = None
    
    def combine_title_selftext(self, df):
        """
        Combine title and selftext columns into a single text column
        
        Args:
            df (pd.DataFrame): DataFrame with title and selftext columns
            
        Returns:
            pd.DataFrame: DataFrame with combined_text column
        """
        print("Combining title and selftext columns...")
        
        # Convert to strings and handle missing values
        df['title'] = df['title'].astype(str).replace('nan', '')
        df['selftext'] = df['selftext'].astype(str).replace('nan', '')
        
        # Combine title and selftext
        df['combined_text'] = df['title'] + ' ' + df['selftext']
        
        # Clean up the combined text
        df['combined_text'] = df['combined_text'].str.strip()
        
        print(f"Combined text for {len(df)} posts")
        return df
    
    def analyze_stance_lexicon_based(self, text, theme):
        """
        Analyze stance using lexicon-based approach
        
        Args:
            text (str): Text to analyze
            theme (str): Theme to analyze stance toward
            
        Returns:
            tuple: (stance, confidence_score)
        """
        if theme not in self.stance_indicators:
            return 'neutral', 0.0
        
        text_lower = text.lower()
        support_indicators = self.stance_indicators[theme]['support']
        oppose_indicators = self.stance_indicators[theme]['oppose']
        
        support_score = 0
        oppose_score = 0
        
        # Count support indicators
        for indicator in support_indicators:
            if re.search(r'\b' + re.escape(indicator.lower()) + r'\b', text_lower):
                support_score += 1
        
        # Count opposition indicators
        for indicator in oppose_indicators:
            if re.search(r'\b' + re.escape(indicator.lower()) + r'\b', text_lower):
                oppose_score += 1
        
        # Determine stance
        if support_score > oppose_score:
            return 'support', support_score / (support_score + oppose_score + 1)
        elif oppose_score > support_score:
            return 'oppose', oppose_score / (support_score + oppose_score + 1)
        else:
            return 'neutral', 0.0
    
    def analyze_stance_llm_based(self, text, theme):
        """
        Analyze stance using LLM-based approach (BART MNLI)
        
        Args:
            text (str): Text to analyze
            theme (str): Theme to analyze stance toward
            
        Returns:
            tuple: (stance, confidence_score)
        """
        if not self.llm_pipeline:
            return 'neutral', 0.0
        
        try:
            # Create hypothesis for MNLI
            hypothesis_support = f"This text supports {theme}."
            hypothesis_oppose = f"This text opposes {theme}."
            
            # Truncate text to avoid token limits
            text_truncated = text[:500]
            
            # Get predictions - MNLI format expects premise and hypothesis
            result_support = self.llm_pipeline(f"{text_truncated} [SEP] {hypothesis_support}")
            result_oppose = self.llm_pipeline(f"{text_truncated} [SEP] {hypothesis_oppose}")
            
            # Extract scores - handle both dict and list formats
            if isinstance(result_support, list):
                result_support = result_support[0]
            if isinstance(result_oppose, list):
                result_oppose = result_oppose[0]
                
            support_score = result_support['score'] if result_support['label'] == 'ENTAILMENT' else 0
            oppose_score = result_oppose['score'] if result_oppose['label'] == 'ENTAILMENT' else 0
            
            # Determine stance
            if support_score > oppose_score and support_score > 0.5:
                return 'support', support_score
            elif oppose_score > support_score and oppose_score > 0.5:
                return 'oppose', oppose_score
            else:
                return 'neutral', max(support_score, oppose_score)
                
        except Exception as e:
            print(f"Error in LLM stance analysis: {str(e)}")
            return 'neutral', 0.0
    
    def analyze_stance_hybrid(self, text, theme):
        """
        Combine lexicon and LLM approaches
        
        Args:
            text (str): Text to analyze
            theme (str): Theme to analyze stance toward
            
        Returns:
            tuple: (stance, confidence_score)
        """
        # Get results from both methods
        lex_stance, lex_score = self.analyze_stance_lexicon_based(text, theme)
        llm_stance, llm_score = self.analyze_stance_llm_based(text, theme)
        
        # If LLM is not available, use lexicon only
        if not self.llm_pipeline:
            return lex_stance, lex_score
        
        # Combine results (weight LLM higher)
        if lex_stance == llm_stance:
            # Agreement - high confidence
            combined_score = (lex_score * 0.3) + (llm_score * 0.7)
            return lex_stance, combined_score
        elif lex_stance == 'neutral':
            # Lexicon neutral, trust LLM
            return llm_stance, llm_score * 0.8
        elif llm_stance == 'neutral':
            # LLM neutral, trust lexicon
            return lex_stance, lex_score * 0.8
        else:
            # Disagreement - default to neutral with low confidence
            return 'neutral', 0.3
    
    def analyze_dataframe_stances(self, df, theme_column='primary_theme', 
                                text_column='combined_text', method='hybrid'):
        """
        Analyze stance for entire dataframe
        
        Args:
            df (pd.DataFrame): DataFrame with posts and themes
            theme_column (str): Column containing theme assignments
            text_column (str): Column containing text to analyze
            method (str): Method to use ('lexicon', 'llm', or 'hybrid')
            
        Returns:
            pd.DataFrame: DataFrame with stance information added
        """
        print(f"Analyzing stance for {len(df)} posts using {method} method...")
        
        stances = []
        confidence_scores = []
        
        for i, row in df.iterrows():
            text = str(row[text_column])
            theme = row[theme_column]
            
            if pd.isna(theme) or theme == '':
                stance, score = 'neutral', 0.0
            elif method == 'lexicon':
                stance, score = self.analyze_stance_lexicon_based(text, theme)
            elif method == 'llm':
                stance, score = self.analyze_stance_llm_based(text, theme)
            elif method == 'hybrid':
                stance, score = self.analyze_stance_hybrid(text, theme)
            else:
                stance, score = 'neutral', 0.0
            
            stances.append(stance)
            confidence_scores.append(score)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(df)} posts")
        
        # Add results to dataframe
        df['stance'] = stances
        df['stance_confidence'] = confidence_scores
        
        # Print distribution
        self.print_stance_distribution(df, theme_column)
        
        return df
    
    def print_stance_distribution(self, df, theme_column):
        """Print stance distribution by theme"""
        print("\nStance Distribution by Theme:")
        print("-" * 50)
        
        for theme in df[theme_column].unique():
            if pd.isna(theme) or theme == '':
                continue
                
            theme_data = df[df[theme_column] == theme]
            stance_counts = theme_data['stance'].value_counts()
            total = len(theme_data)
            
            print(f"\n{theme} ({total} posts):")
            for stance, count in stance_counts.items():
                percentage = (count / total) * 100
                print(f"  {stance}: {count} ({percentage:.1f}%)")

def process_reddit_file(input_file, output_file=None, method='hybrid'):
    """
    Main function to process Reddit posts Excel file
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output Excel file (optional)
        method (str): Analysis method ('lexicon', 'llm', or 'hybrid')
    
    Returns:
        pd.DataFrame: Processed DataFrame with stance analysis
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Reading Excel file: {input_file}")
    
    # Read Excel file
    try:
        df = pd.read_excel(input_file)
        print(f"Loaded {len(df)} rows from Excel file")
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")
    
    # Check required columns
    required_columns = ['title', 'selftext', 'primary_theme']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print("Required columns found:", required_columns)
    
    # Sample 2000 rows if dataset is larger than 2000
    if len(df) > 2000:
        print(f"Dataset has {len(df)} rows. Sampling 2000 rows for analysis...")
        df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        print(f"Sampled dataset now has {len(df)} rows")
    else:
        print(f"Dataset has {len(df)} rows (≤2000). Using all rows.")

    # Initialize analyzer
    analyzer = ThemeStanceAnalyzer()
    
    # Combine title and selftext
    df = analyzer.combine_title_selftext(df)
    
    # Analyze stances
    df = analyzer.analyze_dataframe_stances(df, 'primary_theme', 'combined_text', method)
    
    # Create output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_with_stance.xlsx"
    
    # Save results
    print(f"\nSaving results to: {output_file}")
    df.to_excel(output_file, index=False)
    
    print(f"Analysis complete! Output saved to: {output_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total posts processed: {len(df)}")
    print(f"Stance distribution:")
    stance_counts = df['stance'].value_counts()
    for stance, count in stance_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {stance}: {count} ({percentage:.1f}%)")
    
    return df

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Analyze stance in Reddit posts')
    parser.add_argument('input_file', help='Path to input Excel file')
    parser.add_argument('-o', '--output', help='Path to output Excel file')
    parser.add_argument('-m', '--method', choices=['lexicon', 'llm', 'hybrid'], 
                       default='hybrid', help='Analysis method (default: hybrid)')
    
    args = parser.parse_args()
    
    try:
        process_reddit_file(args.input_file, args.output, args.method)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

# Example usage functions
def example_usage():
    """Example of how to use the analyzer"""
    
    # Example 1: Using the function directly
    input_file = "reddit_posts.xlsx"
    output_file = "reddit_posts_with_stance.xlsx"
    
    try:
        df = process_reddit_file(input_file, output_file, method='hybrid')
        print("Processing complete!")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Using only lexicon-based analysis (faster, no GPU needed)
    try:
        df = process_reddit_file(input_file, method='lexicon')
        print("Lexicon-based analysis complete!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    # If command line arguments provided, use CLI
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        # Otherwise show example usage
        print("Reddit Post Stance Analyzer")
        print("=" * 30)
        print("\nUsage:")
        print("python script.py input_file.xlsx")
        print("python script.py input_file.xlsx -o output_file.xlsx")
        print("python script.py input_file.xlsx -m lexicon")
        print("\nOr use the process_reddit_file() function directly in your code:")
        print("df = process_reddit_file('your_file.xlsx')")