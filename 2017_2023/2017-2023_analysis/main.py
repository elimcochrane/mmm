"""
Main Social Post Analyzer
Combines all analysis modules to provide comprehensive social media post analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing import load_and_validate_data, validate_and_prepare_data
from sentiment_nltk import setup_nltk, analyze_sentiment
from themes import analyze_topics
from temporal import (detect_temporal_spikes, analyze_temporal_patterns, 
                     prepare_temporal_data, create_simple_timeline_graph,
                     run_temporal_analysis_with_graphs)
from sentiment_roberta import analyze_stance
from virality import calculate_virality_score, analyze_virality_patterns, identify_viral_content

class SocialPostAnalyzer:
    def __init__(self, text_columns, date_column='date', virality_columns=None, 
                 custom_themes=None, theme_method="hybrid"):
        """
        Initialize the Social Post Analyzer
        
        Args:
            text_columns (list): List of text column names
            date_column (str): Name of date column
            virality_columns (list): List of virality column names
            custom_themes (dict): Dictionary of custom themes and their keywords
            theme_method (str): Method for theme assignment - "keyword", "semantic", or "hybrid"
        """
        self.text_columns = text_columns
        self.date_column = date_column
        self.virality_columns = virality_columns or ['score', 'num_comments']
        self.custom_themes = custom_themes
        self.theme_method = theme_method
        
        setup_nltk()
        
        print("SocialPostAnalyzer initialized successfully")
        if custom_themes:
            print(f"Custom themes loaded: {list(custom_themes.keys())}")
            print(f"Theme assignment method: {theme_method}")
    
    def analyze_posts(self, df):
        """
        Run comprehensive analysis on social media posts
        
        Args:
            df (pd.DataFrame): DataFrame containing social media posts
            
        Returns:
            pd.DataFrame: DataFrame with analysis results
        """
        try:
            print("="*50)
            print("STARTING COMPREHENSIVE SOCIAL POST ANALYSIS")
            print("="*50)
            
            # step 1: data preprocessing and validation
            print("\n1. DATA PREPROCESSING")
            print("-" * 30)
            df = validate_and_prepare_data(
                df, 
                self.text_columns, 
                self.date_column, 
                self.virality_columns
            )
            
            # step 2: sentiment analysis
            print("\n2. SENTIMENT ANALYSIS")
            print("-" * 30)
            df = analyze_sentiment(df)
            
            # step 3: thematic analysis (topic modeling)
            print("\n3. THEMATIC ANALYSIS")
            print("-" * 30)
            df = analyze_topics(df, 
                              custom_themes=self.custom_themes, 
                              method=self.theme_method)
            
            # step 4: temporal analysis preparation
            print("\n4. TEMPORAL ANALYSIS PREPARATION")
            print("-" * 30)
            df = prepare_temporal_data(df, self.date_column)
            
            # step 5: stance analysis
            print("\n5. STANCE ANALYSIS")
            print("-" * 30)
            df = analyze_stance(df)
            
            # step 6: virality analysis
            print("\n6. VIRALITY ANALYSIS")
            print("-" * 30)
            df = calculate_virality_score(df, self.virality_columns)
            
            print("\n" + "="*50)
            print("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"Final dataset contains {len(df)} analyzed posts")
            
            return df
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def detect_temporal_spikes(self, df, plot=True):
        """
        Detect temporal spikes in posting activity with optional plotting
        
        Args:
            df (pd.DataFrame): DataFrame with temporal data
            plot (bool): Whether to create line graphs
            
        Returns:
            pd.Series: Series containing spike data
        """
        return detect_temporal_spikes(df, self.date_column, plot=plot)
    
    def analyze_temporal_patterns(self, df, plot=True):
        """
        Analyze temporal patterns with optional plotting
        
        Args:
            df (pd.DataFrame): DataFrame with temporal data
            plot (bool): Whether to create line graphs
            
        Returns:
            dict: Dictionary containing temporal analysis results
        """
        return analyze_temporal_patterns(df, self.date_column, plot=plot)
    
    def create_timeline_graph(self, df, title=None):
        """
        Create a simple timeline graph
        
        Args:
            df (pd.DataFrame): DataFrame with temporal data
            title (str): Custom title for the graph
        """
        if title is None:
            title = "Social Media Post Timeline"
        create_simple_timeline_graph(df, self.date_column, title)
    
    def run_full_temporal_analysis(self, df):
        """
        Run complete temporal analysis with all graphs
        
        Args:
            df (pd.DataFrame): DataFrame with temporal data
            
        Returns:
            tuple: (temporal_patterns, spikes)
        """
        return run_temporal_analysis_with_graphs(df, self.date_column)
    
    def generate_insights(self, df):
        """
        Generate analytical insights from the processed data
        
        Args:
            df (pd.DataFrame): DataFrame with analysis results
            
        Returns:
            dict: Dictionary containing insights
        """
        try:
            print("\nGENERATING ANALYTICAL INSIGHTS")
            print("-" * 40)
            
            insights = {}
            
            # basic statistics
            insights['basic_stats'] = {
                'total_posts': len(df),
                'unique_topics': df['topic_id'].nunique() if 'topic_id' in df.columns else 0,
                'date_range': {
                    'start': df[self.date_column].min() if self.date_column in df.columns else None,
                    'end': df[self.date_column].max() if self.date_column in df.columns else None
                }
            }
            
            # sentiment insights
            if 'sentiment_compound' in df.columns:
                insights['sentiment'] = {
                    'average_sentiment': df['sentiment_compound'].mean(),
                    'positive_posts': (df['sentiment_compound'] > 0.1).sum(),
                    'negative_posts': (df['sentiment_compound'] < -0.1).sum(),
                    'neutral_posts': ((df['sentiment_compound'] >= -0.1) & (df['sentiment_compound'] <= 0.1)).sum()
                }
            
            # theme insights
            if 'topic_name' in df.columns:
                insights['themes'] = {
                    'theme_distribution': df['topic_name'].value_counts().to_dict(),
                    'most_common_theme': df['topic_name'].mode().iloc[0] if not df['topic_name'].mode().empty else None,
                    'total_themes': df['topic_name'].nunique()
                }
            
            # virality insights
            insights['virality'] = analyze_virality_patterns(df)
            
            # temporal insights (without plotting here to avoid duplicate graphs)
            insights['temporal'] = analyze_temporal_patterns(df, self.date_column, plot=False)
            
            print("Insights generated successfully!")
            return insights
            
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            return {}

def save_graphs_to_files(output_dir):
    """
    Save all currently open matplotlib figures to files
    
    Args:
        output_dir (Path): Directory to save the graphs
    """
    try:
        output_dir.mkdir(exist_ok=True)
        
        # Get all current figures
        figs = [plt.figure(i) for i in plt.get_fignums()]
        
        if not figs:
            print("No graphs to save.")
            return
        
        graph_names = [
            'timeline_activity.png',
            'temporal_patterns.png', 
            'spike_detection.png',
            'comprehensive_temporal.png'
        ]
        
        for i, fig in enumerate(figs):
            if i < len(graph_names):
                filename = output_dir / graph_names[i]
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved graph: {filename}")
        
        print(f"All graphs saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error saving graphs: {str(e)}")

def run_complete_analysis(config):
    """
    Run complete analysis pipeline with line graph generation
    
    Args:
        config (dict): Configuration dictionary with analysis parameters
    """
    try:
        print("STARTING COMPLETE SOCIAL MEDIA ANALYSIS PIPELINE")
        print("=" * 60)
        
        required_cols = config['text_columns'] + [config['date_column']] + config['virality_columns']
        df = load_and_validate_data(config['input_file'], required_cols)
        
        themes = config.get('custom_themes') 
        theme_method = config.get('theme_method', 'hybrid')
        
        print(f"\nUsing themes: {list(themes.keys())}")
        print(f"Theme assignment method: {theme_method}")
        
        analyzer = SocialPostAnalyzer(
            text_columns=config['text_columns'],
            date_column=config['date_column'],
            virality_columns=config['virality_columns'],
            custom_themes=themes,
            theme_method=theme_method
        )
        
        analyzed_df = analyzer.analyze_posts(df)
        
        if analyzed_df.empty:
            raise ValueError("Analysis produced no data. Check input and column names.")
        
        # Run comprehensive temporal analysis with graphs
        print("\nRUNNING COMPREHENSIVE TEMPORAL ANALYSIS")
        print("-" * 45)
        temporal_patterns, spikes = analyzer.run_full_temporal_analysis(analyzed_df)
        
        # Create additional custom timeline if needed
        print("\nCREATING CUSTOM TIMELINE")
        print("-" * 30)
        analyzer.create_timeline_graph(analyzed_df, "Social Media Activity Timeline")
        
        # Generate insights
        insights = analyzer.generate_insights(analyzed_df)
        
        # Create output directory and save results
        config['output_dir'].mkdir(exist_ok=True)
        
        # Clean up unwanted columns before saving
        columns_to_remove = ['author', 'is_self', 'media']
        cleaned_df = analyzed_df.copy()
        
        for col in columns_to_remove:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.drop(columns=[col])
                print(f"Removed column: {col}")
        
        cleaned_df.to_excel(config['output_dir'] / 'enhanced_analysis.xlsx', index=False)
        print(f"Analysis data saved to: {config['output_dir'] / 'enhanced_analysis.xlsx'}")
        print(f"Cleaned dataset contains {len(cleaned_df.columns)} columns")

        print("\nSAVING RESULTS")
        print("-" * 20)
        save_graphs_to_files(config['output_dir'])
        
        # Save insights to JSON
        import json
        with open(config['output_dir'] / 'insights.json', 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            insights_json = json.loads(json.dumps(insights, default=str))
            json.dump(insights_json, f, indent=2, default=str)
        print(f"Insights saved to: {config['output_dir'] / 'insights.json'}")
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✓ Processed {len(analyzed_df)} posts")
        print(f"✓ Found {len(spikes)} temporal spikes")
        print(f"✓ Results saved to: {config['output_dir']}")
        print("✓ Generated files:")
        print("  - enhanced_analysis.xlsx (main results)")
        print("  - insights.json (summary insights)")
        print("  - timeline_activity.png")
        print("  - temporal_patterns.png")
        print("  - spike_detection.png")
        print("  - comprehensive_temporal.png")
        
        # Print key insights
        if insights:
            print(f"\n📊 KEY INSIGHTS:")
            if 'sentiment' in insights:
                print(f"  - Average sentiment: {insights['sentiment']['average_sentiment']:.3f}")
                print(f"  - Positive posts: {insights['sentiment']['positive_posts']}")
                print(f"  - Negative posts: {insights['sentiment']['negative_posts']}")
            
            if 'themes' in insights:
                print(f"  - Total themes found: {insights['themes']['total_themes']}")
                print(f"  - Most common theme: {insights['themes']['most_common_theme']}")
                print("  - Theme distribution:")
                for theme, count in list(insights['themes']['theme_distribution'].items())[:5]:
                    print(f"    • {theme}: {count}")
            
            if 'virality' in insights:
                print(f"  - Average virality: {insights['virality'].get('mean_virality', 0):.2f}")
                print(f"  - Viral posts: {insights['virality'].get('viral_posts_count', 0)}")
            
            if 'temporal' in insights and 'posting_frequency' in insights['temporal']:
                freq = insights['temporal']['posting_frequency']
                print(f"  - Posts per day: {freq.get('posts_per_day', 0):.1f}")
                print(f"  - Busiest hour: {freq.get('busiest_hour', 'Unknown')}")
                print(f"  - Busiest day: {freq.get('busiest_day', 'Unknown')}")
        
        # Keep graphs open for viewing (optional)
        if config.get('show_graphs', True):
            print(f"\n📈 Graphs are displayed. Close them to continue or set 'show_graphs': False in config.")
        
    except Exception as e:
        print(f"Fatal error in analysis pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example: Define your custom themes for your specific use case
    my_custom_themes = {
        'masculinity': ['men', 'male', 'masculinity', 'masculine', 'man', 'father', 'patriarchy', 'boy', 'boys'],
        'politics_events': ['government', 'politics', 'trump', 'biden', 'maga'],
        'politics_culture': ['left', 'right', 'liberal', 'conservative', 'lefty', 'racism', 'woke', 'sexism'],
        'peterson': ['peterson', 'jbp', 'jordan'],
        'humanities_philosophy': ['philosophy', 'religion', 'christianity'],
        'humanities_psychology':['psychology', 'personality', 'behavior'],
        'humanities_other': ['art', 'music', 'history', 'book', 'reading', 'author', 'authoring'],
        'women': ['woman', 'women', 'girls', 'girl', 'lady'],
        'lgbtq': ['lgbtq', 'lgbt', 'trans', 'transgender', 'gay', 'lesbian', 'nonbinary', 'bisexual'], 
        'self-help': ['help', 'helped', 'helps', 'confidence', 'strong', 'strength', 'gym', 'helping', 'self-help', 'self esteem', 'esteem', 'confident'],
    }
    
    # Configuration
    config = {
        'text_columns': ['title', 'selftext'],
        'date_column': 'date',
        'virality_columns': ['score', 'num_comments'],
        'input_file': Path('peterson_2023-2024.xlsx'),
        'output_dir': Path('results_full_2023-2024'),
        'custom_themes': my_custom_themes,
        'theme_method': 'hybrid',  # choose: 'keyword', 'semantic', or 'hybrid'
        'show_graphs': True  # Set to False to save graphs without displaying them
    }
    
    # Run complete analysis
    run_complete_analysis(config)