import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import ssl
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VADERSentimentAnalyzer:
    def __init__(self):
        self._setup_nltk()
        self.analyzer = SentimentIntensityAnalyzer()
        print("VADER initialized successfully")
    
    def _setup_nltk(self):
        """nltk with ssl workaround"""
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        nltk.download('vader_lexicon', quiet=True)
        print("VADER setup complete")
    
    def _clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _combine_text_columns(self, df, text_columns):
        combined_text = []
        for idx, row in df.iterrows():
            text_parts = []
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(str(row[col]).strip())
            
            combined = ' '.join(text_parts)
            combined_text.append(self._clean_text(combined))
        
        return pd.Series(combined_text, index=df.index)
    
    def _analyze_sentiment(self, text):
        if not text or text.strip() == '':
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        
        text_sample = text[:2000]
        return self.analyzer.polarity_scores(text_sample)
    
    def _get_classification(self, compound_score):
        if compound_score >= 0.05:
            return 'POSITIVE'
        elif compound_score <= -0.05:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    def analyze_dataframe(self, df, text_columns, output_file=None):
        print(f"Starting VADER sentiment analysis on {len(df)} rows...")
        print(f"Analyzing text from columns: {text_columns}")
        
        # validate columns exist
        missing_cols = [col for col in text_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataframe: {missing_cols}")
        
        # create working copy
        result_df = df.copy()
        
        # combine text columns
        print("Combining and cleaning text...")
        combined_text = self._combine_text_columns(df, text_columns)
        
        # VADER Analysis
        print("Running VADER sentiment analysis...")
        sentiment_results = []
        for i, text in enumerate(combined_text):
            sentiment_score = self._analyze_sentiment(text)
            sentiment_results.append(sentiment_score)
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(combined_text)} texts")
        
        # vader -> dataframe + classification -> results
        sentiment_df = pd.DataFrame(sentiment_results)
        sentiment_df.columns = [f'sentiment_{col}' for col in sentiment_df.columns]
        
        sentiment_df['sentiment_classification'] = sentiment_df['sentiment_compound'].apply(self._get_classification)
        
        result_df = pd.concat([result_df, sentiment_df], axis=1)
        
        result_df['text_analyzed'] = combined_text
        
        print(f"Sentiment analysis complete! Added columns:")
        new_columns = [col for col in result_df.columns if col not in df.columns]
        for col in new_columns:
            print(f"  - {col}")
        
        # save results
        if output_file:
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.xlsx':
                result_df.to_excel(output_path, index=False)
            else:
                result_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        
        return result_df
    
    def get_summary_stats(self, df):
        if 'sentiment_compound' not in df.columns:
            print("No sentiment analysis results found in dataframe")
            return {}
        
        stats = {
            'total_posts': len(df),
            'mean_compound': df['sentiment_compound'].mean(),
            'std_compound': df['sentiment_compound'].std(),
            'min_compound': df['sentiment_compound'].min(),
            'max_compound': df['sentiment_compound'].max(),
            'positive_count': (df['sentiment_classification'] == 'POSITIVE').sum(),
            'negative_count': (df['sentiment_classification'] == 'NEGATIVE').sum(),
            'neutral_count': (df['sentiment_classification'] == 'NEUTRAL').sum(),
        }
        
        stats['positive_percentage'] = (stats['positive_count'] / stats['total_posts']) * 100
        stats['negative_percentage'] = (stats['negative_count'] / stats['total_posts']) * 100
        stats['neutral_percentage'] = (stats['neutral_count'] / stats['total_posts']) * 100
        
        stats['mean_positive'] = df['sentiment_pos'].mean()
        stats['mean_negative'] = df['sentiment_neg'].mean()
        stats['mean_neutral'] = df['sentiment_neu'].mean()
        
        return stats
    
    def create_sentiment_timeline(self, df, date_column='date', output_file=None):
        if 'sentiment_compound' not in df.columns:
            print("No sentiment analysis results found. Run analyze_dataframe() first.")
            return
        
        if date_column not in df.columns:
            print(f"Date column '{date_column}' not found in dataframe.")
            return
        
        print("Creating sentiment timeline graph...")
        
        # date formatting
        df_plot = df.copy()
        try:
            df_plot[date_column] = pd.to_datetime(df_plot[date_column], errors='coerce')
            df_plot = df_plot.dropna(subset=[date_column])
            
            if len(df_plot) == 0:
                print("No valid dates found in the data.")
                return
                
        except Exception as e:
            print(f"Error parsing dates: {e}")
            return
        
        # group by date and calculate average sentiment
        daily_sentiment = df_plot.groupby(df_plot[date_column].dt.date).agg({
            'sentiment_compound': ['mean', 'count']
        }).round(3)
        
        # some formatting
        daily_sentiment.columns = ['avg_sentiment', 'post_count']
        daily_sentiment = daily_sentiment.reset_index()
        daily_sentiment[date_column] = pd.to_datetime(daily_sentiment[date_column])
        
        daily_sentiment = daily_sentiment.sort_values(date_column)
        
        print(f"Creating timeline from {daily_sentiment[date_column].min().strftime('%m/%d/%Y')} to {daily_sentiment[date_column].max().strftime('%m/%d/%Y')}")
        
        # visualization
        plt.figure(figsize=(14, 8))
        
        plt.plot(daily_sentiment[date_column], daily_sentiment['avg_sentiment'], 
                color='steelblue', linewidth=2, alpha=0.8, label='Daily Average Sentiment')
        
        if len(daily_sentiment) > 30:
            rolling_avg = daily_sentiment.set_index(date_column)['avg_sentiment'].rolling(window=30, center=True).mean()
            plt.plot(rolling_avg.index, rolling_avg.values, 
                    color='red', linewidth=3, alpha=0.7, label='30-Day Trend')
        
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Positive Threshold')
        plt.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Negative Threshold')
        
        plt.title('Average Sentiment Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Sentiment Score', fontsize=12)
        plt.ylim(-1, 1)
        
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
        plt.xticks(rotation=45)
        
        plt.grid(True, alpha=0.3)
        
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        
        # save if output file specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Timeline graph saved to: {output_file}")
        
        plt.show()
        
        # summary stats
        print(f"\nTimeline Statistics:")
        print(f"  Date range: {daily_sentiment[date_column].min().strftime('%m/%d/%Y')} to {daily_sentiment[date_column].max().strftime('%m/%d/%Y')}")
        print(f"  Total days with posts: {len(daily_sentiment)}")
        print(f"  Average posts per day: {daily_sentiment['post_count'].mean():.1f}")
        print(f"  Overall average sentiment: {daily_sentiment['avg_sentiment'].mean():.3f}")
        print(f"  Most positive day: {daily_sentiment.loc[daily_sentiment['avg_sentiment'].idxmax(), date_column].strftime('%m/%d/%Y')} (score: {daily_sentiment['avg_sentiment'].max():.3f})")
        print(f"  Most negative day: {daily_sentiment.loc[daily_sentiment['avg_sentiment'].idxmin(), date_column].strftime('%m/%d/%Y')} (score: {daily_sentiment['avg_sentiment'].min():.3f})")
        
        return daily_sentiment
    
    # find most extreme sentiment scores
    def find_most_extreme(self, df, n=5):
        if 'sentiment_compound' not in df.columns:
            return {}
        
        most_positive = df.nlargest(n, 'sentiment_compound')[['text_analyzed', 'sentiment_compound', 'sentiment_classification']]
        
        most_negative = df.nsmallest(n, 'sentiment_compound')[['text_analyzed', 'sentiment_compound', 'sentiment_classification']]
        
        return {
            'most_positive': most_positive.to_dict('records'),
            'most_negative': most_negative.to_dict('records')
        }

def main():
    # customization
    config = {
        'input_file': 'peterson_2016-2024_cleaned.xlsx',
        'text_columns': ['title', 'selftext'],
        'output_file': 'sentiment_classification_results.xlsx',
        'date_column': 'date',
        'create_timeline': True,
        'timeline_output': 'sentiment_timeline.png',
        'show_examples': True
    }
    
    try:
        print(f"Loading data from: {config['input_file']}")
        df = pd.read_excel(config['input_file'])
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        analyzer = VADERSentimentAnalyzer()
        
        results_df = analyzer.analyze_dataframe(
            df, 
            text_columns=config['text_columns'],
            output_file=config['output_file']
        )
        
        print("\n" + "="*60)
        print("VADER SENTIMENT ANALYSIS SUMMARY")
        print("="*60)
        
        stats = analyzer.get_summary_stats(results_df)
        
        if stats:
            print(f"\nOverall Statistics:")
            print(f"  Total posts analyzed: {stats['total_posts']}")
            print(f"  Mean compound score: {stats['mean_compound']:.3f}")
            print(f"  Standard deviation: {stats['std_compound']:.3f}")
            print(f"  Score range: {stats['min_compound']:.3f} to {stats['max_compound']:.3f}")
            
            print(f"\nSentiment Distribution:")
            print(f"  Positive: {stats['positive_count']} posts ({stats['positive_percentage']:.1f}%)")
            print(f"  Negative: {stats['negative_count']} posts ({stats['negative_percentage']:.1f}%)")
            print(f"  Neutral: {stats['neutral_count']} posts ({stats['neutral_percentage']:.1f}%)")
            
            print(f"\nAverage Component Scores:")
            print(f"  Positive: {stats['mean_positive']:.3f}")
            print(f"  Negative: {stats['mean_negative']:.3f}")
            print(f"  Neutral: {stats['mean_neutral']:.3f}")
        
        if config.get('show_examples', False):
            print(f"\n" + "="*60)
            print("EXAMPLE POSTS")
            print("="*60)
            
            examples = analyzer.find_most_extreme(results_df, n=3)
            
            if 'most_positive' in examples:
                print(f"\nMost Positive Posts:")
                for i, post in enumerate(examples['most_positive'], 1):
                    text_preview = post['text_analyzed'][:100] + "..." if len(post['text_analyzed']) > 100 else post['text_analyzed']
                    print(f"  {i}. Score: {post['sentiment_compound']:.3f}")
                    print(f"     Text: {text_preview}")
                    print()
            
            if 'most_negative' in examples:
                print(f"Most Negative Posts:")
                for i, post in enumerate(examples['most_negative'], 1):
                    text_preview = post['text_analyzed'][:100] + "..." if len(post['text_analyzed']) > 100 else post['text_analyzed']
                    print(f"  {i}. Score: {post['sentiment_compound']:.3f}")
                    print(f"     Text: {text_preview}")
                    print()
        
        print(f"Results saved to: {config['output_file']}")
        print(f"\nNew columns added:")
        print(f"  - sentiment_neg: Negative sentiment score (0-1)")
        print(f"  - sentiment_neu: Neutral sentiment score (0-1)")  
        print(f"  - sentiment_pos: Positive sentiment score (0-1)")
        print(f"  - sentiment_compound: Overall sentiment (-1 to +1)")
        print(f"  - sentiment_classification: POSITIVE/NEGATIVE/NEUTRAL")
        print(f"  - text_analyzed: Combined and cleaned text that was analyzed")
        
    except FileNotFoundError:
        print(f"Could not find file '{config['input_file']}'")
        print("Please check the file path and make sure the file exists.")
    except KeyError as e:
        print(f"Column not found in Excel file: {e}")
        print("Please check your column names in the config.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()