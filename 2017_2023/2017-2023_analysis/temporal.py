"""
Temporal Analysis Module
Handles time-based analysis including spike detection and temporal patterns with line graphs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def detect_temporal_spikes(df, date_column='date', plot=True):
    """
    Detect temporal spikes in posting activity
    
    Args:
        df (pd.DataFrame): DataFrame with temporal data
        date_column (str): Name of the date column
        plot (bool): Whether to create a line graph
        
    Returns:
        pd.Series: Series containing spike data
    """
    try:
        if df.empty:
            return pd.Series()
            
        print("Detecting temporal spikes...")
        
        # Create time series of post counts
        time_series = df.set_index(date_column).resample('1H').size()
        
        # Calculate rolling mean and standard deviation
        rolling_mean = time_series.rolling(24, min_periods=1).mean()
        std_dev = rolling_mean.std()
        
        # Handle case where std_dev is 0
        if std_dev == 0:
            threshold = rolling_mean.mean() + 1
        else:
            threshold = rolling_mean + 2 * std_dev
            
        # Identify spikes
        spikes = time_series[time_series > threshold]
        
        # Create line graph if requested
        if plot and len(time_series) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(time_series.index, time_series.values, label='Post Count', alpha=0.7)
            plt.plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean (24h)', color='orange')
            plt.plot(threshold.index, threshold.values, label='Spike Threshold', color='red', linestyle='--')
            
            # Highlight spikes
            if len(spikes) > 0:
                plt.scatter(spikes.index, spikes.values, color='red', s=50, label='Spikes', zorder=5)
            
            plt.title('Temporal Spike Detection')
            plt.xlabel('Date')
            plt.ylabel('Post Count per Hour')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        print(f"Found {len(spikes)} temporal spikes")
        return spikes
        
    except Exception as e:
        print(f"Error in spike detection: {str(e)}")
        return pd.Series()

def analyze_temporal_patterns(df, date_column='date', plot=True):
    """
    Analyze temporal patterns in the data
    
    Args:
        df (pd.DataFrame): DataFrame with temporal data
        date_column (str): Name of the date column
        plot (bool): Whether to create line graphs
        
    Returns:
        dict: Dictionary containing temporal analysis results
    """
    try:
        if df.empty:
            return {}
            
        print("Analyzing temporal patterns...")
        
        # Convert to datetime if needed
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Basic temporal statistics
        temporal_stats = {
            'date_range': {
                'start': df[date_column].min(),
                'end': df[date_column].max(),
                'span_days': (df[date_column].max() - df[date_column].min()).days
            },
            'posting_frequency': {
                'total_posts': len(df),
                'posts_per_day': len(df) / max(1, (df[date_column].max() - df[date_column].min()).days),
                'busiest_hour': df[date_column].dt.hour.mode().iloc[0] if not df[date_column].dt.hour.mode().empty else 0,
                'busiest_day': df[date_column].dt.day_name().mode().iloc[0] if not df[date_column].dt.day_name().mode().empty else 'Unknown'
            }
        }
        
        # Hourly patterns
        hourly_counts = df.groupby(df[date_column].dt.hour).size()
        temporal_stats['hourly_patterns'] = hourly_counts.to_dict()
        
        # Daily patterns
        daily_counts = df.groupby(df[date_column].dt.day_name()).size()
        temporal_stats['daily_patterns'] = daily_counts.to_dict()
        
        # Create line graphs if requested
        if plot:
            # Daily posting activity over time
            daily_activity = df.groupby(df[date_column].dt.date).size()
            
            plt.figure(figsize=(15, 10))
            
            # Subplot 1: Daily activity over time
            plt.subplot(2, 2, 1)
            plt.plot(daily_activity.index, daily_activity.values, marker='o', markersize=3)
            plt.title('Daily Posting Activity Over Time')
            plt.xlabel('Date')
            plt.ylabel('Number of Posts')
            plt.xticks(rotation=45)
            
            # Subplot 2: Hourly patterns
            plt.subplot(2, 2, 2)
            hours = list(range(24))
            hourly_values = [hourly_counts.get(h, 0) for h in hours]
            plt.plot(hours, hourly_values, marker='o')
            plt.title('Posting Activity by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('Number of Posts')
            plt.xticks(range(0, 24, 2))
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Weekly patterns
            plt.subplot(2, 2, 3)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_values = [daily_counts.get(day, 0) for day in day_order]
            plt.plot(day_order, weekly_values, marker='o')
            plt.title('Posting Activity by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Number of Posts')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Subplot 4: Monthly trend
            plt.subplot(2, 2, 4)
            monthly_activity = df.groupby(df[date_column].dt.to_period('M')).size()
            plt.plot([str(period) for period in monthly_activity.index], monthly_activity.values, marker='o')
            plt.title('Monthly Posting Trends')
            plt.xlabel('Month')
            plt.ylabel('Number of Posts')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return temporal_stats
        
    except Exception as e:
        print(f"Error in temporal pattern analysis: {str(e)}")
        return {}

def create_simple_timeline_graph(df, date_column='date', title='Timeline Activity'):
    """
    Create a simple line graph showing activity over time
    
    Args:
        df (pd.DataFrame): DataFrame with temporal data
        date_column (str): Name of the date column
        title (str): Title for the graph
    """
    try:
        print("Creating timeline graph...")
        
        # Group by date and count posts
        daily_counts = df.groupby(df[date_column].dt.date).size()
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_counts.index, daily_counts.values, linewidth=2, color='steelblue')
        plt.fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='steelblue')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Posts', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add some basic statistics as text
        avg_posts = daily_counts.mean()
        max_posts = daily_counts.max()
        plt.axhline(y=avg_posts, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_posts:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Timeline created: {len(daily_counts)} days of data")
        print(f"Average posts per day: {avg_posts:.2f}")
        print(f"Peak activity: {max_posts} posts")
        
    except Exception as e:
        print(f"Error creating timeline graph: {str(e)}")

def prepare_temporal_data(df, date_column='date'):
    """
    Prepare data for temporal analysis
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_column (str): Name of the date column
        
    Returns:
        pd.DataFrame: DataFrame with properly formatted temporal data
    """
    print("Preparing temporal data...")
    
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    print(f"Rows with valid dates: {df[date_column].notnull().sum()}")
    
    # Handle invalid dates
    invalid_dates = df[date_column].isnull()
    if invalid_dates.any():
        print(f"Posts with invalid dates (using default): {invalid_dates.sum()}")
        # Use the most common date or a default date
        default_date = df[date_column].mode().iloc[0] if not df[date_column].mode().empty else pd.Timestamp('2018-01-01')
        df[date_column] = df[date_column].fillna(default_date)
    
    return df

# Example usage function
def run_temporal_analysis_with_graphs(df, date_column='date'):
    """
    Run complete temporal analysis with line graphs
    
    Args:
        df (pd.DataFrame): DataFrame with temporal data
        date_column (str): Name of the date column
    """
    # Prepare data
    df_clean = prepare_temporal_data(df.copy(), date_column)
    
    # Create simple timeline
    create_simple_timeline_graph(df_clean, date_column)
    
    # Analyze patterns with graphs
    patterns = analyze_temporal_patterns(df_clean, date_column, plot=True)
    
    # Detect spikes with graph
    spikes = detect_temporal_spikes(df_clean, date_column, plot=True)
    
    return patterns, spikes