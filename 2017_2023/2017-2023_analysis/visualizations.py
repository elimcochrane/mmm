"""
Visualization Module
Handles creation of plots and charts for analysis results
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_sentiment_toxicity_plot(df):
    """
    Create sentiment vs toxicity distribution plot
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment and toxicity data
        
    Returns:
        plotly.graph_objects.Figure: Sentiment vs toxicity plot
    """
    try:
        if df.empty or 'sentiment_compound' not in df.columns:
            return go.Figure()
        
        print("Creating sentiment vs toxicity plot...")
        
        # Determine color column
        color_col = 'toxicity_toxicity' if 'toxicity_toxicity' in df.columns else None
        
        fig = px.histogram(
            df,
            x='sentiment_compound',
            color=color_col,
            marginal='box',
            title='Sentiment vs Toxicity Distribution',
            template='plotly_white',
            labels={
                'sentiment_compound': 'Sentiment Score',
                'toxicity_toxicity': 'Toxicity Score'
            }
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating sentiment-toxicity plot: {str(e)}")
        return go.Figure()

def create_temporal_analysis_plot(df, spikes, date_column='date'):
    """
    Create temporal analysis plot showing posting activity and virality over time
    
    Args:
        df (pd.DataFrame): DataFrame with temporal data
        spikes (pd.Series): Series containing spike data
        date_column (str): Name of date column
        
    Returns:
        plotly.graph_objects.Figure: Temporal analysis plot
    """
    try:
        if df.empty:
            return go.Figure()
        
        print("Creating temporal analysis plot...")
        
        # Create time series data
        time_df = df.set_index(date_column).resample('1H').agg({
            'text_clean': 'count',
            'virality_score': 'mean'
        }).dropna()
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add post volume trace
        fig.add_trace(
            go.Scatter(
                x=time_df.index, 
                y=time_df['text_clean'], 
                name='Post Volume',
                line=dict(color='blue')
            )
        )
        
        # Add spikes if they exist
        if not spikes.empty:
            fig.add_trace(
                go.Scatter(
                    x=spikes.index, 
                    y=spikes, 
                    mode='markers', 
                    name='Spikes', 
                    marker=dict(size=10, color='red')
                )
            )
        
        # Add virality trace on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=time_df.index, 
                y=time_df['virality_score'], 
                name='Virality',
                line=dict(color='green')
            ), 
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title='Posting Activity and Virality Over Time',
            template='plotly_white'
        )
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Post Count", secondary_y=False)
        fig.update_yaxes(title_text="Average Virality Score", secondary_y=True)
        
        return fig
        
    except Exception as e:
        print(f"Error creating temporal analysis plot: {str(e)}")
        return go.Figure()

def create_topic_analysis_plot(df):
    """
    Create topic analysis plot showing virality vs toxicity by topic
    
    Args:
        df (pd.DataFrame): DataFrame with topic data
        
    Returns:
        plotly.graph_objects.Figure: Topic analysis plot
    """
    try:
        if df.empty or 'topic_name' not in df.columns:
            return go.Figure()
        
        print("Creating topic analysis plot...")
        
        # Create topic statistics
        topic_stats = df.groupby('topic_name').agg({
            'virality_score': 'mean',
            'toxicity_toxicity': 'mean',
            'text_clean': 'count'
        }).reset_index()
        topic_stats.columns = ['topic_name', 'avg_virality', 'avg_toxicity', 'post_count']
        
        # Create scatter plot
        fig = px.scatter(
            topic_stats,
            x='avg_virality',
            y='avg_toxicity',
            size='post_count',
            hover_name='topic_name',
            color='avg_virality',
            title='Topic Analysis: Virality vs Toxicity',
            labels={
                'avg_virality': 'Average Virality Score', 
                'avg_toxicity': 'Average Toxicity Score',
                'post_count': 'Post Count'
            },
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating topic analysis plot: {str(e)}")
        return go.Figure()

def create_comprehensive_dashboard(df, spikes, date_column='date'):
    """
    Create a comprehensive dashboard with multiple visualizations
    
    Args:
        df (pd.DataFrame): DataFrame with analysis results
        spikes (pd.Series): Series containing spike data
        date_column (str): Name of date column
        
    Returns:
        tuple: Tuple of (sentiment_fig, temporal_fig, topic_fig)
    """
    try:
        print("Creating comprehensive visualization dashboard...")
        
        # Create individual plots
        sentiment_fig = create_sentiment_toxicity_plot(df)
        temporal_fig = create_temporal_analysis_plot(df, spikes, date_column)
        topic_fig = create_topic_analysis_plot(df)
        
        return sentiment_fig, temporal_fig, topic_fig
        
    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")
        return go.Figure(), go.Figure(), go.Figure()

def save_visualizations(fig1, fig2, fig3, output_dir):
    """
    Save visualizations to HTML files
    
    Args:
        fig1, fig2, fig3: Plotly figures
        output_dir: Output directory path
    """
    try:
        print("Saving visualizations...")
        
        fig1.write_html(output_dir / "sentiment_toxicity.html")
        fig2.write_html(output_dir / "temporal_analysis.html") 
        fig3.write_html(output_dir / "topic_network.html")
        
        print(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error saving visualizations: {str(e)}")

def generate_report(df, spikes, date_column='date'):
    """
    Generate complete analysis report with visualizations
    
    Args:
        df (pd.DataFrame): DataFrame with analysis results
        spikes (pd.Series): Series containing spike data
        date_column (str): Name of date column
        
    Returns:
        tuple: Tuple of (sentiment_fig, temporal_fig, topic_fig)
    """
    return create_comprehensive_dashboard(df, spikes, date_column)