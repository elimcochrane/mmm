"""
Data Preprocessing Module
Handles data cleaning, text preprocessing, and validation
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_validate_data(file_path, required_columns=None):
    """
    Load and validate data from Excel file
    
    Args:
        file_path (str): Path to the Excel file
        required_columns (list): List of required column names
        
    Returns:
        pd.DataFrame: Validated DataFrame
    """
    try:
        print(f"Loading Excel file: {file_path}")
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"Loaded {len(df)} rows")
        print("Available columns:", df.columns.tolist())
        
        # Check for required columns but don't drop rows
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                # Create missing columns with empty values instead of raising error
                for col in missing_cols:
                    df[col] = ''
                    print(f"Created empty column: {col}")
        
        # Print basic data info
        print(f"\nData overview:")
        print(f"- Total rows: {len(df)}")
        if 'title' in df.columns:
            print(f"- Rows with title: {df['title'].notnull().sum()}")
        if 'selftext' in df.columns:
            print(f"- Rows with selftext: {df['selftext'].notnull().sum()}")
        if 'date' in df.columns:
            print(f"- Rows with date: {df['date'].notnull().sum()}")
            # Only show date range if there are valid dates
            valid_dates = df['date'].dropna()
            if len(valid_dates) > 0:
                print(f"- Date range: {valid_dates.min()} to {valid_dates.max()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def clean_text_data(df, text_columns=['title', 'selftext']):
    """
    Clean and preprocess text data
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_columns (list): List of text column names
        
    Returns:
        pd.DataFrame: DataFrame with cleaned text
    """
    print("Cleaning text data...")
    df = df.copy()
    
    # Clean individual text columns
    for col in text_columns:
        if col in df.columns:
            # More robust cleaning that preserves all rows
            df[f'{col}_clean'] = df[col].astype(str).fillna('')
            df[f'{col}_clean'] = df[f'{col}_clean'].replace(['nan', 'None', 'NaN', '<NA>'], '')
            df[f'{col}_clean'] = df[f'{col}_clean'].str.strip()
        else:
            # Create empty column if it doesn't exist
            df[f'{col}_clean'] = ''
            print(f"Warning: Column '{col}' not found, created empty '{col}_clean' column")
    
    # Combine text columns if multiple exist
    text_clean_columns = [f'{col}_clean' for col in text_columns if f'{col}_clean' in df.columns]
    
    if len(text_clean_columns) > 1:
        # More robust text combination
        def combine_text(row):
            texts = []
            for col in text_clean_columns:
                if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
                    texts.append(str(row[col]).strip())
            return ' '.join(texts) if texts else ''
        
        df['text_clean'] = df.apply(combine_text, axis=1)
    elif len(text_clean_columns) == 1:
        df['text_clean'] = df[text_clean_columns[0]]
    else:
        # Create empty text_clean column if no text columns exist
        df['text_clean'] = ''
    
    # Additional cleaning - ensure no nulls
    df['text_clean'] = df['text_clean'].fillna('').astype(str)
    df['text_clean'] = df['text_clean'].replace(['nan', 'None', 'NaN', '<NA>'], '')
    
    # Handle very short texts - but don't drop rows
    short_text_mask = df['text_clean'].str.len() < 10
    if short_text_mask.any():
        print(f"Posts with very short text (enhancing with title): {short_text_mask.sum()}")
        # Try to use title for short posts, or create meaningful placeholder
        for idx in df[short_text_mask].index:
            current_text = str(df.loc[idx, 'text_clean']).strip()
            title_text = str(df.loc[idx, 'title_clean']).strip() if 'title_clean' in df.columns else ''
            
            if title_text and title_text != current_text:
                df.loc[idx, 'text_clean'] = f"{title_text} {current_text}".strip()
            elif not current_text:
                df.loc[idx, 'text_clean'] = "No content available"
    
    print(f"Text cleaning completed. Dataset size maintained: {len(df)} rows")
    return df

def validate_and_prepare_data(df, text_columns, date_column, virality_columns):
    """
    Validate and prepare data for analysis
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_columns (list): List of text column names
        date_column (str): Name of date column
        virality_columns (list): List of virality column names
        
    Returns:
        pd.DataFrame: Prepared DataFrame
    """
    print(f"Starting data validation with {len(df)} rows")
    initial_rows = len(df)
    
    # Clean text data
    df = clean_text_data(df, text_columns)
    
    # Handle dates - but don't drop rows with invalid dates
    if date_column in df.columns:
        print(f"Processing date column: {date_column}")
        original_dates = df[date_column].copy()
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        valid_dates_count = df[date_column].notnull().sum()
        print(f"Rows with valid dates: {valid_dates_count}")
        
        # Handle invalid dates - use imputation instead of dropping
        invalid_dates = df[date_column].isnull()
        if invalid_dates.any():
            print(f"Posts with invalid dates (using imputation): {invalid_dates.sum()}")
            
            # Try multiple imputation strategies
            if valid_dates_count > 0:
                # Use median date if available
                median_date = df[date_column].median()
                df.loc[invalid_dates, date_column] = median_date
                print(f"Used median date for imputation: {median_date}")
            else:
                # Use a reasonable default if no valid dates exist
                default_date = pd.Timestamp('2020-01-01')  # More recent default
                df.loc[invalid_dates, date_column] = default_date
                print(f"Used default date for imputation: {default_date}")
    
    # Handle virality columns - ensure they exist and have valid values
    if virality_columns:
        for col in virality_columns:
            if col not in df.columns:
                df[col] = 0  # Create with default value
                print(f"Created missing virality column '{col}' with default value 0")
            else:
                # Fill missing virality values with 0 instead of dropping
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna(0)
                    print(f"Filled {missing_count} missing values in '{col}' with 0")
                
                # Ensure numeric type
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Validate minimum data requirements - but be more flexible
    final_rows = len(df)
    if final_rows != initial_rows:
        print(f"WARNING: Row count changed from {initial_rows} to {final_rows}")
    
    if final_rows < 10:  # Lowered threshold
        print(f"Warning: Very small dataset with only {final_rows} rows. Consider reviewing data quality.")
    elif final_rows < 50:
        print(f"Warning: Small dataset with {final_rows} rows. Results may be less reliable.")
    
    print(f"Data validation completed. Final dataset: {final_rows} rows (preserved {final_rows/initial_rows*100:.1f}% of original data)")
    return df