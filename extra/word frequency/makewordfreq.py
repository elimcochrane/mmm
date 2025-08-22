import pandas as pd
from collections import Counter
import re
from string import punctuation

def clean_text(text):
    """Preprocess text by removing punctuation and making lowercase"""
    if pd.isna(text):
        return ""
    # Remove punctuation and make lowercase
    text = re.sub(f'[{re.escape(punctuation)}]', '', str(text)).lower()
    return text

def get_word_frequencies(df, text_columns, exclude_words=None):
    """Calculate word frequencies from specified columns"""
    all_words = []

    if exclude_words is None:
        exclude_words = []

    # Convert exclude words to lowercase for case-insensitive matching
    exclude_words = [word.lower() for word in exclude_words]

    for col in text_columns:
        # Combine all text from the column
        text = ' '.join(df[col].apply(clean_text))
        # Split into words, exclude stopwords, and count
        words = [word for word in text.split() if word not in exclude_words]
        all_words.extend(words)

    # Count word occurrences
    word_counts = Counter(all_words)
    # Convert to DataFrame
    freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'frequency'])
    # Sort by frequency (descending)
    freq_df = freq_df.sort_values('frequency', ascending=False)

    return freq_df

# Load your Excel file
input_file = 'peterson_2018-2019.xlsx'  # Change to your file name
df = pd.read_excel(input_file)

# List of words to exclude (add your own here)
exclude_words = [
    'the', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'is', 'for',
    'on', 'was', 'with', 'as', 'at', 'be', 'this', 'are', 'or', 'but',
    'not', 'they', 'their', 'what', 'which', 'have', 'has', 'had', 'you',
    'your', 'i', 'we', 'he', 'she', 'his', 'her', 'them', 'those', 'these',
    'if', 'so', 'about', 'from', 'an', 'do', 'can', 'by', 'its'
]

# Get word frequencies (excluding specified words)
freq_df = get_word_frequencies(df, ['title', 'selftext'], exclude_words)

# Save to new Excel file
output_file = 'word_frequencies_filtered.xlsx'
freq_df.to_excel(output_file, index=False)

print(f"✅ Word frequency analysis complete. Saved to {output_file}")
print(f"Excluded {len(exclude_words)} words from analysis.")
print(f"\nTop 10 words after filtering:")
print(freq_df.head(10))