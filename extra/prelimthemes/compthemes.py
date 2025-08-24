import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import wordnet
import nltk
import re
import os
nltk.download('wordnet')

def get_synonyms(keyword):
    synonyms = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return list(synonyms)

base_keyword_groups = {
    'masculinity': ['men', 'male', 'masculinity', 'masculine', 'man', 'father', 'patriarchy', 'boy', 'boys'],
    'politics': ['government', 'politics', 'left', 'right', 'liberal', 'conservative', 'lefty', 'trump', 'maga', 'fascism', 'democracy'],
    'humanities': ['psychology', 'personality', 'philosophy', 'religion', 'christianity', 'art', 'history', 'book', 'reading', 'history'],
    'women': ['woman', 'women', 'girls', 'girl'],
    'self-help': ['help', 'helped', 'helps', 'confidence', 'strong', 'strength', 'gym', 'helping', 'self-help', 'self esteem', 'esteem', 'confident'],
}

keyword_groups = {}
for theme, keywords in base_keyword_groups.items():
    expanded_keywords = []
    for word in keywords:
        expanded_keywords.append(word)
        expanded_keywords.extend(get_synonyms(word))
    keyword_groups[theme] = sorted(list(set(expanded_keywords)))

# save expanded keywords
keywords_df = pd.DataFrame([(k, ', '.join(v)) for k, v in keyword_groups.items()], 
                         columns=['Theme', 'Expanded Keywords'])
keywords_output_filename = 'expanded_keyword_lists.xlsx'
keywords_df.to_excel(keywords_output_filename, index=False)
print(f"\nExpanded keyword lists saved to {os.path.abspath(keywords_output_filename)}")

# get data + cleaning
print("Please enter the path to your input file:")
filename = input("File path: ").strip('"')

df = pd.read_excel(filename)
df['text_combined'] = df['title'].str.lower() + ' ' + df['selftext'].str.lower()

df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce')
df['score'] = df['score'].fillna(0)
df['num_comments'] = df['num_comments'].fillna(0)

total_posts = len(df)

# results
results = []
for group_name, keywords in keyword_groups.items():
    pattern = '|'.join([re.escape(k) for k in keywords])
    matches = df[df['text_combined'].str.contains(pattern, na=False, regex=True)]
    num_posts = len(matches)

    results.append({
        'Theme': group_name,
        'Keyword Count': len(keywords),
        'Number of Posts': num_posts,
        'Percent of Total': f"{num_posts/total_posts*100:.1f}%",
        'Avg Upvotes': matches['score'].mean(),
        'Avg Comments': matches['num_comments'].mean(),
        'Sample Keywords': ', '.join(keywords[:5]) + '...'
    })

results_df = pd.DataFrame(results)
print("\nFull Analysis Results (with synonym expansion):")
print(results_df.to_string())

output_filename = 'keyword_analysis_with_synonyms.xlsx'
results_df.to_excel(output_filename, index=False)
print(f"\nResults saved to {os.path.abspath(output_filename)}")

# visualizations
blues = plt.cm.Blues(np.linspace(0.3, 1, len(keyword_groups)))
engagement_colors = [blues[1], blues[-1]] 

print("\nKey Metrics Visualization:")
fig, axes = plt.subplots(1, 3, figsize=(21, 7)) 

# 1 - posts by theme
results_df.plot(x='Theme', y='Number of Posts', kind='bar', ax=axes[0],
               title='Posts per Theme', color=blues, legend=False)
axes[0].tick_params(axis='x', rotation=45)

# 2 - engagement metrics
results_df.plot(x='Theme', y=['Avg Upvotes', 'Avg Comments'], kind='bar', ax=axes[1],
               title='Engagement Metrics', color=engagement_colors, width=0.8)
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(loc='upper right')

# 3 - % of total
results_df['Percent'] = results_df['Percent of Total'].str.replace('%', '').astype(float)
wedges, texts = axes[2].pie(results_df['Percent'],
                          colors=blues,
                          startangle=90,
                          wedgeprops=dict(width=0.4, edgecolor='w'))

legend_labels = [f"{theme} ({percent:.1f}%)" 
                for theme, percent in zip(results_df['Theme'], results_df['Percent'])]

axes[2].set_title('Percentage of Total Posts')
axes[2].legend(wedges, legend_labels,
              title="Themes",
              loc="center left",
              bbox_to_anchor=(1.05, 0.5),
              frameon=False) 

plt.tight_layout(pad=4.0, w_pad=5.0)  

figure_filename = 'visualization_results.png'
plt.savefig(figure_filename, bbox_inches='tight', dpi=300, pad_inches=0.5)
print(f"\nVisualization saved to {os.path.abspath(figure_filename)}")

plt.show()

# show expanded keyword lists
print("\nExpanded Keyword Lists:")
for theme, keywords in keyword_groups.items():
    print(f"\n{theme.upper()} ({len(keywords)} terms):")
    print(', '.join(keywords[:10]), '...' if len(keywords) > 10 else '')