import json
import matplotlib.pyplot as plt
import numpy as np

with open('engagement_analysis.json', 'r') as f:
    data = json.load(f)

themes = data['theme_performance']

# custom titles
theme_mapping = {
    'lgbtq': 'LGBTQ+',
    'women': 'Women',
    'politics': 'Politics',
    'masculinity': 'Masculinity',
    'peterson': 'Peterson',
    'self-help': 'Self-help',
    'humanities': 'Humanities',
    'socialsciences': 'Social sciences'
}

# prepare data
theme_names = [theme_mapping.get(theme['theme'], theme['theme']) for theme in themes]
sentiment_scores = [theme['mean_sentiment'] for theme in themes]
upvotes = [theme['mean_upvotes'] for theme in themes]
total_posts = [theme['total_posts'] for theme in themes]

# bubble chart
fig, ax = plt.subplots(figsize=(12, 10))
colors = '#0051ba'
bubble_sizes = [posts/100 for posts in total_posts]  # scaled down
scatter = ax.scatter(sentiment_scores, upvotes, s=bubble_sizes, 
                    c=colors, alpha=0.6, edgecolors='black', linewidth=1)

for i, theme in enumerate(theme_names):
    ax.annotate(theme, (sentiment_scores[i], upvotes[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9, 
                fontweight='bold', ha='left')

ax.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Upvotes', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# summary stats
print("Theme Performance Summary:")
print("-" * 40)
for i, theme in enumerate(themes):
    print(f"{theme_names[i]:<15} | Posts: {theme['total_posts']:>6,} | "
          f"Sentiment: {theme['mean_sentiment']:>6.3f} | "
          f"Upvotes: {theme['mean_upvotes']:>6.1f}")