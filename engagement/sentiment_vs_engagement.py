import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('engagement_analysis.json', 'r') as f:
    data = json.load(f)

# Extract theme performance data
themes = data['theme_performance']

# Map theme names to custom titles
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

# Prepare data for plotting
theme_names = [theme_mapping.get(theme['theme'], theme['theme']) for theme in themes]
sentiment_scores = [theme['mean_sentiment'] for theme in themes]
upvotes = [theme['mean_upvotes'] for theme in themes]
total_posts = [theme['total_posts'] for theme in themes]

# Create the bubble chart
fig, ax = plt.subplots(figsize=(12, 10))

# Create colors - use a single color scheme
colors = '#0051ba'

# Create bubble sizes (scale them for better visualization)
bubble_sizes = [posts/100 for posts in total_posts]  # Scale down for reasonable bubble sizes

# Create the scatter plot
scatter = ax.scatter(sentiment_scores, upvotes, s=bubble_sizes, 
                    c=colors, alpha=0.6, edgecolors='black', linewidth=1)

# Add labels for each bubble
for i, theme in enumerate(theme_names):
    ax.annotate(theme, (sentiment_scores[i], upvotes[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9, 
                fontweight='bold', ha='left')

# Customize the plot
ax.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Upvotes', fontsize=12, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3)

# Improve layout and display
plt.tight_layout()
plt.show()

# Print summary statistics
print("Theme Performance Summary:")
print("-" * 40)
for i, theme in enumerate(themes):
    print(f"{theme_names[i]:<15} | Posts: {theme['total_posts']:>6,} | "
          f"Sentiment: {theme['mean_sentiment']:>6.3f} | "
          f"Upvotes: {theme['mean_upvotes']:>6.1f}")