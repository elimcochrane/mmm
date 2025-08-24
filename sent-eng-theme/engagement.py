import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('2016-2024_posts.xlsx')
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
df = df[df['date'] >= pd.Timestamp('2017-01-01')]
df = df.dropna(subset=['date'])
df = df.rename(columns={
    'primary_theme': 'theme', 
    'sentiment_compound': 'sentiment_score'
})
df = df[df['theme'] != 'Other']

# ensure numeric engagement metrics
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce')
df = df.dropna(subset=['score', 'num_comments'])

# show column names and basic stats
print("Columns in DataFrame:", df.columns.tolist())
print("Number of themes:", df['theme'].nunique())
print("Total posts:", len(df))
print("Upvotes stats - Min:", df['score'].min(), "Max:", df['score'].max(), "Mean:", df['score'].mean())
print("Comments stats - Min:", df['num_comments'].min(), "Max:", df['num_comments'].max(), "Mean:", df['num_comments'].mean())

# engagement
scaler = MinMaxScaler()
df[['upvotes_normalized', 'comments_normalized']] = scaler.fit_transform(
    df[['score', 'num_comments']]
)
df['post_engagement_score'] = (
    0.6 * df['upvotes_normalized'] + 
    0.4 * df['comments_normalized']  # Slightly favor upvotes as they're more common
)

# find high engagement posts
top_10_percent_threshold = df['post_engagement_score'].quantile(0.9)
df['is_top_10_percent_post'] = df['post_engagement_score'] >= top_10_percent_threshold

high_engagement_posts = df[df['is_top_10_percent_post']]

print(f"\nTop 10% engagement threshold: {top_10_percent_threshold:.3f}")
print(f"Number of high engagement posts: {len(high_engagement_posts)}")

# analyze themes based on their representation in high engagement posts
theme_analysis = df.groupby('theme').agg(
    total_posts=('theme', 'count'),
    high_engagement_posts=('is_top_10_percent_post', 'sum'),
    mean_sentiment=('sentiment_score', 'mean'),
    mean_upvotes=('score', 'mean'),
    mean_comments=('num_comments', 'mean'),
    mean_engagement_score=('post_engagement_score', 'mean'),
    max_engagement_score=('post_engagement_score', 'max')
).reset_index()

# calculate what % of each theme's posts are high engagement
theme_analysis['high_engagement_rate'] = (
    theme_analysis['high_engagement_posts'] / theme_analysis['total_posts']
)

# calculate what %  of all high engagement posts each theme represents
total_high_engagement_posts = len(high_engagement_posts)
theme_analysis['share_of_viral_posts'] = (
    theme_analysis['high_engagement_posts'] / total_high_engagement_posts
)

# Identify themes that are over-represented in high engagement posts
expected_rate = 0.1  # 10% baseline
theme_analysis['overperforming'] = theme_analysis['high_engagement_rate'] > expected_rate * 2  # 2x the baseline

# Sort by high engagement rate
theme_analysis = theme_analysis.sort_values('high_engagement_rate', ascending=False)

# visualizations
# 1 - scatter plot
plt.figure(figsize=(16, 10))
scatter = sns.scatterplot(
    data=theme_analysis,
    x='mean_sentiment',
    y='mean_upvotes',
    size='total_posts',
    hue='high_engagement_rate',
    sizes=(100, 800),
    alpha=0.8
)

plt.colorbar(scatter.collections[0], label='High-Engagement Post Rate')

# label overperforming themes
significant_themes = theme_analysis[
    (theme_analysis['overperforming']) | 
    (theme_analysis['high_engagement_posts'] >= 5)  # At least 5 viral posts
]

for i, row in significant_themes.iterrows():
    plt.annotate(
        f"{row['theme']}\n({row['high_engagement_posts']:.0f} viral posts, {row['high_engagement_rate']:.1%} rate)", 
        (row['mean_sentiment'], row['mean_upvotes']),
        xytext=(10, 5),
        textcoords='offset points',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6)
    )

plt.title('Theme Performance: Sentiment vs Engagement\n(Bubble size = total posts, Color = high-engagement rate)', fontsize=16)
plt.xlabel('Mean Sentiment Score', fontsize=12)
plt.ylabel('Mean Upvotes', fontsize=12)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Neutral Sentiment')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('sentiment_vs_engagement.png', dpi=300, bbox_inches='tight')
plt.close()

# 2 - high engagement post distribution by theme
plt.figure(figsize=(14, 8))
top_themes = theme_analysis.head(10)

bars = plt.bar(range(len(top_themes)), top_themes['high_engagement_rate'], 
               color=plt.cm.viridis(top_themes['high_engagement_rate']))
plt.xlabel('Theme')
plt.ylabel('High-Engagement Post Rate')
plt.title('Top Themes by High-Engagement Post Rate (Top 10% of All Posts)')
plt.xticks(range(len(top_themes)), top_themes['theme'], rotation=45, ha='right')

for i, (bar, rate, count) in enumerate(zip(bars, top_themes['high_engagement_rate'], top_themes['high_engagement_posts'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{rate:.1%}\n({count:.0f} posts)', 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('high_engagement_themes.png', dpi=300, bbox_inches='tight')
plt.close()

# save to json
analysis_data = {
    "summary": {
        "total_posts": len(df),
        "high_engagement_posts": len(high_engagement_posts),
        "engagement_threshold": round(top_10_percent_threshold, 3),
        "themes_analyzed": len(theme_analysis)
    },
    "theme_performance": []
}

for _, row in theme_analysis.iterrows():
    analysis_data["theme_performance"].append({
        "theme": row['theme'],
        "total_posts": int(row['total_posts']),
        "high_engagement_posts": int(row['high_engagement_posts']),
        "high_engagement_rate": round(row['high_engagement_rate'], 3),
        "share_of_all_viral_posts": round(row['share_of_viral_posts'], 3),
        "mean_sentiment": round(row['mean_sentiment'], 3),
        "mean_upvotes": round(row['mean_upvotes'], 1),
        "mean_comments": round(row['mean_comments'], 1),
        "mean_engagement_score": round(row['mean_engagement_score'], 3),
        "max_engagement_score": round(row['max_engagement_score'], 3),
        "is_overperforming": bool(row['overperforming'])
    })

# save the actual high engagement posts for reference
analysis_data["sample_high_engagement_posts"] = []
sample_posts = high_engagement_posts.nlargest(20, 'post_engagement_score')  # Top 20 posts
for _, post in sample_posts.iterrows():
    analysis_data["sample_high_engagement_posts"].append({
        "theme": post['theme'],
        "sentiment_score": round(post['sentiment_score'], 3),
        "upvotes": int(post['score']),
        "comments": int(post['num_comments']),
        "engagement_score": round(post['post_engagement_score'], 3),
        "date": post['date'].strftime('%Y-%m-%d') if pd.notnull(post['date']) else None
    })

with open('engagement_analysis.json', 'w') as f:
    json.dump(analysis_data, f, indent=2)

# print summary
print("\n" + "="*60)
print("High Engagement Themes")
print("="*60)
print(f"Analyzed {len(df):,} posts across {df['theme'].nunique()} themes")
print(f"Identified {len(high_engagement_posts):,} high engagement posts")
print(f"Engagement threshold: {top_10_percent_threshold:.3f}")

print(f"\nTop 10 themes by high engagement post rate:")
top_10 = theme_analysis.head(10)
for i, (_, theme) in enumerate(top_10.iterrows(), 1):
    print(f"{i:2d}. {theme['theme']:<25} {theme['high_engagement_rate']:>6.1%} "
          f"({theme['high_engagement_posts']:>3.0f}/{theme['total_posts']:>3.0f} posts)")

overperforming = theme_analysis[theme_analysis['overperforming']]
print(f"\nThemes significantly overperforming (>20% high engagement rate): {len(overperforming)}")
for _, theme in overperforming.iterrows():
    print(f"  • {theme['theme']}: {theme['high_engagement_rate']:.1%} rate")

print("\nFiles generated:")
print("- sentiment_vs_engagement.png (sentiment vs upvotes with engagement rates)")
print("- high_engagement_themes.png (top themes by viral post rate)")
print("- engagement_analysis.json (comprehensive data)")