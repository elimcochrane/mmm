import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('2016-2024_posts.xlsx')

# datetime formatting
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
df = df[(df['date'] >= pd.Timestamp('2017-01-01')) & (df['date'] <= pd.Timestamp('2024-12-31'))]
df = df.dropna(subset=['date'])
df = df.rename(columns={
    'sentiment_compound': 'sentiment_score'
})
df = df[df['primary_theme'] != 'Other']

# ensure numeric
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce')
df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
df = df.dropna(subset=['score', 'num_comments', 'sentiment_score'])

# calculate engagement
scaler = MinMaxScaler()
df[['upvotes_normalized', 'comments_normalized']] = scaler.fit_transform(
    df[['score', 'num_comments']]
)
df['engagement_score'] = 0.6 * df['upvotes_normalized'] + 0.4 * df['comments_normalized']

# quarters
df['year_quarter'] = df['date'].dt.to_period('Q')

# define theme groups
theme_groups = {
    'Identity Grounding': ['self-help', 'humanities', 'socialsciences'],
    'Central Topics': ['peterson', 'masculinity', 'politics'],
    'Out-groups': ['lgbtq', 'women']
}

# categorize themes
def categorize_theme(theme):
    theme_lower = theme.lower()
    for group, themes in theme_groups.items():
        if theme_lower in themes:
            return group
    return 'Other'  # should not happen

df['theme_group'] = df['primary_theme'].apply(categorize_theme)

# print theme distribution to verify categorization
print("Theme Distribution:")
print("Individual themes:")
print(df['primary_theme'].value_counts())
print("\nTheme Group Distribution:")
print(df['theme_group'].value_counts())
print("\nThemes by group:")
for group in theme_groups.keys():
    themes_in_group = df[df['theme_group'] == group]['primary_theme'].unique()
    print(f"{group}: {list(themes_in_group)}")

# calculate quarterly metrics for each theme group
quarterly_metrics = df.groupby(['year_quarter', 'theme_group']).agg(
    avg_sentiment=('sentiment_score', 'mean'),
    avg_engagement=('engagement_score', 'mean'),
    avg_upvotes=('score', 'mean'),
    avg_comments=('num_comments', 'mean'),
    post_count=('theme_group', 'count')
).reset_index()

# convert period to string for easier handling
quarterly_metrics['quarter_str'] = quarterly_metrics['year_quarter'].astype(str)
quarterly_metrics['date'] = quarterly_metrics['year_quarter'].dt.start_time

# filter any remaining 'Other' and ensure enough data
quarterly_metrics = quarterly_metrics[
    (quarterly_metrics['theme_group'] != 'Other') & 
    (quarterly_metrics['post_count'] >= 3)
]

print(f"\nAnalyzing {len(quarterly_metrics)} quarterly data points")
print(f"Date range: {quarterly_metrics['quarter_str'].min()} to {quarterly_metrics['quarter_str'].max()}")

# visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Theme Group Evolution Over Time (2017-2024)', fontsize=20, y=0.98)
colors = {'Identity Grounding': '#2E8B57', 'Central Topics': '#4682B4', 'Out-groups': '#DC143C'}

# 1 - sentiment over time
ax1 = axes[0, 0]
for group in quarterly_metrics['theme_group'].unique():
    group_data = quarterly_metrics[quarterly_metrics['theme_group'] == group]
    ax1.plot(group_data['date'], group_data['avg_sentiment'], 
             marker='o', linewidth=2.5, markersize=6, label=group, color=colors.get(group, 'gray'))

ax1.set_title('Average Sentiment by Quarter', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Average Sentiment Score')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# 2 - engagement over time
ax2 = axes[0, 1]
for group in quarterly_metrics['theme_group'].unique():
    group_data = quarterly_metrics[quarterly_metrics['theme_group'] == group]
    ax2.plot(group_data['date'], group_data['avg_engagement'], 
             marker='s', linewidth=2.5, markersize=6, label=group, color=colors.get(group, 'gray'))

ax2.set_title('Average Engagement by Quarter', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time')
ax2.set_ylabel('Average Engagement Score')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.tick_params(axis='x', rotation=45)

# 3 - post volume over time
ax3 = axes[1, 0]
for group in quarterly_metrics['theme_group'].unique():
    group_data = quarterly_metrics[quarterly_metrics['theme_group'] == group]
    ax3.plot(group_data['date'], group_data['post_count'], 
             marker='^', linewidth=2.5, markersize=6, label=group, color=colors.get(group, 'gray'))

ax3.set_title('Post Volume by Quarter', fontsize=14, fontweight='bold')
ax3.set_xlabel('Time')
ax3.set_ylabel('Number of Posts')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.tick_params(axis='x', rotation=45)

# 4 - sent vs eng scatter (latest quarter)
ax4 = axes[1, 1]
latest_quarter = quarterly_metrics['year_quarter'].max()
latest_data = quarterly_metrics[quarterly_metrics['year_quarter'] == latest_quarter]

scatter = ax4.scatter(latest_data['avg_sentiment'], latest_data['avg_engagement'], 
                     s=latest_data['post_count']*3, alpha=0.7,
                     c=[colors.get(group, 'gray') for group in latest_data['theme_group']])

for i, row in latest_data.iterrows():
    ax4.annotate(row['theme_group'], 
                (row['avg_sentiment'], row['avg_engagement']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax4.set_title(f'Sentiment vs Engagement ({latest_quarter})', fontsize=14, fontweight='bold')
ax4.set_xlabel('Average Sentiment Score')
ax4.set_ylabel('Average Engagement Score')
ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('theme_groups_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax1 = plt.subplots(figsize=(16, 10))

# sent on y
for group in quarterly_metrics['theme_group'].unique():
    group_data = quarterly_metrics[quarterly_metrics['theme_group'] == group].sort_values('date')
    ax1.plot(group_data['date'], group_data['avg_sentiment'], 
             marker='o', linewidth=3, markersize=7, label=f'{group} (Sentiment)', 
             color=colors.get(group, 'gray'), alpha=0.8)

ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Average Sentiment Score', fontsize=12, color='black')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# eng on y
ax2 = ax1.twinx()
for group in quarterly_metrics['theme_group'].unique():
    group_data = quarterly_metrics[quarterly_metrics['theme_group'] == group].sort_values('date')
    ax2.plot(group_data['date'], group_data['avg_engagement'], 
             marker='s', linewidth=3, markersize=7, label=f'{group} (Engagement)', 
             color=colors.get(group, 'gray'), linestyle='--', alpha=0.6)

ax2.set_ylabel('Average Engagement Score', fontsize=12, color='black')

# combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

plt.title('Theme Group Sentiment and Engagement Over Time\n(Solid lines = Sentiment, Dashed lines = Engagement)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('sentiment_engagement_dual_axis.png', dpi=300, bbox_inches='tight')
plt.close()

# summary statistics
summary_stats = {}
for group in quarterly_metrics['theme_group'].unique():
    group_data = quarterly_metrics[quarterly_metrics['theme_group'] == group].sort_values('date')
    
    # calculate trends
    from scipy import stats
    quarters_numeric = range(len(group_data))
    sentiment_slope, sentiment_intercept, sentiment_r, _, _ = stats.linregress(quarters_numeric, group_data['avg_sentiment'])
    engagement_slope, engagement_intercept, engagement_r, _, _ = stats.linregress(quarters_numeric, group_data['avg_engagement'])
    
    summary_stats[group] = {
        'avg_sentiment': group_data['avg_sentiment'].mean(),
        'avg_engagement': group_data['avg_engagement'].mean(),
        'sentiment_trend': sentiment_slope,
        'engagement_trend': engagement_slope,
        'sentiment_correlation': sentiment_r,
        'engagement_correlation': engagement_r,
        'total_posts': group_data['post_count'].sum(),
        'quarters_active': len(group_data)
    }

# summary stats to json
json_summary_stats = {}
for group, stats in summary_stats.items():
    json_summary_stats[group] = {
        'avg_sentiment': float(stats['avg_sentiment']),
        'avg_engagement': float(stats['avg_engagement']),
        'sentiment_trend': float(stats['sentiment_trend']),
        'engagement_trend': float(stats['engagement_trend']),
        'sentiment_correlation': float(stats['sentiment_correlation']),
        'engagement_correlation': float(stats['engagement_correlation']),
        'total_posts': int(stats['total_posts']),
        'quarters_active': int(stats['quarters_active'])
    }

# save results
results = {
    "analysis_period": "2017-Q1 to 2024-Q4",
    "theme_groups": theme_groups,
    "summary_statistics": json_summary_stats,
    "quarterly_data": []
}

for _, row in quarterly_metrics.iterrows():
    results["quarterly_data"].append({
        "quarter": str(row['quarter_str']),
        "theme_group": str(row['theme_group']),
        "avg_sentiment": float(round(row['avg_sentiment'], 3)),
        "avg_engagement": float(round(row['avg_engagement'], 3)),
        "avg_upvotes": float(round(row['avg_upvotes'], 1)),
        "avg_comments": float(round(row['avg_comments'], 1)),
        "post_count": int(row['post_count'])
    })

with open('theme_groups_time_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

# print summary
print("\n" + "="*80)
print("THEME GROUP TIME SERIES ANALYSIS SUMMARY")
print("="*80)

for group, stats in summary_stats.items():
    print(f"\n{group.upper()}:")
    print(f"  Average Sentiment: {stats['avg_sentiment']:+.3f}")
    print(f"  Average Engagement: {stats['avg_engagement']:.3f}")
    print(f"  Sentiment Trend: {stats['sentiment_trend']:+.4f} per quarter (r={stats['sentiment_correlation']:+.3f})")
    print(f"  Engagement Trend: {stats['engagement_trend']:+.4f} per quarter (r={stats['engagement_correlation']:+.3f})")
    print(f"  Total Posts: {stats['total_posts']:,}")
    print(f"  Active Quarters: {stats['quarters_active']}")

print(f"\nFiles generated:")
print("- theme_groups_time_series.png (4-panel overview)")
print("- sentiment_engagement_dual_axis.png (focused dual-axis plot)")
print("- theme_groups_time_analysis.json (complete data)")