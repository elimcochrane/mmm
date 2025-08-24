import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import json
from scipy import stats
import seaborn as sns
from datetime import datetime

df = pd.read_excel('2016-2024_posts.xlsx')

# formatting
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')

start_date = pd.Timestamp('2017-01-01')
df = df[df['date'] >= start_date]

df = df.dropna(subset=['date'])

df = df.rename(columns={
    'primary_theme': 'theme',
    'sentiment_compound': 'sentiment_score'
})

df = df[df['theme'] != 'Other']

# print dataset info
print(f"Filtered dataset contains {len(df)} posts")
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Unique themes: {df['theme'].nunique()}")

# 1 - trends by quarter
df['quarter'] = df['date'].dt.to_period('Q').astype(str)

# theme prevalence
theme_counts = df.groupby(['quarter', 'theme']).size().unstack(fill_value=0)
theme_percent = theme_counts.div(theme_counts.sum(axis=1), axis=0) * 100

# visualization
plt.figure(figsize=(14, 8))
theme_percent.plot.area(alpha=0.8, stacked=True, colormap='tab20')
plt.title('Theme Prevalence by Quarter (Excluding "Other")', fontsize=16)
plt.ylabel('Percentage of Posts', fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Themes')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('theme_prevalence_quarterly.png', dpi=300, bbox_inches='tight')
plt.close()

# 2 - theme-sentiment by quarter
sentiment_metrics = df.groupby(['quarter', 'theme'])['sentiment_score'].agg(
    ['mean', 'median', 'std', 'count']
).reset_index()

all_themes = df['theme'].unique()
sentiment_all = sentiment_metrics[sentiment_metrics['theme'].isin(all_themes)]

# visualization
plt.figure(figsize=(16, 10))
palette = sns.color_palette("husl", len(all_themes))

ax = sns.lineplot(
    data=sentiment_all,
    x='quarter',
    y='mean',
    hue='theme',
    style='theme',
    markers=True,
    dashes=False,
    markersize=8,
    linewidth=2,
    palette=palette
)

plt.title('Sentiment Evolution of All Themes', fontsize=16)
plt.ylabel('Mean Sentiment Score', fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.2)

for theme in all_themes:
    theme_data = sentiment_all[sentiment_all['theme'] == theme]
    if 'std' in theme_data.columns and not theme_data['std'].isnull().all():
        plt.fill_between(
            theme_data['quarter'],
            theme_data['mean'] - theme_data['std'],
            theme_data['mean'] + theme_data['std'],
            alpha=0.15
        )

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, title='Themes', 
           loc='upper left', bbox_to_anchor=(1, 1),
           ncol=2 if len(all_themes) > 15 else 1)

plt.tight_layout()
plt.savefig('theme_sentiment_quarterly.png', dpi=300, bbox_inches='tight')
plt.close()

# 3 - json output
output_data = {
    "temporal_trends": {
        "quarterly_summary": [],
        "theme_volatility": {}
    },
    "sentiment_relationships": {
        "quarterly_correlations": {},
        "significant_changes": []
    }
}

# quarterly theme prevalence stats
for quarter, row in theme_percent.iterrows():
    quarter_data = {
        "quarter": quarter,
        "total_posts": int(theme_counts.loc[quarter].sum()),
        "dominant_theme": row.idxmax(),
        "dominant_theme_percent": round(row.max(), 2),
        "theme_distribution": row.round(2).to_dict()
    }
    output_data["temporal_trends"]["quarterly_summary"].append(quarter_data)

# theme volatility
for theme in theme_percent.columns:
    theme_series = theme_percent[theme]
    if theme_series.mean() > 0:
        volatility = (theme_series.std() / theme_series.mean()) * 100
    else:
        volatility = 0
    output_data["temporal_trends"]["theme_volatility"][theme] = round(volatility, 2)

# sentiment correlations and significant changes
for theme in sentiment_metrics['theme'].unique():
    theme_data = sentiment_metrics[sentiment_metrics['theme'] == theme].sort_values('quarter')
    
    if len(theme_data) > 1:
        try:
            time_corr = stats.spearmanr(
                np.arange(len(theme_data)), 
                theme_data['mean']
            ).correlation
        except:
            time_corr = None
        
        # significant sentiment changes (>1 SD shift)
        theme_data['mean_diff'] = theme_data['mean'].diff().abs()
        theme_data['significant_change'] = theme_data['mean_diff'] > theme_data['std']
        
        if time_corr is not None:
            output_data["sentiment_relationships"]["quarterly_correlations"][theme] = round(time_corr, 3)
        
        for idx, row in theme_data[theme_data['significant_change']].iterrows():
            prev_idx = theme_data.index.get_loc(idx) - 1
            if prev_idx >= 0:
                prev_row = theme_data.iloc[prev_idx]
                change_record = {
                    "theme": theme,
                    "quarter": row['quarter'],
                    "sentiment_change": round(row['mean_diff'], 3),
                    "previous_sentiment": round(prev_row['mean'], 3),
                    "new_sentiment": round(row['mean'], 3),
                    "magnitude": "Large" if row['mean_diff'] > 1.0 else "Moderate"
                }
                output_data["sentiment_relationships"]["significant_changes"].append(change_record)

# save json output
with open('analysis_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("Output files generated:")
print("- theme_prevalence_quarterly.png")
print("- theme_sentiment_quarterly.png")
print("- analysis_results.json")
print(f"Processed {len(df)} posts from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")