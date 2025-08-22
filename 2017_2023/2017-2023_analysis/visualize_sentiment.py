import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("stance_analysis_results.xlsx")
df.columns = df.columns.map(str)
df_subset = df[['theme', '2017', '2023']]

# Melt the data to long format for seaborn
df_melted = df_subset.melt(id_vars='theme', value_vars=['2017', '2023'],
                           var_name='Year', value_name='Avg_Sentiment')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='theme', y='Avg_Sentiment', hue='Year', alpha=0.85, 
            palette=['#0051ba', '#e8000d'])

plt.title('Average Sentiment per Theme: 2017 vs 2023')
plt.ylabel('Average Sentiment Score')
plt.xlabel('Theme')
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='gray', linestyle='--')
plt.legend(title='Year')
plt.tight_layout()
plt.show()