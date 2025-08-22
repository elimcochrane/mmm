import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_reddit_themes(file_2017_2019, file_2023_2024, theme_column='primary_theme', exclude_themes=['other']):
    """
    Compare the percentage distribution of themes between two time periods
    Includes both primary_theme and secondary_theme in calculations
    
    Parameters:
    file_2017_2019: path to Excel file containing 2017-2019 data
    file_2023_2024: path to Excel file containing 2023-2024 data
    theme_column: name of the column containing themes (default: 'primary_theme')
    exclude_themes: list of themes to exclude from analysis (default: ['other'])
    """
    
    # Read Excel files
    print("Reading Excel files...")
    df_old = pd.read_excel(file_2017_2019)
    df_new = pd.read_excel(file_2023_2024)
    
    def count_themes(df, exclude_themes):
        """Count themes from both primary and secondary columns"""
        total_posts = len(df)
        theme_counts = {}
        
        # Combine primary and secondary themes
        all_themes = pd.concat([df['primary_theme'], df['secondary_theme']]).dropna()
        
        # Filter out excluded themes
        if exclude_themes:
            all_themes = all_themes[~all_themes.isin(exclude_themes)]
        
        # Count unique posts that mention each theme
        for theme in all_themes.unique():
            posts_with_theme = df[
                (df['primary_theme'] == theme) | (df['secondary_theme'] == theme)
            ]
            if exclude_themes:
                # Only count posts that don't have excluded themes as primary
                posts_with_theme = posts_with_theme[~posts_with_theme['primary_theme'].isin(exclude_themes)]
            
            theme_counts[theme] = len(posts_with_theme)
        
        return theme_counts, total_posts
    
    # Calculate theme counts for each period
    old_counts, old_total = count_themes(df_old, exclude_themes)
    new_counts, new_total = count_themes(df_new, exclude_themes)
    
    if exclude_themes:
        print(f"Excluded themes: {', '.join(exclude_themes)}")
    
    # Convert to percentages
    old_percentages = {theme: (count / old_total * 100) for theme, count in old_counts.items()}
    new_percentages = {theme: (count / new_total * 100) for theme, count in new_counts.items()}
    
    old_percentages = pd.Series(old_percentages).round(2)
    new_percentages = pd.Series(new_percentages).round(2)
    
    # Create comparison dataframe
    all_themes = set(old_percentages.index) | set(new_percentages.index)
    
    comparison_df = pd.DataFrame({
        '2017-2019 (%)': [old_percentages.get(theme, 0) for theme in all_themes],
        '2023-2024 (%)': [new_percentages.get(theme, 0) for theme in all_themes],
        'Change (%)': [new_percentages.get(theme, 0) - old_percentages.get(theme, 0) for theme in all_themes]
    }, index=list(all_themes))
    
    # Sort by absolute change
    comparison_df = comparison_df.reindex(comparison_df['Change (%)'].abs().sort_values(ascending=False).index)
    
    # Display results
    print(f"\nTheme Distribution Comparison (Primary + Secondary Themes)")
    print(f"{'='*60}")
    print(f"2017-2019 period: {old_total:,} total posts")
    print(f"2023-2024 period: {new_total:,} total posts")
    print(f"\nPercentage by Theme:")
    print(comparison_df.to_string())
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Define colors
    palette = ['#0051ba', '#e8000d']
    
    # Prepare data for Seaborn
    top_themes = comparison_df.index[:10]  # Top 10 themes by change
    df_melted = comparison_df.loc[top_themes, ['2017-2019 (%)', '2023-2024 (%)']].reset_index()
    df_melted = df_melted.melt(id_vars='index', value_vars=['2017-2019 (%)', '2023-2024 (%)'],
                              var_name='Period', value_name='Percentage')
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Define consistent color mapping
    color_mapping = {
    '2017-2019 (%)': '#0051ba',
    '2023-2024 (%)': '#e8000d',
    'Increase': '#0051ba',
    'Decrease': '#e8000d'
}

    # First plot - Comparison bars
    sns.barplot(data=df_melted, x='index', y='Percentage', hue='Period',
            palette=[color_mapping['2017-2019 (%)'], color_mapping['2023-2024 (%)']],
            alpha=0.85, ax=ax1)
    
    ax1.set_xlabel('Themes')
    ax1.set_ylabel('Percentage of Posts')
    ax1.set_title('Theme Distribution Comparison (Top 10 by Change)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Period')
    ax1.grid(True, alpha=0.3)
    
    # Second plot - Changes
    changes_df = comparison_df.loc[top_themes, 'Change (%)'].reset_index()
    changes_df['color'] = changes_df['Change (%)'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')
    
    sns.barplot(data=changes_df, y='index', x='Change (%)', hue='color',
            palette=[color_mapping['Increase'], color_mapping['Decrease']],
            alpha=0.85, ax=ax2, legend=False)
    
    ax2.set_xlabel('Percentage Point Change')
    ax2.set_ylabel('Themes')
    ax2.set_title('Change in Theme Prevalence (2017-2019 vs 2023-2024)')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

def save_results(comparison_df, output_file='theme_comparison_results.xlsx'):
    """Save comparison results to Excel file"""
    comparison_df.to_excel(output_file)
    print(f"\nResults saved to: {output_file}")

# Configuration
if __name__ == "__main__":
    file1 = "enhanced_analysis_2017-2019.xlsx"
    file2 = "enhanced_analysis_2023-2024.xlsx"
    
    try:
        results = compare_reddit_themes(file1, file2, exclude_themes=['Other'])
        
        # To include 'other' theme, use:
        # results = compare_reddit_themes(file1, file2, exclude_themes=[])
        
        # To exclude multiple themes, use:
        # results = compare_reddit_themes(file1, file2, exclude_themes=['other', 'misc', 'unknown'])
        
        save_results(results)
        
        print(f"\nSummary Statistics:")
        print(f"Number of unique themes: {len(results)}")
        print(f"Biggest increase: {results['Change (%)'].max():.2f}% ({results['Change (%)'].idxmax()})")
        print(f"Biggest decrease: {results['Change (%)'].min():.2f}% ({results['Change (%)'].idxmin()})")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please make sure your Excel files are in the same directory as this script")
        print("and update the file paths in the script")
    except Exception as e:
        print(f"Error: {e}")