import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

class RedditSentimentAnalysis:
    def __init__(self, file_path_2017_2019, file_path_2023_2024):
        """
        Initialize the analysis with two Excel files containing Reddit sentiment data.
        
        Expected columns in Excel files:
        - primary_theme: categorical variable (e.g., 'self-help', 'politics_events', etc.)
        - sentiment_compound: numerical sentiment score
        - stance: categorical variable
        - virality_score: numerical variable
        - sentiment_category: categorical variable
        """
        self.df_2017_2019 = pd.read_excel(file_path_2017_2019)
        self.df_2023_2024 = pd.read_excel(file_path_2023_2024)
        self.results = {}
        
        # Add time period labels
        self.df_2017_2019['time_period'] = '2017-2019'
        self.df_2023_2024['time_period'] = '2023-2024'
        
        # Combine datasets
        self.combined_df = pd.concat([self.df_2017_2019, self.df_2023_2024], ignore_index=True)
        
        print("Data loaded successfully!")
        print(f"2017-2019 dataset: {len(self.df_2017_2019)} posts")
        print(f"2023-2024 dataset: {len(self.df_2023_2024)} posts")
        print(f"Combined dataset: {len(self.combined_df)} posts")
        
        # Display available columns
        print(f"\nColumns in dataset: {list(self.combined_df.columns)}")
    
    def explore_data(self):
        """Explore the data structure and basic statistics."""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print("\nDataset columns:")
        print(self.combined_df.columns.tolist())
        
        print("\nPrimary themes in dataset:")
        themes = self.combined_df['primary_theme'].unique()
        print(themes)
        
        print("\nSample size by primary theme and time period:")
        sample_sizes = self.combined_df.groupby(['primary_theme', 'time_period']).size().unstack(fill_value=0)
        print(sample_sizes)
        
        print("\nBasic statistics by primary theme and time period:")
        desc_stats = self.combined_df.groupby(['primary_theme', 'time_period'])['sentiment_compound'].describe()
        print(desc_stats)
        
        # Additional exploration for new columns
        print("\nStance distribution:")
        stance_dist = self.combined_df.groupby(['stance', 'time_period']).size().unstack(fill_value=0)
        print(stance_dist)
        
        print("\nSentiment category distribution:")
        sentiment_cat_dist = self.combined_df.groupby(['sentiment_category', 'time_period']).size().unstack(fill_value=0)
        print(sentiment_cat_dist)
        
        print("\nVirality score statistics:")
        virality_stats = self.combined_df.groupby('time_period')['virality_score'].describe()
        print(virality_stats)
        
        return themes, sample_sizes
    
    def check_assumptions(self, theme):
        """Check statistical assumptions for a specific theme."""
        # Get data for the theme
        data_2017 = self.df_2017_2019[self.df_2017_2019['primary_theme'] == theme]['sentiment_compound']
        data_2023 = self.df_2023_2024[self.df_2023_2024['primary_theme'] == theme]['sentiment_compound']
        
        # Remove any missing values
        data_2017 = data_2017.dropna()
        data_2023 = data_2023.dropna()
        
        if len(data_2017) < 30 or len(data_2023) < 30:
            return {
                'theme': theme,
                'sufficient_data': False,
                'normality_2017': None,
                'normality_2023': None,
                'equal_variances': None,
                'recommendation': 'Insufficient data for statistical testing'
            }
        
        # For large samples (>30,000), we'll use visual inspection and statistical tests
        # but rely more on Central Limit Theorem for normality assumption
        
        # Test normality - for large samples, we'll use a subsample for Shapiro-Wilk
        # and rely on CLT for the actual analysis
        sample_size_for_test = min(5000, len(data_2017))
        data_2017_sample = data_2017.sample(sample_size_for_test, random_state=42)
        data_2023_sample = data_2023.sample(min(5000, len(data_2023)), random_state=42)
        
        shapiro_2017 = shapiro(data_2017_sample)
        shapiro_2023 = shapiro(data_2023_sample)
        
        # For large samples, we'll assume normality due to CLT, but report the test results
        normality_2017 = True  # CLT assumption for large samples
        normality_2023 = True  # CLT assumption for large samples
        
        # Test equal variances (Levene's test)
        levene_stat, levene_p = levene(data_2017, data_2023)
        equal_variances = levene_p > 0.05
        
        # For large samples, we'll use Welch's t-test by default due to robustness
        recommendation = "Welch's t-test (robust for large samples)"
        
        return {
            'theme': theme,
            'sufficient_data': True,
            'n_2017': len(data_2017),
            'n_2023': len(data_2023),
            'normality_2017': normality_2017,
            'normality_2023': normality_2023,
            'shapiro_p_2017': shapiro_2017.pvalue,
            'shapiro_p_2023': shapiro_2023.pvalue,
            'equal_variances': equal_variances,
            'levene_p': levene_p,
            'recommendation': recommendation
        }
    
    def perform_statistical_test(self, theme):
        """Perform appropriate statistical test for a specific theme."""
        assumptions = self.check_assumptions(theme)
        
        if not assumptions['sufficient_data']:
            return {
                'theme': theme,
                'test_performed': 'None',
                'statistic': None,
                'p_value': None,
                'effect_size': None,
                'mean_2017': None,
                'mean_2023': None,
                'std_2017': None,
                'std_2023': None,
                'difference': None,
                'confidence_interval': None,
                'interpretation': 'Insufficient data'
            }
        
        # Get data
        data_2017 = self.df_2017_2019[self.df_2017_2019['primary_theme'] == theme]['sentiment_compound'].dropna()
        data_2023 = self.df_2023_2024[self.df_2023_2024['primary_theme'] == theme]['sentiment_compound'].dropna()
        
        mean_2017 = data_2017.mean()
        mean_2023 = data_2023.mean()
        std_2017 = data_2017.std()
        std_2023 = data_2023.std()
        difference = mean_2023 - mean_2017
        
        # For large samples, use Welch's t-test (more robust)
        statistic, p_value = ttest_ind(data_2017, data_2023, equal_var=False)
        test_performed = "Welch's t-test"
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data_2017) - 1) * data_2017.var() + 
                            (len(data_2023) - 1) * data_2023.var()) / 
                           (len(data_2017) + len(data_2023) - 2))
        effect_size = difference / pooled_std
        
        # Calculate confidence interval for mean difference
        se_diff = np.sqrt(data_2017.var()/len(data_2017) + data_2023.var()/len(data_2023))
        # Use conservative df for Welch's t-test
        df = ((data_2017.var()/len(data_2017) + data_2023.var()/len(data_2023))**2) / \
             ((data_2017.var()/len(data_2017))**2/(len(data_2017)-1) + 
              (data_2023.var()/len(data_2023))**2/(len(data_2023)-1))
        
        t_critical = stats.t.ppf(0.975, df)
        ci_lower = difference - t_critical * se_diff
        ci_upper = difference + t_critical * se_diff
        confidence_interval = (ci_lower, ci_upper)
        
        return {
            'theme': theme,
            'test_performed': test_performed,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'mean_2017': mean_2017,
            'mean_2023': mean_2023,
            'std_2017': std_2017,
            'std_2023': std_2023,
            'difference': difference,
            'confidence_interval': confidence_interval,
            'n_2017': len(data_2017),
            'n_2023': len(data_2023)
        }
    
    def analyze_by_stance(self):
        """Analyze sentiment differences by stance across time periods."""
        print("\n" + "="*50)
        print("STANCE ANALYSIS")
        print("="*50)
        
        stance_results = []
        stances = self.combined_df['stance'].dropna().unique()
        
        for stance in stances:
            data_2017 = self.df_2017_2019[self.df_2017_2019['stance'] == stance]['sentiment_compound'].dropna()
            data_2023 = self.df_2023_2024[self.df_2023_2024['stance'] == stance]['sentiment_compound'].dropna()
            
            if len(data_2017) >= 30 and len(data_2023) >= 30:
                # Perform Welch's t-test
                statistic, p_value = ttest_ind(data_2017, data_2023, equal_var=False)
                
                # Calculate effect size
                pooled_std = np.sqrt(((len(data_2017) - 1) * data_2017.var() + 
                                    (len(data_2023) - 1) * data_2023.var()) / 
                                   (len(data_2017) + len(data_2023) - 2))
                effect_size = (data_2023.mean() - data_2017.mean()) / pooled_std
                
                stance_results.append({
                    'stance': stance,
                    'mean_2017': data_2017.mean(),
                    'mean_2023': data_2023.mean(),
                    'difference': data_2023.mean() - data_2017.mean(),
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'n_2017': len(data_2017),
                    'n_2023': len(data_2023)
                })
        
        return stance_results
    
    def analyze_virality_correlation(self):
        """Analyze correlation between virality and sentiment."""
        print("\n" + "="*50)
        print("VIRALITY CORRELATION ANALYSIS")
        print("="*50)
        
        # Calculate correlations for each time period
        corr_2017 = self.df_2017_2019['sentiment_compound'].corr(self.df_2017_2019['virality_score'])
        corr_2023 = self.df_2023_2024['sentiment_compound'].corr(self.df_2023_2024['virality_score'])
        
        print(f"Correlation (2017-2019): {corr_2017:.3f}")
        print(f"Correlation (2023-2024): {corr_2023:.3f}")
        
        # Test if correlations are significantly different
        # Fisher's z-transformation for comparing correlations
        n1, n2 = len(self.df_2017_2019), len(self.df_2023_2024)
        
        z1 = 0.5 * np.log((1 + corr_2017) / (1 - corr_2017))
        z2 = 0.5 * np.log((1 + corr_2023) / (1 - corr_2023))
        
        se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
        z_stat = (z1 - z2) / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'corr_2017': corr_2017,
            'corr_2023': corr_2023,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }
    
    def run_full_analysis(self):
        """Run the complete statistical analysis for all themes."""
        print("\n" + "="*50)
        print("RUNNING STATISTICAL ANALYSIS")
        print("="*50)
        
        themes, sample_sizes = self.explore_data()
        
        # Check assumptions for all themes
        print("\nChecking assumptions for each theme...")
        assumption_results = []
        for theme in themes:
            if theme in self.df_2017_2019['primary_theme'].values and theme in self.df_2023_2024['primary_theme'].values:
                assumptions = self.check_assumptions(theme)
                assumption_results.append(assumptions)
        
        # Display assumption results
        print("\nAssumption Check Results:")
        print("-" * 100)
        for result in assumption_results:
            if result['sufficient_data']:
                print(f"{result['theme']:<25} | N1: {result['n_2017']:<6} N2: {result['n_2023']:<6} | {result['recommendation']}")
            else:
                print(f"{result['theme']:<25} | {result['recommendation']}")
        
        # Perform statistical tests
        print("\nPerforming statistical tests...")
        test_results = []
        for theme in themes:
            if theme in self.df_2017_2019['primary_theme'].values and theme in self.df_2023_2024['primary_theme'].values:
                result = self.perform_statistical_test(theme)
                test_results.append(result)
        
        # Apply multiple comparison corrections
        p_values = [r['p_value'] for r in test_results if r['p_value'] is not None]
        if p_values:
            # Bonferroni correction
            bonferroni_corrected = multipletests(p_values, alpha=0.05, method='bonferroni')[1]
            
            # FDR correction (Benjamini-Hochberg)
            fdr_corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')[1]
            
            # Add corrections to results
            p_idx = 0
            for result in test_results:
                if result['p_value'] is not None:
                    result['p_bonferroni'] = bonferroni_corrected[p_idx]
                    result['p_fdr'] = fdr_corrected[p_idx]
                    p_idx += 1
        
        # Additional analyses
        stance_results = self.analyze_by_stance()
        virality_results = self.analyze_virality_correlation()
        
        self.results = test_results
        self.stance_results = stance_results
        self.virality_results = virality_results
        
        return test_results, stance_results, virality_results
    
    def display_results(self):
        """Display the results in a formatted table."""
        print("\n" + "="*50)
        print("STATISTICAL TEST RESULTS")
        print("="*50)
        
        # Create results DataFrame
        results_data = []
        for result in self.results:
            if result['p_value'] is not None:
                results_data.append({
                    'Theme': result['theme'],
                    'Test': result['test_performed'],
                    'N_2017': result['n_2017'],
                    'N_2023': result['n_2023'],
                    'Mean_2017': f"{result['mean_2017']:.3f}",
                    'Mean_2023': f"{result['mean_2023']:.3f}",
                    'Difference': f"{result['difference']:.3f}",
                    'Effect_Size': f"{result['effect_size']:.3f}",
                    'p_value': f"{result['p_value']:.6f}",
                    'p_bonferroni': f"{result.get('p_bonferroni', 'N/A'):.6f}" if result.get('p_bonferroni') != 'N/A' else 'N/A',
                    'p_fdr': f"{result.get('p_fdr', 'N/A'):.6f}" if result.get('p_fdr') != 'N/A' else 'N/A',
                    'Significant': 'Yes' if result['p_value'] < 0.05 else 'No',
                    'Sig_Bonferroni': 'Yes' if result.get('p_bonferroni', 1) < 0.05 else 'No',
                    'Sig_FDR': 'Yes' if result.get('p_fdr', 1) < 0.05 else 'No'
                })
        
        results_df = pd.DataFrame(results_data)
        
        # Sort by p-value
        results_df = results_df.sort_values('p_value')
        
        print(results_df.to_string(index=False))
        
        # Display stance results
        if hasattr(self, 'stance_results'):
            print("\n" + "="*50)
            print("STANCE ANALYSIS RESULTS")
            print("="*50)
            
            stance_df = pd.DataFrame(self.stance_results)
            if not stance_df.empty:
                print(stance_df.to_string(index=False))
        
        # Display virality results
        if hasattr(self, 'virality_results'):
            print("\n" + "="*50)
            print("VIRALITY CORRELATION RESULTS")
            print("="*50)
            
            vr = self.virality_results
            print(f"Correlation 2017-2019: {vr['corr_2017']:.3f}")
            print(f"Correlation 2023-2024: {vr['corr_2023']:.3f}")
            print(f"Difference significant: {vr['significant_difference']}")
            print(f"p-value: {vr['p_value']:.6f}")
        
        # Summary statistics
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        
        total_tests = len([r for r in self.results if r['p_value'] is not None])
        sig_uncorrected = len([r for r in self.results if r['p_value'] is not None and r['p_value'] < 0.05])
        sig_bonferroni = len([r for r in self.results if r.get('p_bonferroni') is not None and r['p_bonferroni'] < 0.05])
        sig_fdr = len([r for r in self.results if r.get('p_fdr') is not None and r['p_fdr'] < 0.05])
        
        print(f"Total themes tested: {total_tests}")
        print(f"Significant (uncorrected p < 0.05): {sig_uncorrected}")
        print(f"Significant (Bonferroni corrected): {sig_bonferroni}")
        print(f"Significant (FDR corrected): {sig_fdr}")
        print(f"Bonferroni alpha level: {0.05/total_tests:.6f}")
        
        return results_df
    
    def create_visualizations(self):
        """Create visualizations of the results."""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots for main analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Bar plot comparing means
        themes = [r['theme'] for r in self.results if r['p_value'] is not None]
        means_2017 = [r['mean_2017'] for r in self.results if r['p_value'] is not None]
        means_2023 = [r['mean_2023'] for r in self.results if r['p_value'] is not None]
        
        x = np.arange(len(themes))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, means_2017, width, label='2017-2019', alpha=0.8)
        axes[0, 0].bar(x + width/2, means_2023, width, label='2023-2024', alpha=0.8)
        axes[0, 0].set_xlabel('Theme')
        axes[0, 0].set_ylabel('Mean Sentiment')
        axes[0, 0].set_title('Mean Sentiment by Theme and Time Period')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(themes, rotation=45, ha='right')
        axes[0, 0].legend()
        
        # 2. Effect sizes
        effect_sizes = [r['effect_size'] for r in self.results if r['p_value'] is not None]
        colors = ['red' if es < 0 else 'blue' for es in effect_sizes]
        
        axes[0, 1].bar(themes, effect_sizes, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Theme')
        axes[0, 1].set_ylabel('Effect Size (Cohen\'s d)')
        axes[0, 1].set_title('Effect Sizes by Theme')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add effect size interpretation lines
        axes[0, 1].axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
        axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
        axes[0, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
        axes[0, 1].legend()
        
        # 3. P-values with multiple comparison corrections
        p_values = [r['p_value'] for r in self.results if r['p_value'] is not None]
        p_fdr = [r.get('p_fdr', 1) for r in self.results if r['p_value'] is not None]
        
        x_pos = np.arange(len(themes))
        axes[1, 0].bar(x_pos - 0.2, p_values, 0.4, label='Original p-values', alpha=0.7)
        axes[1, 0].bar(x_pos + 0.2, p_fdr, 0.4, label='FDR corrected', alpha=0.7)
        axes[1, 0].set_xlabel('Theme')
        axes[1, 0].set_ylabel('P-value')
        axes[1, 0].set_title('P-values (Original vs FDR Corrected)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(themes, rotation=45, ha='right')
        axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α = 0.05')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        
        # 4. Differences with confidence intervals
        differences = [r['difference'] for r in self.results if r['p_value'] is not None]
        ci_lower = [r['confidence_interval'][0] for r in self.results if r['p_value'] is not None]
        ci_upper = [r['confidence_interval'][1] for r in self.results if r['p_value'] is not None]
        
        # Plot differences
        axes[1, 1].bar(themes, differences, alpha=0.7, color='green')
        
        # Add confidence intervals
        for i, (theme, diff, ci_l, ci_u) in enumerate(zip(themes, differences, ci_lower, ci_upper)):
            axes[1, 1].errorbar(i, diff, yerr=[[diff - ci_l], [ci_u - diff]], 
                               fmt='none', color='black', capsize=5)
        
        axes[1, 1].set_xlabel('Theme')
        axes[1, 1].set_ylabel('Mean Difference (2023-2024 minus 2017-2019)')
        axes[1, 1].set_title('Mean Differences with 95% Confidence Intervals')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional analyses visualizations
        self.create_additional_visualizations()
    
    def create_additional_visualizations(self):
        """Create additional visualizations for stance and virality analyses."""
        
        # Stance analysis visualization
        if hasattr(self, 'stance_results') and self.stance_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            stance_df = pd.DataFrame(self.stance_results)
            
            # Stance means comparison
            x = np.arange(len(stance_df))
            width = 0.35
            
            ax1.bar(x - width/2, stance_df['mean_2017'], width, label='2017-2019', alpha=0.8)
            ax1.bar(x + width/2, stance_df['mean_2023'], width, label='2023-2024', alpha=0.8)
            ax1.set_xlabel('Stance')
            ax1.set_ylabel('Mean Sentiment')
            ax1.set_title('Mean Sentiment by Stance and Time Period')
            ax1.set_xticks(x)
            ax1.set_xticklabels(stance_df['stance'], rotation=45, ha='right')
            ax1.legend()
            
            # Stance effect sizes
            colors = ['red' if es < 0 else 'blue' for es in stance_df['effect_size']]
            ax2.bar(stance_df['stance'], stance_df['effect_size'], color=colors, alpha=0.7)
            ax2.set_xlabel('Stance')
            ax2.set_ylabel('Effect Size (Cohen\'s d)')
            ax2.set_title('Effect Sizes by Stance')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Virality correlation visualization
        if hasattr(self, 'virality_results'):
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plots for each time period
            axes[0].scatter(self.df_2017_2019['sentiment_compound'], 
                           self.df_2017_2019['virality_score'], 
                           alpha=0.1, s=1)
            axes[0].set_xlabel('Sentiment Compound')
            axes[0].set_ylabel('Virality Score')
            axes[0].set_title(f'2017-2019 (r = {self.virality_results["corr_2017"]:.3f})')
            
            axes[1].scatter(self.df_2023_2024['sentiment_compound'], 
                           self.df_2023_2024['virality_score'], 
                           alpha=0.1, s=1)
            axes[1].set_xlabel('Sentiment Compound')
            axes[1].set_ylabel('Virality Score')
            axes[1].set_title(f'2023-2024 (r = {self.virality_results["corr_2023"]:.3f})')
            
            plt.tight_layout()
            plt.show()

# Updated main function
def main():
    """
    Main function demonstrating how to use the analysis.
    
    UPDATED FOR YOUR DATA STRUCTURE:
    - Uses 'primary_theme' instead of 'theme'
    - Includes analysis of 'stance', 'virality_score', and 'sentiment_category'
    - Optimized for large datasets (30,000-50,000 posts)
    """
    
    # STEP 1: Update these file paths to your actual Excel files
    file_path_2017_2019 = "C:/Users/elise/Dropbox/Research/results/results_2017-2019/enhanced_analysis_2017-2019.xlsx"
    file_path_2023_2024 = "C:/Users/elise/Dropbox/Research/results/results_2023-2024/enhanced_analysis_2023-2024.xlsx"
    
    # STEP 2: Initialize the analysis
    try:
        analysis = RedditSentimentAnalysis(file_path_2017_2019, file_path_2023_2024)
        
        # STEP 3: Run the complete analysis
        theme_results, stance_results, virality_results = analysis.run_full_analysis()
        
        # STEP 4: Display results
        results_df = analysis.display_results()
        
        # STEP 5: Create visualizations
        analysis.create_visualizations()
        
        # STEP 6: Save results to Excel
        # Save main theme analysis results
        results_df.to_excel("sentiment_analysis_results.xlsx", index=False)
        
        # Save stance analysis results
        if stance_results:
            stance_df = pd.DataFrame(stance_results)
            stance_df.to_excel("stance_analysis_results.xlsx", index=False)
        
        # Save virality analysis results
        virality_df = pd.DataFrame([virality_results])
        virality_df.to_excel("virality_analysis_results.xlsx", index=False)
        
        print("\nResults saved to multiple Excel files:")
        print("- sentiment_analysis_results.xlsx (main theme analysis)")
        print("- stance_analysis_results.xlsx (stance analysis)")
        print("- virality_analysis_results.xlsx (virality correlation analysis)")
        
        return analysis, results_df
        
    except FileNotFoundError as e:
        print(f"Error: Could not find the Excel file. Please check the file path.")
        print(f"Make sure your Excel files are in the same directory as this script.")
        print(f"Expected files: {file_path_2017_2019} and {file_path_2023_2024}")
        return None, None
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Run the analysis
    analysis, results = main()
    
    # If successful, you can access individual results like this:
    if analysis and results is not None:
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE - ADDITIONAL INSIGHTS")
        print("="*50)
        
        # Display some key insights
        print("\nKey Findings:")
        
        # Most significant changes by theme
        significant_themes = [r for r in analysis.results if r['p_value'] is not None and r['p_value'] < 0.05]
        if significant_themes:
            print(f"\nThemes with significant sentiment changes: {len(significant_themes)}")
            for theme in sorted(significant_themes, key=lambda x: x['p_value'])[:5]:
                direction = "increased" if theme['difference'] > 0 else "decreased"
                print(f"- {theme['theme']}: {direction} by {abs(theme['difference']):.3f} (p={theme['p_value']:.6f})")
        
        # Largest effect sizes
        large_effects = [r for r in analysis.results if r['p_value'] is not None and abs(r['effect_size']) > 0.2]
        if large_effects:
            print(f"\nThemes with notable effect sizes (>0.2): {len(large_effects)}")
            for theme in sorted(large_effects, key=lambda x: abs(x['effect_size']), reverse=True)[:5]:
                print(f"- {theme['theme']}: Cohen's d = {theme['effect_size']:.3f}")
        
        # Sample size information
        total_posts_2017 = sum([r['n_2017'] for r in analysis.results if r['p_value'] is not None])
        total_posts_2023 = sum([r['n_2023'] for r in analysis.results if r['p_value'] is not None])
        print(f"\nTotal posts analyzed: {total_posts_2017 + total_posts_2023}")
        print(f"2017-2019 period: {total_posts_2017} posts")
        print(f"2023-2024 period: {total_posts_2023} posts")
        
        print("\nAnalysis files have been saved. You can now:")
        print("1. Review the Excel files for detailed results")
        print("2. Use the visualizations to understand the patterns")
        print("3. Access analysis.results for programmatic analysis")
        print("4. Run additional custom analyses using the loaded data")