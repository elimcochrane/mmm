import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
import warnings
warnings.filterwarnings('ignore')

class StanceAnalysis:
    def __init__(self, file_path_2017_2019, file_path_2023_2024):
        """
        Initialize the analysis with two Excel files containing Reddit stance data.
        
        Expected columns:
        - primary_theme: categorical variable (e.g., 'self-help', 'politics_events', etc.)
        - stance: categorical variable ('negative', 'positive', 'neutral')
        """
        self.df_2017_2019 = pd.read_excel(file_path_2017_2019)
        self.df_2023_2024 = pd.read_excel(file_path_2023_2024)
        
        # Add time period labels
        self.df_2017_2019['time_period'] = '2017-2019'
        self.df_2023_2024['time_period'] = '2023-2024'
        
        # Combine datasets
        self.combined_df = pd.concat([self.df_2017_2019, self.df_2023_2024], ignore_index=True)
        
        print("Data loaded successfully!")
        print(f"2017-2019 dataset: {len(self.df_2017_2019)} posts")
        print(f"2023-2024 dataset: {len(self.df_2023_2024)} posts")
        print(f"Combined dataset: {len(self.combined_df)} posts")
    
    def analyze_stance_by_theme(self):
        """Analyze stance distribution changes by theme between time periods."""
        themes = self.combined_df['primary_theme'].unique()
        results = []
        
        for theme in themes:
            # Get data for this theme
            theme_2017 = self.df_2017_2019[self.df_2017_2019['primary_theme'] == theme]
            theme_2023 = self.df_2023_2024[self.df_2023_2024['primary_theme'] == theme]
            
            if len(theme_2017) < 10 or len(theme_2023) < 10:
                continue
            
            # Calculate stance proportions for each time period
            stance_2017 = theme_2017['stance'].value_counts(normalize=True)
            stance_2023 = theme_2023['stance'].value_counts(normalize=True)
            
            # Calculate counts for chi-square test
            stance_counts_2017 = theme_2017['stance'].value_counts()
            stance_counts_2023 = theme_2023['stance'].value_counts()
            
            # Create contingency table
            all_stances = set(stance_counts_2017.index) | set(stance_counts_2023.index)
            contingency_table = []
            
            for stance in ['negative', 'neutral', 'positive']:
                if stance in all_stances:
                    count_2017 = stance_counts_2017.get(stance, 0)
                    count_2023 = stance_counts_2023.get(stance, 0)
                    contingency_table.append([count_2017, count_2023])
            
            # Perform chi-square test
            if len(contingency_table) >= 2:
                contingency_array = np.array(contingency_table)
                try:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_array)
                    
                    # Calculate effect size (Cramer's V)
                    n = contingency_array.sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_array.shape) - 1)))
                    
                    result = {
                        'theme': theme,
                        'n_2017': len(theme_2017),
                        'n_2023': len(theme_2023),
                        'negative_2017': stance_2017.get('negative', 0),
                        'neutral_2017': stance_2017.get('neutral', 0), 
                        'positive_2017': stance_2017.get('positive', 0),
                        'negative_2023': stance_2023.get('negative', 0),
                        'neutral_2023': stance_2023.get('neutral', 0),
                        'positive_2023': stance_2023.get('positive', 0),
                        'negative_diff': stance_2023.get('negative', 0) - stance_2017.get('negative', 0),
                        'neutral_diff': stance_2023.get('neutral', 0) - stance_2017.get('neutral', 0),
                        'positive_diff': stance_2023.get('positive', 0) - stance_2017.get('positive', 0),
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'cramers_v': cramers_v,
                        'significant': p_value < 0.05
                    }
                    
                    results.append(result)
                    
                except ValueError:
                    # Skip if chi-square test fails
                    continue
        
        return results
    
    def analyze_overall_stance_change(self):
        """Analyze overall stance distribution changes across all themes."""
        # Overall stance distributions
        stance_2017 = self.df_2017_2019['stance'].value_counts(normalize=True)
        stance_2023 = self.df_2023_2024['stance'].value_counts(normalize=True)
        
        # Overall counts for chi-square test
        stance_counts_2017 = self.df_2017_2019['stance'].value_counts()
        stance_counts_2023 = self.df_2023_2024['stance'].value_counts()
        
        # Create contingency table
        contingency_table = []
        for stance in ['negative', 'neutral', 'positive']:
            count_2017 = stance_counts_2017.get(stance, 0)
            count_2023 = stance_counts_2023.get(stance, 0)
            contingency_table.append([count_2017, count_2023])
        
        # Perform chi-square test
        contingency_array = np.array(contingency_table)
        chi2, p_value, dof, expected = chi2_contingency(contingency_array)
        
        # Calculate effect size (Cramer's V)
        n = contingency_array.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_array.shape) - 1)))
        
        overall_result = {
            'theme': 'OVERALL',
            'n_2017': len(self.df_2017_2019),
            'n_2023': len(self.df_2023_2024),
            'negative_2017': stance_2017.get('negative', 0),
            'neutral_2017': stance_2017.get('neutral', 0),
            'positive_2017': stance_2017.get('positive', 0),
            'negative_2023': stance_2023.get('negative', 0),
            'neutral_2023': stance_2023.get('neutral', 0),
            'positive_2023': stance_2023.get('positive', 0),
            'negative_diff': stance_2023.get('negative', 0) - stance_2017.get('negative', 0),
            'neutral_diff': stance_2023.get('neutral', 0) - stance_2017.get('neutral', 0),
            'positive_diff': stance_2023.get('positive', 0) - stance_2017.get('positive', 0),
            'chi2_statistic': chi2,
            'p_value': p_value,
            'cramers_v': cramers_v,
            'significant': p_value < 0.05
        }
        
        return overall_result
    
    def run_analysis(self):
        """Run the complete stance analysis."""
        print("\nRunning stance analysis...")
        
        # Analyze by theme
        theme_results = self.analyze_stance_by_theme()
        
        # Analyze overall
        overall_result = self.analyze_overall_stance_change()
        
        # Combine results
        all_results = [overall_result] + theme_results
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Round numeric columns
        numeric_cols = ['negative_2017', 'neutral_2017', 'positive_2017', 
                       'negative_2023', 'neutral_2023', 'positive_2023',
                       'negative_diff', 'neutral_diff', 'positive_diff',
                       'chi2_statistic', 'p_value', 'cramers_v']
        
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = results_df[col].round(4)
        
        # Sort by p-value
        results_df = results_df.sort_values('p_value')
        
        print(f"\nAnalysis complete!")
        print(f"Analyzed {len(theme_results)} themes")
        print(f"Significant changes found: {sum(1 for r in all_results if r['significant'])}")
        
        return results_df
    
    def save_results(self, results_df, filename="stance_analysis_results.xlsx"):
        """Save results to Excel file."""
        # Create a summary sheet
        summary_stats = {
            'Total_Themes_Analyzed': len(results_df) - 1,  # -1 for overall row
            'Significant_Changes': sum(results_df['significant']),
            'Most_Significant_Theme': results_df.iloc[0]['theme'] if len(results_df) > 1 else 'N/A',
            'Lowest_P_Value': results_df.iloc[0]['p_value'] if len(results_df) > 0 else 'N/A',
            'Highest_Effect_Size': results_df['cramers_v'].max(),
            'Average_Effect_Size': results_df['cramers_v'].mean()
        }
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main results
            results_df.to_excel(writer, sheet_name='Stance_Analysis', index=False)
            
            # Summary statistics
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Data overview
            overview_data = {
                'Period': ['2017-2019', '2023-2024'],
                'Total_Posts': [len(self.df_2017_2019), len(self.df_2023_2024)],
                'Negative_Posts': [
                    len(self.df_2017_2019[self.df_2017_2019['stance'] == 'negative']),
                    len(self.df_2023_2024[self.df_2023_2024['stance'] == 'negative'])
                ],
                'Neutral_Posts': [
                    len(self.df_2017_2019[self.df_2017_2019['stance'] == 'neutral']),
                    len(self.df_2023_2024[self.df_2023_2024['stance'] == 'neutral'])
                ],
                'Positive_Posts': [
                    len(self.df_2017_2019[self.df_2017_2019['stance'] == 'positive']),
                    len(self.df_2023_2024[self.df_2023_2024['stance'] == 'positive'])
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Data_Overview', index=False)
        
        print(f"\nResults saved to: {filename}")
        return filename

def main():
    """Main function to run the stance analysis."""
    
    # STEP 1: Update these file paths to your actual Excel files
    file_path_2017_2019 = "C:/Users/elise/Dropbox/Research/results/results_2017-2019/enhanced_analysis_2017-2019.xlsx"
    file_path_2023_2024 = "C:/Users/elise/Dropbox/Research/results/results_2023-2024/enhanced_analysis_2023-2024.xlsx"
    
    try:
        # STEP 2: Initialize and run analysis
        analysis = StanceAnalysis(file_path_2017_2019, file_path_2023_2024)
        results_df = analysis.run_analysis()
        
        # STEP 3: Display key results
        print("\n" + "="*80)
        print("STANCE ANALYSIS RESULTS")
        print("="*80)
        
        # Show top 10 most significant results
        print("\nTop 10 Most Significant Changes:")
        display_cols = ['theme', 'negative_diff', 'neutral_diff', 'positive_diff', 
                       'p_value', 'cramers_v', 'significant']
        print(results_df[display_cols].head(10).to_string(index=False))
        
        # STEP 4: Save results
        filename = analysis.save_results(results_df)
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {filename}")
        print("\nThe Excel file contains 3 sheets:")
        print("1. Stance_Analysis - Detailed results by theme")
        print("2. Summary - Key statistics")
        print("3. Data_Overview - Basic data information")
        
        return analysis, results_df
        
    except FileNotFoundError:
        print(f"Error: Could not find the Excel files.")
        print(f"Please check the file paths:")
        print(f"- {file_path_2017_2019}")
        print(f"- {file_path_2023_2024}")
        return None, None
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

if __name__ == "__main__":
    analysis, results = main()