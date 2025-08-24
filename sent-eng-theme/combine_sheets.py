import pandas as pd

def merge_excel_sheets(file1_path, file2_path, output_path):
    """
    merge two excel sheets with same rows but different columns
    had to do this after running sentiment and theme classifications
    """
    
    df1 = pd.read_excel(file1_path)
    df2 = pd.read_excel(file2_path)
    
    print(f"Sheet 1 shape: {df1.shape}")
    print(f"Sheet 2 shape: {df2.shape}")
    
    # just keep sentiment stuff from df2
    columns_to_keep = ['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 
                       'sentiment_compound', 'sentiment_classification']
    available_columns = [col for col in columns_to_keep if col in df2.columns]
    df2_filtered = df2[available_columns]
    print(f"Keeping columns from file 2: {available_columns}")
    
    # merge based on index position (assumes same row order)
    merged_df = pd.concat([df1, df2_filtered], axis=1)
    print("Merged based on row position")
    
    print(f"Merged shape: {merged_df.shape}")
    
    # save
    merged_df.to_excel(output_path, index=False)
    print(f"Merged file saved as: {output_path}")
    
    return merged_df

# run + customize
if __name__ == "__main__":
    merged = merge_excel_sheets(
        file1_path="theme_classification.xlsx", # must be theme
        file2_path="sentiment_classification_results.xlsx", # must be sentiment
        output_path="merged_output.xlsx"
    )