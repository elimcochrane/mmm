import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict, Counter
import json

class ThemeClassifier:
    def __init__(self, custom_themes, method="hybrid"):
        self.custom_themes = custom_themes
        self.method = method
        self.embedding_model = None
        self.theme_embeddings = {}
        
        if method in ["semantic", "hybrid"]:
            print("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._compute_theme_embeddings()
        
        print(f"ThemeClassifier initialized with {len(custom_themes)} themes using {method} method")
        print(f"Themes: {list(custom_themes.keys())}")
    
    def _compute_theme_embeddings(self):
        for theme_name, keywords in self.custom_themes.items():
            theme_text = " ".join(keywords)
            embedding = self.embedding_model.encode([theme_text])[0]
            self.theme_embeddings[theme_name] = embedding
    
    def preprocess_texts(self, texts):
        clean_texts = []
        for i, text in enumerate(texts):
            if isinstance(text, str) and len(text.strip()) > 0 and text.lower() not in ['nan', 'none']:
                clean_texts.append(text.strip().lower())
            else:
                placeholder = f"empty post {i}"
                clean_texts.append(placeholder)
        
        return clean_texts
    
    def assign_themes_keyword_based(self, texts):
        primary_themes = []
        secondary_themes = []
        primary_scores = []
        secondary_scores = []
        
        for text in texts:
            scores = defaultdict(int)
            
            for theme_name, keywords in self.custom_themes.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text.lower()):
                        scores[theme_name] += 1
            
            if scores:
                sorted_themes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                primary_themes.append(sorted_themes[0][0])
                primary_scores.append(sorted_themes[0][1])
                
                if len(sorted_themes) > 1 and sorted_themes[1][1] > 0:
                    secondary_themes.append(sorted_themes[1][0])
                    secondary_scores.append(sorted_themes[1][1])
                else:
                    secondary_themes.append(None)
                    secondary_scores.append(0)
            else:
                primary_themes.append("Other")
                secondary_themes.append(None)
                primary_scores.append(0)
                secondary_scores.append(0)
        
        return primary_themes, secondary_themes, primary_scores, secondary_scores
    
    def assign_themes_semantic_based(self, texts):
        print(f"Computing embeddings for {len(texts)} texts using batch processing...")
        
        # process in batches
        batch_size = 1000
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_texts = texts[i:batch_end]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} "
                  f"(texts {i+1}-{batch_end})")
            
            batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        # combine embeddings
        text_embeddings = np.vstack(all_embeddings)
        print(f"Completed embedding computation. Shape: {text_embeddings.shape}")
        
        primary_themes = []
        secondary_themes = []
        primary_scores = []
        secondary_scores = []
        
        theme_list = list(self.custom_themes.keys())
        theme_embedding_matrix = np.array([self.theme_embeddings[theme] for theme in theme_list])
        
        print("Computing theme similarities...")
        # process similarities in batches too
        for i in range(0, len(text_embeddings), batch_size):
            batch_end = min(i + batch_size, len(text_embeddings))
            batch_embeddings = text_embeddings[i:batch_end]
            
            if i % (batch_size * 10) == 0:  # print progress every 10 batches
                print(f"Processing similarities: {i+1}-{batch_end}/{len(text_embeddings)}")
            
            # compute similarities for this batch
            batch_similarities = cosine_similarity(batch_embeddings, theme_embedding_matrix)
            
            for similarities in batch_similarities:
                top_2_indices = np.argsort(similarities)[-2:][::-1]
                
                primary_idx = top_2_indices[0]
                primary_sim = similarities[primary_idx]
                
                if primary_sim > 0.3:  # similarity threshold
                    primary_themes.append(theme_list[primary_idx])
                    primary_scores.append(round(primary_sim, 3))
                else:
                    primary_themes.append("Other")
                    primary_scores.append(0)
                
                if len(top_2_indices) > 1:
                    secondary_idx = top_2_indices[1]
                    secondary_sim = similarities[secondary_idx]
                    
                    if secondary_sim > 0.3 and secondary_idx != primary_idx:
                        secondary_themes.append(theme_list[secondary_idx])
                        secondary_scores.append(round(secondary_sim, 3))
                    else:
                        secondary_themes.append(None)
                        secondary_scores.append(0)
                else:
                    secondary_themes.append(None)
                    secondary_scores.append(0)
        
        print("Semantic similarity computation completed!")
        return primary_themes, secondary_themes, primary_scores, secondary_scores
    
    def assign_themes_hybrid(self, texts):
        print("Running hybrid classification (keyword + semantic)...")
        
        # run keyword-based first
        print("Step 1/2: Keyword-based classification...")
        kw_primary, kw_secondary, kw_p_scores, kw_s_scores = self.assign_themes_keyword_based(texts)
        
        # run semantic-based second (slower, with batching)
        print("Step 2/2: Semantic-based classification...")
        sem_primary, sem_secondary, sem_p_scores, sem_s_scores = self.assign_themes_semantic_based(texts)
        
        # combine results
        print("Combining keyword and semantic results...")
        final_primary = []
        final_secondary = []
        final_p_scores = []
        final_s_scores = []
        
        for i in range(len(texts)):
            combined_scores = defaultdict(float)
            
            # add keyword scores (weighted)
            if kw_primary[i] != "Other":
                combined_scores[kw_primary[i]] += kw_p_scores[i] * 0.1
            if kw_secondary[i] is not None:
                combined_scores[kw_secondary[i]] += kw_s_scores[i] * 0.1
            
            # add semantic scores
            if sem_primary[i] != "Other":
                combined_scores[sem_primary[i]] += sem_p_scores[i]
            if sem_secondary[i] is not None:
                combined_scores[sem_secondary[i]] += sem_s_scores[i]
            
            if combined_scores:
                sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                
                final_primary.append(sorted_combined[0][0])
                final_p_scores.append(round(sorted_combined[0][1], 3))
                
                if len(sorted_combined) > 1 and sorted_combined[1][1] > 0:
                    final_secondary.append(sorted_combined[1][0])
                    final_s_scores.append(round(sorted_combined[1][1], 3))
                else:
                    final_secondary.append(None)
                    final_s_scores.append(0)
            else:
                final_primary.append("Other")
                final_secondary.append(None)
                final_p_scores.append(0)
                final_s_scores.append(0)
        
        print("Hybrid classification completed")
        return final_primary, final_secondary, final_p_scores, final_s_scores
    
    def classify_posts(self, df, text_columns):
        print(f"\nClassifying {len(df)} posts by theme")
        print("-" * 40)
        
        # combine text columns
        combined_texts = []
        for _, row in df.iterrows():
            text_parts = []
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            combined_text = " ".join(text_parts)
            combined_texts.append(combined_text)
        
        clean_texts = self.preprocess_texts(combined_texts)
        
        if self.method == "keyword":
            primary_themes, secondary_themes, primary_scores, secondary_scores = self.assign_themes_keyword_based(clean_texts)
        elif self.method == "semantic":
            primary_themes, secondary_themes, primary_scores, secondary_scores = self.assign_themes_semantic_based(clean_texts)
        elif self.method == "hybrid":
            primary_themes, secondary_themes, primary_scores, secondary_scores = self.assign_themes_hybrid(clean_texts)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        df_result = df.copy()
        df_result['combined_text'] = combined_texts
        df_result['primary_theme'] = primary_themes
        df_result['secondary_theme'] = secondary_themes
        df_result['primary_theme_score'] = primary_scores
        df_result['secondary_theme_score'] = secondary_scores
        
        self._print_theme_distribution(primary_themes, secondary_themes)
        
        return df_result
    
    def _print_theme_distribution(self, primary_themes, secondary_themes):
        print("\nTheme Distribution:")
        print("-" * 20)
        
        primary_dist = Counter(primary_themes)
        print("Primary themes:")
        for theme, count in primary_dist.most_common():
            percentage = (count / len(primary_themes)) * 100
            print(f"  {theme}: {count} ({percentage:.1f}%)")
        
        secondary_clean = [theme for theme in secondary_themes if theme is not None]
        if secondary_clean:
            print("\nSecondary themes:")
            secondary_dist = Counter(secondary_clean)
            for theme, count in secondary_dist.most_common():
                percentage = (count / len(secondary_themes)) * 100
                print(f"  {theme}: {count} ({percentage:.1f}%)")

def load_data(file_path, required_columns):
    try:
        print(f"Loading data from: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def run_theme_classification(config):
    try:
        print("STARTING THEME CLASSIFICATION PIPELINE")
        print("=" * 50)
        
        df = load_data(config['input_file'], config['text_columns'])
        
        classifier = ThemeClassifier(
            custom_themes=config['custom_themes'],
            method=config.get('method', 'hybrid')
        )
        
        classified_df = classifier.classify_posts(df, config['text_columns'])
        
        config['output_dir'].mkdir(exist_ok=True)
        
        # results
        output_file = config['output_dir'] / 'theme_classification.xlsx'
        classified_df.to_excel(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        primary_dist = Counter(classified_df['primary_theme'])
        secondary_dist = Counter([theme for theme in classified_df['secondary_theme'] if theme is not None])
        
        summary = {
            'total_posts': len(classified_df),
            'classification_method': config.get('method', 'hybrid'),
            'themes_defined': list(config['custom_themes'].keys()),
            'primary_theme_distribution': dict(primary_dist),
            'secondary_theme_distribution': dict(secondary_dist),
            'posts_with_secondary_theme': sum(1 for theme in classified_df['secondary_theme'] if theme is not None)
        }
        
        summary_file = config['output_dir'] / 'classification_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file}")
        
        print("\n" + "=" * 50)
        print("Theme Classification Completed")
        print("=" * 50)
        print(f"✓ Classified {len(classified_df)} posts")
        print(f"✓ Found {len(primary_dist)} unique primary themes")
        print(f"✓ Method used: {config.get('method', 'hybrid')}")
        print(f"✓ Output files:")
        print(f"  - {output_file.name}")
        print(f"  - {summary_file.name}")
        
        # print top themes
        print(f"\nTop Primary Themes:")
        for theme, count in primary_dist.most_common(5):
            percentage = (count / len(classified_df)) * 100
            print(f"  {theme}: {count} posts ({percentage:.1f}%)")
        
        return classified_df
        
    except Exception as e:
        print(f"Error in theme classification pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    custom_themes = {
        'masculinity': ['men', 'male', 'masculinity', 'masculine', 'man', 'father', 'patriarchy', 'boy', 'boys'],
        'politics': ['government', 'politics', 'trump', 'biden', 'maga', 'left', 'right', 'liberal', 'conservative', 'lefty', 'racism', 'woke', 'sexism'],
        'peterson': ['peterson', 'jbp', 'jordan', 'lobster'],
        'socialsciences': ['philosophy', 'religion', 'christianity', 'psychology', 'personality', 'behavior'],
        'humanities': ['art', 'music', 'history', 'book', 'reading', 'author', 'authoring'],
        'women': ['woman', 'women', 'girls', 'girl', 'lady'],
        'lgbtq': ['lgbtq', 'lgbt', 'trans', 'transgender', 'gay', 'lesbian', 'nonbinary', 'bisexual'],
        'self-help': ['help', 'helped', 'helps', 'confidence', 'strong', 'strength', 'gym', 'helping', 'self-help', 'self esteem', 'esteem', 'confident'],
    }
    
    config = {
        'text_columns': ['title', 'selftext'],
        'input_file': Path('peterson_2016-2024_cleaned.xlsx'),
        'output_dir': Path('theme_results'),
        'custom_themes': custom_themes,
        'method': 'hybrid' # 'keyword', 'semantic', or 'hybrid'
    }
    
    result_df = run_theme_classification(config)