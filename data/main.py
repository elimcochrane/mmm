import json
import pandas as pd
from datetime import datetime
import os

def process_reddit_chunks(input_file, output_file, chunk_size=1000):
    """
    Processes large Reddit JSON files in chunks with memory efficiency.
    Includes: id, title, author, date, score, num_comments, url, selftext, is_self, media
    
    Args:
        input_file: Path to input JSON file
        output_file: Path for output CSV file
        chunk_size: Number of posts to process at once
    """
    # Initialize variables
    chunk_list = []
    processed_count = 0

    # Process file in chunks
    with open(input_file, 'r', encoding='utf-8') as f:
        current_chunk = []
        for line in f:
            if line.strip():
                try:
                    post = json.loads(line)

                    # Parse media information
                    media = post.get("media")
                    media_str = ""
                    if media:
                        if isinstance(media, dict):
                            media_type = media.get('type', 'unknown')
                            media_url = media.get('url', '')
                            if 'oembed' in media:
                                media_url = media['oembed'].get('url', media_url)
                            media_str = f"{media_type}: {media_url}"
                        else:
                            media_str = str(media)

                    # Create post dictionary
                    processed_post = {
                        "id": post.get("id", ""),
                        "title": post.get("title", "[No Title]"),
                        "author": post.get("author", "[Deleted]"),
                        "date": datetime.utcfromtimestamp(post.get("created_utc", 0)).strftime('%m/%d/%Y'),
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "url": post.get("url", ""),
                        "selftext": post.get("selftext", ""),
                        "is_self": post.get("is_self", False),
                        "media": media_str
                    }

                    current_chunk.append(processed_post)
                    processed_count += 1

                    # Process chunk when size reached
                    if len(current_chunk) >= chunk_size:
                        chunk_list.append(pd.DataFrame(current_chunk))
                        current_chunk = []
                        print(f"Processed {processed_count} posts...")

                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line: {e}")

        # Add final chunk
        if current_chunk:
            chunk_list.append(pd.DataFrame(current_chunk))

    # Combine all chunks
    if chunk_list:
        full_df = pd.concat(chunk_list, ignore_index=True)

        # Ensure column order
        columns = [
            "id", "title", "selftext", "author", "date", "score",
            "num_comments", "url", "is_self", "media"
        ]
        full_df = full_df[columns]

        # Save to CSV
        full_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\nSuccessfully processed {len(full_df)} posts")
        print(f"Saved to: {os.path.abspath(output_file)}")
        print("\nSample output:")
        return full_df.head(3)
    else:
        print("No valid posts found in the file")
        return None

# Simplified output to working directory
output_filename = "jpeterson_posts_processed.csv"

# Process the file and save to current directory
result_df = process_reddit_chunks(
    input_file='JordanPeterson_submissions_2024',
    output_file=output_filename,
    chunk_size=5000
)