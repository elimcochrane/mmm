# This one is similar to main.py (uses praw) but does it month by month for processing 

import praw
import pandas as pd
from datetime import datetime, timezone, timedelta
import time

# Initialize PRAW with your credentials
reddit = praw.Reddit(
    client_id='RreVS2U0-eL-2-e7od9UUw',
    client_secret='mP_cEbML3zMX_HyXJNATx79Ud6zELg',
    user_agent='windows:webscraper:v1.0 (by /u/Hyaena113)'
)

def scrape_by_month(subreddit_name, start_year, start_month, end_year, end_month):
    current_date = datetime(start_year, start_month, 1, tzinfo=timezone.utc)
    end_date = datetime(end_year, end_month, 1, tzinfo=timezone.utc)
    
    all_data = []
    
    while current_date <= end_date:
        next_month = current_date.replace(day=28) + timedelta(days=4)  # Move to next month
        next_month = next_month.replace(day=1)
        
        print(f"\nScraping {current_date.strftime('%Y-%m')}...")
        
        # Get posts from this month
        posts = list(reddit.subreddit(subreddit_name).search(
            query=f'timestamp:{int(current_date.timestamp())}..{int(next_month.timestamp())}',
            limit=1000,
            sort='new'
        ))
        
        for post in posts:
            # Skip posts without text content
            if not post.selftext or len(post.selftext.strip()) < 100:
                continue
                
            print(f"Processing: {post.title[:50]}...")
            
            # Get all comments
            post.comments.replace_more(limit=None)
            comments = [c.body for c in post.comments.list()]
            
            # Store data
            all_data.append({
                'Title': post.title,
                'Content': post.selftext,
                'Date': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
                'Time': datetime.fromtimestamp(post.created_utc).strftime('%H:%M:%S'),
                'Total_Comments': len(comments),
                **{f'Comment_{i+1}': comment for i, comment in enumerate(comments[:50])}  # First 50 comments
            })
            
            time.sleep(1)  # Rate limiting
        
        current_date = next_month
    
    return all_data

# Run scraper for Jan 2017 - Jan 2019
print("Starting historical scrape...")
data = scrape_by_month('JordanPeterson', 2017, 1, 2019, 1)

if data:
    df = pd.DataFrame(data)
    df.to_excel('jpeterson_historical_praw.xlsx', index=False)
    print(f"\n✅ Saved {len(df)} posts with up to {df['Total_Comments'].max()} comments each")
else:
    print("\n❌ No data collected. Possible reasons:")
    print("- Reddit's search API isn't returning historical data")
    print("- Try smaller date ranges (e.g. 3 months at a time)")

print("\nScrape completed.")