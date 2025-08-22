# Currently, this creates a an Excel sheet of the 200 most recent posts and the top 10 comments on each 

import praw
import pandas as pd
from datetime import datetime, timedelta, timezone  # Added timezone and timedelta
import time

# Reddit API Setup
reddit = praw.Reddit(
    client_id='RreVS2U0-eL-2-e7od9UUw',        # Replace with your credentials
    client_secret='mP_cEbML3zMX_HyXJNATx79Ud6zELg',
    user_agent='windows:webscraper:v1.0 (by /u/Hyaena113)'
)

def scrape_subreddit(subreddit_name, start_date, end_date):
    print(f"Scraping r/{subreddit_name} from {start_date} to {end_date}")
    
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    
    # Convert dates to timestamps
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    
    try:
        # Get recent posts (adjust limit as needed)
        for submission in subreddit.new(limit=200):  # Start with 200 posts
            post_date = datetime.fromtimestamp(submission.created_utc, timezone.utc)
            
            # Skip if outside date range
            if not (start_ts <= submission.created_utc <= end_ts):
                continue
                
            print(f"Processing: {submission.title[:50]}...")
            
            # Get top 10 comments (adjust as needed)
            submission.comments.replace_more(limit=0)
            comments = [comment.body for comment in submission.comments[:10]]
            
            posts_data.append({
                'Title': submission.title,
                'Content': submission.selftext,
                'Date': post_date.strftime('%Y-%m-%d'),
                'Time': post_date.strftime('%H:%M:%S'),
                'Comment_1': comments[0] if len(comments) > 0 else '',
                'Comment_2': comments[1] if len(comments) > 1 else '',
                'Comment_3': comments[2] if len(comments) > 2 else '',
                'Comment_4': comments[3] if len(comments) > 3 else '',
                'Comment_5': comments[4] if len(comments) > 4 else '',
                'Comment_6': comments[5] if len(comments) > 5 else '',
                'Comment_7': comments[6] if len(comments) > 6 else '',
                'Comment_8': comments[7] if len(comments) > 7 else '',
                'Comment_9': comments[8] if len(comments) > 8 else '',
                'Comment_10': comments[9] if len(comments) > 9 else ''
            })
            
            time.sleep(1)  # Rate limiting
            
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return posts_data

# Test with recent dates first (last 30 days) - FIXED version
start_date = datetime.now(timezone.utc) - timedelta(days=30)  # Correct timezone-aware
end_date = datetime.now(timezone.utc)

print("Starting scrape...")
data = scrape_subreddit('JordanPeterson', start_date, end_date)

if data:
    df = pd.DataFrame(data)
    df.to_excel('reddit_data_praw.xlsx', index=False)
    print(f"✅ Saved {len(df)} posts to reddit_data_praw.xlsx")
else:
    print("❌ No posts found. Try adjusting the date range.")

input("Press Enter to exit...")