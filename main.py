import praw
import pandas as pd
import time
import os

# --------------------- CONSTANCE VARIABLES -----------------------------
CLIENT_ID = ''
CLIENT_SECRET = ''
USER_AGENT = 'Projekt: Data Analysis'



def getDataFromReddit(subreddit):
  data = []

  # Scraping posts and Comments
  for post in subreddit.top(time_filter='year', limit= 10): 
    data.append({
        'post_id': post.id,
        'type': 'post',
        'author': str(post.author),
        'timestamp_utc': post.created_utc,
        'title': post.title,
        'text': post.selftext,
        'flair': str(post.link_flair_text),
        'score': post.score,
        'url': post.url
    })

    # Check if the post has comments
    if post.num_comments > 0:
      # Scraping comments for each post
      post.comments.replace_more(limit=5)
      for comment in post.comments.list():
        data.append({
          'post_id': post.id,
          'type': 'comment',
          'author': comment.author.name if comment.author else 'unknown',
          'timestamp_utc': pd.to_datetime(comment.created_utc, unit='s'),
          'title': post.title,
          'text': comment.body,
          'flair': str(post.link_flair_text),
          'score': comment.score,
          'total_comments': 0,
          'Post_URL': None
        })
        # time.sleep(2)
    time.sleep(2)
  
  munich_data = pd.DataFrame(data)
  return munich_data


def main():
  munich_data_path = '/home/user/Development/Projekt_Data_Analysis/Reddit/munich_reddit_data.csv'

  # Only if munich_reddit_data.csv is not present, extract data from Reddit via PRAW
  # and create/write it to munich_reddit_data.csv
  if not os.path.exists(munich_data_path):
    print('PRAW initialized')
    # Initialize Reddit instance
    reddit = praw.Reddit(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        user_agent=USER_AGENT)

    # Subreddit to scrape
    subreddit = reddit.subreddit('munich')
    # subreddit = reddit.subreddit('fitness')
    

    munich_data = getDataFromReddit(subreddit)
    # Write data to munich_reddit_data.csv
    munich_data.to_csv(munich_data_path, index=False, encoding='utf-8-sig')

  # Read Reddit-Data from csv-file
  munich_data_df = pd.read_csv(munich_data_path)
  # print(munich_data_df[['title', 'text']])

  # Replace NaN values with empty strings
  munich_data_df['title'] = munich_data_df['title'].fillna('')
  munich_data_df['text'] = munich_data_df['text'].fillna('')

  # Combine title and text for theme examination
  munich_data_df['full_text'] = munich_data_df['title'] + " " + munich_data_df['text']
  print(munich_data_df['full_text'])



  return munich_data_df


if __name__=='__main__':
  result = main()


# print(result)
