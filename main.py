import praw
import pandas as pd
import time
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


# --------------------- CONSTANCE VARIABLES -----------------------------
CLIENT_ID = ''
CLIENT_SECRET = ''
USER_AGENT = 'Projekt: Data Analysis'
STANDARD_AUTHORS = ['unknown', 'AutoModerator']

# # -------------------- DOWNLOAD NLTK PACKAGES ---------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



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



def process(text):
  # Replace NaN values with empty strings
  if pd.isna(text):
    return ""
  
  text = text.lower()

  text = re.sub(r'http\S+', '', text)
  # remove literal \n
  # text = text.replace('\n', '')
  tokens = word_tokenize(text)

  stop_words_en = set(stopwords.words('english'))
  stop_words_de = set(stopwords.words('german'))
  all_stopwords = stop_words_en.union(stop_words_de)

  # 3.6 NLTK Book
  wnl = nltk.WordNetLemmatizer()
  

  # clean_tokens = [w for w in tokens if w.isalpha() and w not in all_stopwords]
  clean_tokens = [wnl.lemmatize(w) for w in tokens if w.isalpha() and w not in all_stopwords]
  clean_tokens_string = " ".join(clean_tokens)


  return clean_tokens_string



def main():
  munich_data_path = '/home/user/Development/Projekt_Data_Analysis/Reddit/munich_reddit_data.csv'
  output_debug_path = '/home/user/Development/Projekt_Data_Analysis/Reddit/debug.csv'

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
  # print(munich_data_df['full_text'])

  munich_data_df['clean_text'] = munich_data_df['full_text'].apply(process)
  # print(munich_data_df['clean_text'])

  output_debug = munich_data_df
  output_debug.to_csv(output_debug_path, index=False, encoding='utf-8-sig')

  # Extract most active users
  authors = munich_data_df['author'].dropna().tolist()
  # print(authors)
  filtered_authors = [a for a in authors if a not in STANDARD_AUTHORS]
  
  author_counts = Counter(filtered_authors)
  most_active_authors = author_counts.most_common(10)
  # print(most_active_authors)


  # Extract most used flairs
  flairs = munich_data_df['flair'].dropna().tolist()
  flair_counts = Counter(flairs)
  most_common_flairs = flair_counts.most_common(10)
  print(most_common_flairs)



  return munich_data_df


if __name__=='__main__':
  result = main()

