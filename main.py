import praw
import pandas as pd


# --------------------- CONSTANCE VARIABLES -----------------------------
CLIENT_ID = ''
CLIENT_SECRET = ''
USER_AGENT = 'Projekt: Data Analysis'


def getDataFromReddit(subreddit):
  data = []

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

    if post.num_comments > 0:
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
  
  munich_data = pd.DataFrame(data)
  return munich_data



def main():
  reddit = praw.Reddit(client_id=CLIENT_ID,
                       client_secret=CLIENT_SECRET,
                       user_agent=USER_AGENT)

  subreddit = reddit.subreddit('munich')

  

  munich_data = getDataFromReddit(subreddit)
  munich_data.to_csv('munich_reddit_data.csv', index=False, encoding='utf-8-sig')

  return munich_data


if __name__=='__main__':
  result = main()


print(result)
