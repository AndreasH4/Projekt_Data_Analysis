import praw
import pandas as pd
import time
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
from dotenv import load_dotenv


# --------------------- CONSTANCE VARIABLES -----------------------------
# CLIENT_ID = ''
# CLIENT_SECRET = ''
USER_AGENT = 'Projekt: Data Analysis'
STANDARD_AUTHORS = ['unknown', 'AutoModerator']
N_TOP_WORDS = 20
MUNICH_DATA_PATH = 'munich_reddit_data.csv'
OUTPUT_DEBUG_PATH = 'debug.csv'
LDA_RESULTS_JSON_PATH = 'lda_results.json'
TOPICS_LDA_PNG = 'Figures/topics_lda.png'

# # -------------------- DOWNLOAD NLTK PACKAGES ---------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# https://www.kdnuggets.com/managing-secrets-and-api-keys-in-python-projects-env-guide
def getSecret(secret_name):
  load_dotenv()
  secret = os.getenv(secret_name)
  return secret


def getDataFromReddit(subreddit):
  data = []

  # Scraping posts and Comments
  for post in subreddit.top(time_filter='year', limit= 30): 
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
      # Scraping only top-level comments for each post
      post.comments.replace_more(limit=0)
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


# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def print_top_words(model, feature_names, n_top_words):
  # https://stackoverflow.com/questions/28669482/appending-pandas-dataframes-generated-in-a-for-loop
  # https://stackoverflow.com/questions/30635145/create-multiple-dataframes-in-loop
  lda_results = {}

  for topic_idx, topic in enumerate(model.components_):
    topic_title_idx = f'Thema {topic_idx+1}'
    top_features_ind = topic.argsort()[-n_top_words:]
    # https://statistikguru.de/python/python-listen-rueckwaerts.html
    top_features_ind_rev = top_features_ind[::-1]
    top_features = feature_names[top_features_ind_rev]
    # print(f'top_features: {top_features}')
    weights = topic[top_features_ind_rev]

    # https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    top_features_list = top_features.tolist()
    weights_list = weights.tolist()

    # https://stackoverflow.com/questions/209840/make-a-dictionary-dict-from-separate-lists-of-keys-and-values
    topic_dict = dict(zip(top_features_list, weights_list))

    lda_results[topic_title_idx] = topic_dict

    # df_topic_words = pd.DataFrame({
    #   'Word': top_features,
    #   'Weight': weights
    # })
    # print(topic_title_idx)
    # print(df_topic_words)
    # # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html
    # data[topic_title_idx] = df_topic_words.to_dict()
    # appended_data.append(df_topic_words)
  # appended_data = pd.concat(appended_data)
  return lda_results




# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 5, figsize=(30, 8), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        # https://statistikguru.de/python/python-listen-rueckwaerts.html
        top_features_ind_rev = top_features_ind[::-1]
        top_features = feature_names[top_features_ind_rev]

        # print(f'top_features: {top_features}')
        weights = topic[top_features_ind_rev]

        ax = axes[topic_idx]

        # https://seaborn.pydata.org/generated/seaborn.barplot.html
        sns.barplot(
          x=weights,
          y=top_features,
          ax=ax
        )

        ax.set_title(f"Thema {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.set_xlabel("Gewichtung (TF-IDF)", fontsize=20)

        if topic_idx == 0:
          ax.set_ylabel("Wörter", fontsize=20)

        ax.tick_params(axis="both", which="major", labelsize=20)
        # for i in "top right left".split():
        #     ax.spines[i].set_visible(False)

        # https://seaborn.pydata.org/generated/seaborn.despine.html
        # https://medium.com/@tttgm/styling-charts-in-seaborn-92136331a541
        sns.despine(ax=ax, left=True, bottom=True)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.80, bottom=0.15, wspace=0.90, hspace=0.3)
    plt.savefig(TOPICS_LDA_PNG)
    plt.show()



def main():

  # https://www.kdnuggets.com/managing-secrets-and-api-keys-in-python-projects-env-guide
  client_id = getSecret('CLIENT_ID')
  # print(f'Retrieved Secret CLIENT_ID:\n{client_id}\n')
  client_secret = getSecret('CLIENT_SECRET')
  # print(f'Retrieved Secret CLIENT_SECRET:\n{client_id}\n')

  if not client_id:
    print('No client_id was found')
    return

  if not client_secret:
    print('No client_secret was found')
    return

  # Guard Clause: Only if munich_reddit_data.csv is not present, extract data from Reddit via PRAW
  # and create/write it to munich_reddit_data.csv
  if not os.path.exists(MUNICH_DATA_PATH):
    print('PRAW initialized')
    # Initialize Reddit instance
    reddit = praw.Reddit(client_id=client_id,
                        client_secret=client_secret,
                        user_agent=USER_AGENT)

    # Subreddit to scrape
    subreddit = reddit.subreddit('munich')
    # subreddit = reddit.subreddit('fitness')
    

    munich_data = getDataFromReddit(subreddit)
    # Write data to munich_reddit_data.csv
    munich_data.to_csv(MUNICH_DATA_PATH, index=False, encoding='utf-8-sig')

  # Read Reddit-Data from csv-file
  munich_data_df = pd.read_csv(MUNICH_DATA_PATH)
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
  output_debug.to_csv(OUTPUT_DEBUG_PATH, index=False, encoding='utf-8-sig')

  # Extract most active users
  authors = munich_data_df['author'].dropna().tolist()
  # print(authors)
  filtered_authors = [a for a in authors if a not in STANDARD_AUTHORS]
  author_counts = Counter(filtered_authors)
  most_active_authors = author_counts.most_common(10)
  print(f'most_active_authors: {most_active_authors}')

  # Extract most used flairs
  flairs = munich_data_df['flair'].dropna().tolist()
  flair_counts = Counter(flairs)
  most_common_flairs = flair_counts.most_common(10)
  print(f'most_common_flairs: {most_common_flairs}')

  text_corpus = munich_data_df[munich_data_df['clean_text'].str.strip() != '']['clean_text'].tolist()
  # print(text_corpus)

  # TF-IDF
  # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
  vectorizer = TfidfVectorizer(use_idf=True, max_features=1000, smooth_idf=True)
  model = vectorizer.fit_transform(text_corpus)
  feature_names = vectorizer.get_feature_names_out()
  # print(model)

  # Latent Dirichlet Allocation (LDA)
  # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html 
  lda_model = LatentDirichletAllocation(n_components=5, learning_method='online', random_state=42, max_iter=20)
  lda_top = lda_model.fit_transform(model)
  # print(lda_top)
  
  for i, topic in enumerate(lda_top[0]):
    print("Topic ",i+1,": ",topic*100,"%")

  # dfs_top_words_result = dfs_top_words(lda_model, feature_names, N_TOP_WORDS)
  # print(f'\ndfs_top_words:\n{dfs_top_words_result}\n')
  # dfs_top_words_result.to_csv('/home/user/Development/Projekt_Data_Analysis/Reddit/output.csv', index=False, encoding='utf-8-sig')
  lda_results = print_top_words(lda_model, feature_names, N_TOP_WORDS)
  print(lda_results)

  # Write results as json to 
  # https://www.geeksforgeeks.org/python/write-multiple-variables-to-a-file-using-python/
  # https://www.geeksforgeeks.org/python/how-to-convert-python-dictionary-to-json/
  with open(LDA_RESULTS_JSON_PATH, 'w') as file:
    json.dump(lda_results, file, ensure_ascii=False, indent=2)
  

  # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#gallery-examples
  # https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
  plot_top_words(lda_model, feature_names, N_TOP_WORDS, "Themen nach Latent Dirichlet Allocation (LDA)")

  return munich_data_df


if __name__=='__main__':
  result = main()

