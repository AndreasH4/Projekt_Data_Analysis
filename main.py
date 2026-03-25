import praw
import pandas as pd
import time
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
from dotenv import load_dotenv
import numpy as np
from HanTa import HanoverTagger as ht
from langdetect import detect, LangDetectException


# ---------------------------- CONSTANTS --------------------------------
USER_AGENT = 'Projekt: Data Analysis'
STANDARD_AUTHORS = ['unknown', 'AutoModerator', '[deleted]']
N_TOP_WORDS = 20
MUNICH_DATA_PATH = 'munich_reddit_data.csv'

# INPUTS
CUSTOM_STOPWORDS_PATH = 'Inputs/custom_stopwords.json'

# OUTPUTS
OUTPUT_DEBUG_PATH = 'Outputs/debug.csv'
LDA_RESULTS_JSON_PATH = 'Outputs/lda_results.json'
MOST_ACTIVE_AUTHORS_CSV_PATH = 'Outputs/most_active_authors.csv'
MOST_COMMON_FLAIRS_CSV_PATH = 'Outputs/most_common_flairs.csv'
UNIQUE_AUTHORS_CSV_PATH = 'Outputs/unique_authors.csv'
UNIQUE_FLAIRS_CSV_PATH = 'Outputs/unique_flairs.csv'


# -------------------- DOWNLOAD NLTK PACKAGES ---------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def get_secret(secret_name):
  """
  Retrieve secret from .env file
  """
  load_dotenv()
  secret = os.getenv(secret_name)
  return secret


def get_custom_stopwords(path):
  """
  Get custom stopwords from stopwords-file provided by path
  """
  try:
    with open(path, 'r', encoding='utf-8') as file:
      custom_stopwords = json.loads(file.read())
    return custom_stopwords
  except FileNotFoundError as e:
    return f'File not found: {e}'
  except PermissionError as e:
    return f'Permission denied: {e}'



def join_post_id_text(text):
  """
  Join text
  """
  return ' '.join(text)



def get_data_from_reddit(subreddit):
  """
  Retrieve posts and top-level comments from specified subreddit
  """

  data = []
  # Scraping posts and Comments
  for post in subreddit.top(time_filter='year', limit=None): 
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



def detect_language(text):
  """
  Detect language of text
  """
  if pd.isna(text) or not str(text).strip():
    return 'empty_text'
  try:
    return detect(text)
  except LangDetectException:
    return 'unknown'



def determine_language_by_source(text, is_title, clean_text):
  """
  Determine based on is_title, which source for
  language detection should be chosen
  """
  if not is_title:
    return detect_language(text)
  
  # Detect language from title
  lan = detect_language(text)

  # Language form title is valid
  if lan == 'de' or lan == 'en':
    print(f'Language from title is valid')
    return lan
  
  # Detect language from clean_text
  print('Detect language from clean_text')
  return detect_language(clean_text)



def process(text, custom_stopwords, tagger, is_title, clean_text):
  """
  Additional pre-processing steps for each 'full text' in munich_data_df
  """
  lan = ''

  # Check whether normal text from post/comments or title is processed for
  # language detection by title based on language of whole text grouped by post_id
  lan = determine_language_by_source(text, is_title, clean_text)
  
  text = text.lower()
  # Remove links
  text = re.sub(r'http\S+', '', text)

  if lan == 'en':
    clean_tokens_en = []
    tokens = word_tokenize(text, language='english')

    # Custom and predifiend stopwords
    custom_stopwords_english = set(custom_stopwords['english'])
    stopwords_en = set(stopwords.words('english'))
    all_stopwords_eng = stopwords_en.union(custom_stopwords_english)

    wnl = nltk.WordNetLemmatizer()
    lemmata = [wnl.lemmatize(w) for w in tokens]
    clean_tokens_en = [w for w in lemmata if w.isalpha() and w not in all_stopwords_eng]
    clean_tokens_en_string = ' '.join(clean_tokens_en)
    return clean_tokens_en_string
  
  if lan == 'de':
    clean_tokens_de = []
    tokens = word_tokenize(text, language='german')
    custom_stopwords_german = set(custom_stopwords['german'])
    stopwords_de = set(stopwords.words('german'))
    all_stopwords_de = stopwords_de.union(custom_stopwords_german)

    # Lemmatise HanTa
    tags = tagger.tag_sent(tokens)
    for word, lemma, pos in tags:
      # Write lemma in lower case
      lemma_lower = lemma.lower()

      if lemma_lower.isalpha() and lemma_lower not in all_stopwords_de:
        clean_tokens_de.append(lemma_lower)

    clean_tokens_de_string = ' '.join(clean_tokens_de)
    return clean_tokens_de_string
  return ''



def plot_bar_chart(data, x_label, y_label):
  """
  Plot a bar chart based on pandas series
  """

  ax = sns.barplot(
              data=data,
              x=x_label,
              y=y_label,
              )
  
  sns.despine(left=True, bottom=True)
  plt.show()
  return



def top_words_in_dict_format(lda_model, feature_names, n_top_words):
  """
  Write top words/weights from LDA in dictionary format
  """

  lda_results = {}

  for topic_idx, topic in enumerate(lda_model.components_):
    topic_title_idx = f'Thema {topic_idx+1}'
    top_features_ind = topic.argsort()[-n_top_words:]
    top_features_ind_rev = top_features_ind[::-1]
    top_features = feature_names[top_features_ind_rev]
    weights = topic[top_features_ind_rev]

    top_features_list = top_features.tolist()
    weights_list = weights.tolist()

    topic_dict = dict(zip(top_features_list, weights_list))

    lda_results[topic_title_idx] = topic_dict
  return lda_results



def plot_top_words(model, feature_names, n_top_words):
  """
  Plot top words/weights from LDA
  """
  fig, axes = plt.subplots(1, 5, figsize=(30, 8), sharex=True)
  axes = axes.flatten()
  for topic_idx, topic in enumerate(model.components_):
    top_features_ind = topic.argsort()[-n_top_words:]
    top_features_ind_rev = top_features_ind[::-1]
    top_features = feature_names[top_features_ind_rev]

    weights = topic[top_features_ind_rev]

    ax = axes[topic_idx]

    sns.barplot(
      x=weights,
      y=top_features,
      ax=ax
    )

    ax.set_title(f'Thema {topic_idx + 1}', fontdict={'fontsize': 30})
    ax.set_xlabel('Gewichtung (TF-IDF)', fontsize=20)

    if topic_idx == 0:
      ax.set_ylabel('Wörter', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)

    sns.despine(ax=ax, left=True, bottom=True)

  plt.subplots_adjust(top=0.80, bottom=0.15, wspace=0.90, hspace=0.3)
  plt.show()
  return



def main():
  """
  1. Retrieve secrets
  2. Get custom stopwords from JSON file
  3. If no munich_reddit_data.csv, retrieve data from Reddit
  4. Pre-Process retrieved data
  5. Extract most active users and used flairs
  6. TF-IDF and LDA
  7. Plot results
  """

  # Initialize HanoverTagger
  hanover_tagger = ht.HanoverTagger('morphmodel_ger.pgz')

  # Retrieve secrets
  client_id = get_secret('CLIENT_ID')
  client_secret = get_secret('CLIENT_SECRET')

  if not client_id:
    print('No client_id was found')
    return

  if not client_secret:
    print('No client_secret was found')
    return

  # Get custom stopwords from JSON file
  custom_stopwords = get_custom_stopwords(CUSTOM_STOPWORDS_PATH)

  if not isinstance(custom_stopwords, dict):
    print(custom_stopwords)
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
    munich_data = get_data_from_reddit(subreddit)

    # Write data to munich_reddit_data.csv
    munich_data.to_csv(MUNICH_DATA_PATH, index=False, encoding='utf-8-sig')


  # ---------------------------- Pre-Processing -------------------------------------------------
  print('Start Pre-Processing')
  # Read Reddit-Data from csv-file
  munich_data_df = pd.read_csv(MUNICH_DATA_PATH)

  # Replace NaN values with empty strings
  munich_data_df['title'] = munich_data_df['title'].fillna('')
  munich_data_df['text'] = munich_data_df['text'].fillna('')

  

  # Filter out STANDARD_AUTHORS
  munich_data_df_rm_std_author = munich_data_df[~munich_data_df['author'].isin(STANDARD_AUTHORS)]
  munich_data_df_rm_std_author = munich_data_df_rm_std_author.reset_index(drop=True)

  # Before goupby data pre-processing due to different used languages for each post
  print('Start process posts/comments')
  munich_data_df_rm_std_author['clean_text'] = munich_data_df_rm_std_author['text'].apply(process, custom_stopwords=custom_stopwords, tagger=hanover_tagger, is_title=False, clean_text=None)
  munich_data_df_rm_std_author.to_csv('Outputs/munich_data_df_rm_std_author.csv', index=False, encoding='utf-8-sig')
  print('End pre-process posts/comments')

  munich_data_df_groupby_post_id = munich_data_df_rm_std_author.groupby('post_id').agg(
    title=pd.NamedAgg(column='title', aggfunc='first'),
    flair=pd.NamedAgg(column='flair', aggfunc='first'),
    clean_text=pd.NamedAgg(column='clean_text', aggfunc=join_post_id_text)
  ).reset_index()

  # Data pre-processing of title
  print('Start pre-process titles')
  munich_data_df_groupby_post_id['clean_title'] = munich_data_df_groupby_post_id.apply(lambda x: process(x['title'],
                                                                                                        custom_stopwords=custom_stopwords,
                                                                                                        tagger=hanover_tagger,
                                                                                                        is_title=True,
                                                                                                        clean_text=x['clean_text']
                                                                                                        ),
                                                                                      axis=1)
  


  munich_data_df_groupby_post_id['full_text'] = munich_data_df_groupby_post_id['clean_title'] + ' ' + munich_data_df_groupby_post_id['clean_text']
  munich_data_df_groupby_post_id.to_csv('Outputs/munich_data_df_groupby_post_id.csv', index=False, encoding='utf-8-sig')
  print('End pre-process titles')

  text_corpus_list_groupby_post_id = munich_data_df_groupby_post_id['full_text'].tolist()

  text_corpus_list_groupby_post_id_txt = open('Outputs/text_corpus_list_groupby_post_id.txt', 'w')
  text_corpus_list_groupby_post_id_txt.writelines(str(text_corpus_list_groupby_post_id))
  text_corpus_list_groupby_post_id_txt.close()
  print('End Pre-Processing')


  # ------------------- Extract most active users and used flairs ----------------------------------

  # Drop out authors with no value
  authors = munich_data_df_rm_std_author['author'].dropna()

  # Return number of unique authors
  nunique_authors = authors.nunique()
  print(f'Anzahl unterschiedlicher Nutzer im Reddit-Datensatz:\n{nunique_authors}\n')
  
  # Return all unique authors 
  unique_authors = authors.unique()

  # Save all unique authors to csv file
  np.savetxt(UNIQUE_AUTHORS_CSV_PATH, unique_authors, delimiter=',', fmt='%s')

  # Return a Series with count of unique authors 
  # reset_index for writing in csv for two columns, instead of one, if not .reset_index() author is index
  most_active_authors_df = authors.value_counts().head(10).reset_index()

  # Define column headings and write to csv file
  most_active_authors_df.columns = ['Nutzer', 'Anzahl Posts/Kommentare']
  most_active_authors_df.to_csv(MOST_ACTIVE_AUTHORS_CSV_PATH, index=False, encoding='utf-8-sig')

  # Extract most used flairs
  flairs = munich_data_df_rm_std_author['flair'].dropna()
  nunique_flairs = flairs.nunique()
  print(f'Anzahl unterschiedlicher Flairs im Reddit-Datensatz:\n{nunique_flairs}\n')
  unique_flairs = flairs.unique()
  
  # Save all unique flairs to csv file
  np.savetxt(UNIQUE_FLAIRS_CSV_PATH, unique_flairs, delimiter=',', fmt='%s')

  # Return a Series with count of unique authors 
  # reset_index for writing in csv for two columns, instead of one, if not .reset_index() flair is index 
  most_common_flairs_df = flairs.value_counts().head(10).reset_index()

  # Define column headings and write to csv file
  most_common_flairs_df.columns = ['Flairs', 'Anzahl Flairs']
  most_common_flairs_df.to_csv(MOST_COMMON_FLAIRS_CSV_PATH, index=False, encoding='utf-8-sig')


  # ---------------------------- TF-IDF and LDA -------------------------------------------------

  # TF-IDF
  # Converts posts/comments from text_corpus_list_groupby_post_id to a matrix of TF-IDF featurs, where words are weighted
  # Matrix contains importance of specific words relative to whode post/comment collection from text_corpus_list_groupby_post_id 
  vectorizer = TfidfVectorizer(use_idf=True, # Enable inverse-document-frequency (IDF)
                               min_df=5, # Ignore terms with document frequency lower than 3
                               max_df=0.98, # Ignore terms with document frequency higher than 90%
                               smooth_idf=True # Prevents zero divisions by adding one to document frequencies by default
                               )

  # Learn voocabulary from text_corpus_list_groupby_post_id and return sparse matrix 
  # of (n_samples, n_features) Tf-idf-weighted document-term matrix
  # model = vectorizer.fit_transform(text_corpus_list_groupby_post_id)
  model = vectorizer.fit_transform(text_corpus_list_groupby_post_id)

  # Get output feature names for transformation
  feature_names = vectorizer.get_feature_names_out()

  # Latent Dirichlet Allocation (LDA)
  lda_model = LatentDirichletAllocation(n_components=5, # Number of topics to discover
                                        learning_method='online', # Method used to update model components (due to large data set 'online' is used)
                                        random_state=42, # Fixed parameter to control random number generator used to produce same results accross different calls
                                        max_iter=10, # Maximum number of passes over the training data
                                        n_jobs=-2, # Use all Threads of CPU except for one
                                        evaluate_every=1, # Evaluate perplexity for each iteration
                                        verbose=1 # Verbose information during training
                                        )
  
  print('Start Training')
  # Train model
  lda_model.fit(model)
  print('End Training')
  
  # Format top words with each weight from each topic as dictionary
  lda_results = top_words_in_dict_format(lda_model, feature_names, N_TOP_WORDS)

  # Write results from LDA as json to lda_results.json
  with open(LDA_RESULTS_JSON_PATH, 'w') as file:
    json.dump(lda_results, file, ensure_ascii=False, indent=2)
  

  # ---------------------------- Plot results ----------------------------------------------------
  plot_bar_chart(most_active_authors_df, 'Anzahl Posts/Kommentare', 'Nutzer')
  plot_bar_chart(most_common_flairs_df, 'Anzahl Flairs', 'Flairs')
  plot_top_words(lda_model, feature_names, N_TOP_WORDS)
  return munich_data_df


if __name__=='__main__':
  result = main()

