# Projekt Data Analysis

## Aufgabe 2: Extrahieren der häufigsten Themen aus Twitter-Nachrichten
### (Task 2: Extracting the most frequent topics from Twitter messages)

The aim of the project is to extract the most active users, most common flairs and most frequently discussed topics from defined Subreddit posts and comments. The project uses langdetect for data sanitization, several Natural Language Processing (NLP) techniques like Latent Dirichlet Allocation (LDA) and Latent Semantic Analysis (LSA) as well as a comparison of different combinations of the models based on coherence score.

Due to costs for the Twitter API, Reddit was used instead as an alternative social media platform. Especially, I want to thank [archanakkokate](https://medium.com/@archanakkokate/scraping-reddit-data-using-python-and-praw-a-beginners-guide-7047962f5d29) for the great tutorial on Medium about the Reddit PRAW Python library, which was a great source and inspiration.

### Requirements

For using this project several steps are needed beforehand, which will be described in the following (based on archanakkokate):

- Register Reddit account 
- Create a new app under https://www.reddit.com/prefs/apps
- Retrieve client id, client secret and user agent
- Optional: Run the Code from main.py in a Jupyter Notebook 

### Installation

Clone the repository
```bash
git clone https://github.com/AndreasH4/Projekt_Data_Analysis.git
```

Enter the repository
```bash
cd Projekt_Data_Analysis
```

Create a python virtual environment
```bash
python3.12 -m venv <your-environment-name>
```

Activate the virtual environment
```bash
source <your-environment-name>/bin/activate
```

Install requirements e.g. with `pip` (Easter Egg: abhaengigkeiten means requirements in german) ;)
```bash
pip install -r abhaengigkeiten.txt
```

Enter your previously retrieved credentials in the `.env` file
```python
CLIENT_ID='<your-client-id>'
CLIENT_SECRET='<your-client-secret>'
```

Run the script (or optionally as a notebook main.ipynb within jupyter lab)
```python
python main.py
```
