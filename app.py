import os
os.environ['TF_USE_LEGACY_KERAS'] = "True"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dummy_sentiment_model")

import re
import praw
import ktrain
import emoji
import gdown

from ktrain import text
from flask import Flask, render_template, request
import pandas as pd
from GoogleNews import GoogleNews
from collections import Counter



app = Flask(__name__)

file_ids = {
    "tf_model.h5": "1-9Ie7NqGnMDCQD2PrTdgmaY0HEbxSROl",
    "tf_model.preproc": "1-AVaLocYCUZLujYA4pAWcZp_TpFt20y7",
    "vocab.txt" : "1-N6Ck0PS9dXSrxojF9IpKhy7SNboX0f2",
    "tokenizer.json": "1-JqEsyNsCPDGlK-W9n1tV76aTpHbcj3u",
    "tokenizer_config.json": "1-ZfE8tRXsBQfTOXxqDMxWKpXWF3YKs3r",
    "special_tokens_map.json": "1-NqlVZnpIju3FlYh8HSDnRaM2ZbrV0PQ",
    "config.json": "1--WQAmROfPZF22EyU3B2GWBpFb-iijyB"
}
# MODEL_PATH = "dummy_sentiment_model"

# Download each file
for filename, file_id in file_ids.items():
    output_path = os.path.join(MODEL_PATH, filename)
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

print("Model files downloaded!")

# Load pre-trained sentiment model
model = ktrain.load_predictor(MODEL_PATH)

# Initialize Reddit API
reddit = praw.Reddit(client_id='9WA1fYAv9WAwoOrBQzPq6Q',
                     client_secret='wpqongdwvlHYB2gJc6iif8Xt8kK9RQ',
                     user_agent='emdem')

# Initialize Google News API
google_news = GoogleNews()

# Function to clean and preprocess text
def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = request.form.get("keyword")
        
        reddit_data = []
        subreddits = ["ChikaPH", "Philippines", "LawPH"]
        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(keyword, limit=5):
                reddit_data.append(submission.title + " " + submission.selftext)
        
        # Scrape Google News
        google_news.search(keyword)
        google_results = google_news.results()
        news_data = [result['title'] + " " + result['desc'] for result in google_results[:5]]
        
        # Combine data and preprocess
        combined_data = reddit_data + news_data
        cleaned_data = [remove_emojis(text) for text in combined_data]
        
        # Perform sentiment analysis
        predictions = model.predict(cleaned_data)

        # Convert predictions to integers if necessary
        predictions = [int(pred) for pred in predictions]

        results = [{"text": text, "sentiment": pred} for text, pred in zip(cleaned_data, predictions)]

        # Sentiment counts for the chart
        sentiment_counts = dict(Counter(predictions))

        # Ensure all sentiment categories are included (0 = Neutral, 1 = Negative, 2 = Positive, 3 = Mixed)
        for sentiment in range(4):
            if sentiment not in sentiment_counts:
                sentiment_counts[sentiment] = 0
        
        # Word frequencies for the word cloud
        all_words = " ".join(cleaned_data).split()
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(10)
        word_frequencies = dict(common_words)
        
        return render_template("index.html", keyword=keyword, results=results,
                               sentiment_counts=sentiment_counts, word_frequencies=word_frequencies)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
