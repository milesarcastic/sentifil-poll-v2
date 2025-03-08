import os
os.environ['TF_USE_LEGACY_KERAS'] = "True"

import re
import praw
import ktrain
import emoji
import gdown
import nltk,json
import secrets  # For generating a secure secret key
import csv
import io

nltk.download('punkt_tab')
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from ktrain import text
from flask import Flask, render_template, request, Response, session
import pandas as pd
from GoogleNews import GoogleNews
from collections import Counter



app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Set a unique secret key


file_ids = {
    "tf_model.h5": "1-9Ie7NqGnMDCQD2PrTdgmaY0HEbxSROl",
    "tf_model.preproc": "1-AVaLocYCUZLujYA4pAWcZp_TpFt20y7",
    "vocab.txt" : "1-N6Ck0PS9dXSrxojF9IpKhy7SNboX0f2",
    "tokenizer.json": "1-JqEsyNsCPDGlK-W9n1tV76aTpHbcj3u",
    "tokenizer_config.json": "1-ZfE8tRXsBQfTOXxqDMxWKpXWF3YKs3r",
    "special_tokens_map.json": "1-NqlVZnpIju3FlYh8HSDnRaM2ZbrV0PQ",
    "config.json": "1--WQAmROfPZF22EyU3B2GWBpFb-iijyB"
}
MODEL_PATH = "dummy_sentiment_model"
RESULTS_FILE = "results.json"

# # Download each file
# for filename, file_id in file_ids.items():
#     output_path = os.path.join(MODEL_PATH, filename)
#     gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

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

def trigrams_preprocess(text):
    """Lowercases, removes special characters, and tokenizes text."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)
    return tokens
    
def get_top_trigrams(texts, top_n=5):
    """Extracts the most common trigrams from a list of texts."""
    all_trigrams = []
    for text in texts:
        tokens = trigrams_preprocess(text)
        trigrams = list(ngrams(tokens, 3))  # Extract trigrams
        all_trigrams.extend(trigrams)

    trigram_counts = Counter(all_trigrams)
    return trigram_counts.most_common(top_n)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/results", methods=["GET"])
def results():
    keyword = request.args.get("keyword")

    # Scrape Reddit
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

    # Combine & preprocess data
    combined_data = reddit_data + news_data
    cleaned_data = [remove_emojis(text) for text in combined_data]

    # Perform sentiment analysis
    predictions = model.predict(cleaned_data)
    predictions = [int(pred) for pred in predictions]

    results = [{"text": text, "sentiment": pred} for text, pred in zip(cleaned_data, predictions)]
   # Save results in a JSON file instead of session
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f)

    # Sentiment counts for pie chart
    sentiment_counts = dict(Counter(predictions))
    for sentiment in range(4):  # Ensure all sentiment categories exist
        sentiment_counts.setdefault(sentiment, 0)

  # Word frequencies for word cloud
    all_words = " ".join(cleaned_data).split()
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(10)
    word_frequencies = dict(common_words)

     # Extract top 5 trigrams per sentiment
    sentiment_texts = {0: [], 1: [], 2: [], 3: []}  # Group texts by sentiment
    for result in results:
        sentiment_texts[result["sentiment"]].append(result["text"])

    top_trigrams = {
        sentiment: get_top_trigrams(sentiment_texts[sentiment], top_n=5)
        for sentiment in sentiment_texts
    }


    return render_template(
        "results.html",
        keyword=keyword,
        results=results,
        sentiment_counts=sentiment_counts,
        word_frequencies=word_frequencies,
        top_trigrams=top_trigrams
    )
@app.route("/download_csv")
def download_csv():
    # Load results from JSON file
    if not os.path.exists(RESULTS_FILE):
        return "No data available for download", 400

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Mapping sentiment numbers to words
    sentiment_map = {1: "Negative", 2: "Positive", 0: "Neutral", 3: "Mixed"}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Text", "Sentiment"])  # Header row

    for result in results:
        sentiment_text = sentiment_map.get(result["sentiment"], "Unknown")  # Convert number to word
        writer.writerow([result["text"], sentiment_text])

    output.seek(0)
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=sentiment_data.csv"

    return response
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
