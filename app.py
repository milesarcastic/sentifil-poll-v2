import os
os.environ['TF_USE_LEGACY_KERAS'] = "True"

import re
import praw
import ktrain
import emoji
import gdown
import nltk,json
import secrets
import csv
import io
import pandas as pd


nltk.download('punkt_tab')
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from ktrain import text
from flask import Flask, render_template, request, Response, session
from GoogleNews import GoogleNews
from collections import Counter
from nltk.corpus import stopwords


app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 

file_ids = {
    "tf_model.h5": "1-9Ie7NqGnMDCQD2PrTdgmaY0HEbxSROl",
    "tf_model.preproc": "1-AVaLocYCUZLujYA4pAWcZp_TpFt20y7",
    "vocab.txt" : "1-N6Ck0PS9dXSrxojF9IpKhy7SNboX0f2",
    "tokenizer.json": "1-JqEsyNsCPDGlK-W9n1tV76aTpHbcj3u",
    "tokenizer_config.json": "1-ZfE8tRXsBQfTOXxqDMxWKpXWF3YKs3r",
    "special_tokens_map.json": "1-NqlVZnpIju3FlYh8HSDnRaM2ZbrV0PQ",
    "config.json": "1--WQAmROfPZF22EyU3B2GWBpFb-iijyB"
}
MODEL_PATH = "5e5-model"
RESULTS_FILE = "results.json"

ENGLISH_STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = {
    "akin", "aking", "ako", "alin", "am", "amin", "aming", "ang", "ano", "anumang", "apat", "at", "atin", "ating", "ay",
    "bago", "bakit", "bawat", "bilang", "dahil", "dalawa", "dapat", "din", "dito", "doon", "gagawin", "gayunman", "habang",
    "hanggang", "hindi", "iba", "ibaba", "ibabaw", "ibig", "would", "also", "dont","tapos","im","ikaw", "ilagay", "ilalim", "ilan", "inyong", "isa", "isang",
    "itaas", "ito", "iyo", "iyon", "iyong", "ka", "kahit", "kami", "kanila", "kanilang", "kanino", "kanya", "kanyang",
    "kapag", "katulad", "kaya", "kaysa", "ko", "kong", "kulang", "kumuha", "kung", "lamang", "likod", "lima", "maaari",
    "maaaring", "maging", "makita", "marami", "marapat", "masyado", "may", "mayroon", "mga", "minsan", "mismo", "mula",
    "muli", "na", "nabanggit", "naging", "nagkaroon", "nais", "nakita", "namin", "napaka", "narito", "nasaan", "ng",
    "ngayon", "ni", "nila", "nilang", "nito", "niya", "niyang", "noon", "o", "pa", "paano", "pababa", "pagitan",
    "pagkakaroon", "pagkatapos", "palabas", "pamamagitan", "pangalawa", "para", "pero", "pumupunta", "sa", "saan",
    "sabi", "sabihin", "sarili", "sila", "sino", "siya", "tatlo", "tayo", "tulad", "tungkol", "una", "walang", "po",
    "opo", "lang", "talaga", "nya", "mag", "nung", "ba", "di", "si", "daw", "din", "pano", "edi", "wag", "kc", "dhil",
    "bkit", "diba", "yung", "kasi", "mo", "naman", "natin", "ung", "pag", "nag", "parang", "rin", "yan", "nyo", "yun",
    "eh", "ma", "nga", "nasa", "kayo", "sya", "u", "may", "kay", "bakit", "gobyerno", "mas", "ako","amua","ato","busa","ikaw","ila","ilang",
    "imo","imong","iya","iyang","kaayo","kana","kaniya","kaugalingon","kay",
    "kini","kinsa","kita","lamang","mahimong","mga","mismo","nahimo","nga","pareho","pud","sila","siya","unsa"
} | ENGLISH_STOPWORDS  # Merge both sets


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

# Initialize Google News API (set to PH)
google_news = GoogleNews(region='PH')

# Function to remove emojis from text
def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

def clean_text(text, search_keyword):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = text.split()  # Tokenize by spaces

    # Ensure all words in search_keyword are treated as stopwords
    all_stopwords = set(CUSTOM_STOPWORDS)  # Convert to set for faster lookup
    search_keywords = search_keyword.lower().split()  # Split multi-word keyword into individual words
    all_stopwords.update(search_keywords)  # Add all words of the keyword phrase to stopwords

    # Remove stopwords and keyword words
    tokens = [word for word in tokens if word not in all_stopwords]
    
    return " ".join(tokens)  # Reconstruct cleaned text

# Function to preprocess text for trigrams
def trigrams_preprocess(texts, search_keyword):
    """Cleans and tokenizes text for trigram extraction."""
    cleaned_texts = [clean_text(text, search_keyword) for text in texts]
    all_tokens = [text.split() for text in cleaned_texts]  # Tokenize each text
    return all_tokens  # Return list of token lists

# Function to get top trigrams
def get_top_trigrams(texts, top_n=5):
    """Extracts the most common trigrams from cleaned texts."""
    all_trigrams = []
    for tokens in texts:
        trigrams = list(ngrams(tokens, 3))  # Extract trigrams
        all_trigrams.extend(trigrams)

    trigram_counts = Counter(all_trigrams)
    return trigram_counts.most_common(top_n)

# Index route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Results route
@app.route("/results", methods=["GET"])
def results():
    keyword = request.args.get("keyword")

    # Scrape Reddit
    reddit_data = []
    subreddits = ['Philippines', 'ChikaPH', 'newsPH',
                  'inquirerdotnet', 'CasualPH', 'LawPH',
                  'AskPH', 'adviceph', 'adultingph', 'EpalPH', 'Cebu']

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.search(keyword, limit=1000):
            reddit_data.append({"text": submission.title + " " + submission.selftext, "source": "Reddit"})

    # Scrape Google News
    google_news.search(keyword)
    google_results = google_news.results()
    news_data = [{"text": result['title'] + " " + result['desc'], "source": "Google News"} 
                 for result in google_results[:1000]]
    print("NEWS:", news_data)

    # Combine & preprocess data
    combined_data = reddit_data + news_data
    cleaned_data = [{"text": remove_emojis(item["text"]), "source": item["source"]} for item in combined_data]
    
    # Automatically include keyword as a stopword
    no_stopwords_data = [{"text": clean_text(item["text"], keyword), "source": item["source"]} for item in cleaned_data]

    # Perform sentiment analysis
    predictions = model.predict([item["text"] for item in cleaned_data])
    predictions = [int(pred) for pred in predictions]

    results = [{"text": item["text"], "sentiment": pred, "source": item["source"]} 
               for item, pred in zip(cleaned_data, predictions)]

    # Save results in a JSON file
    RESULTS_FILE = "sentiment_results.json"
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f)

    # Sentiment counts for pie chart
    sentiment_counts = dict(Counter(predictions))
    for sentiment in range(4):  # Ensure all sentiment categories exist
        sentiment_counts.setdefault(sentiment, 0)

    # Word frequency analysis (no stopwords & no keyword)
    all_words = " ".join([item["text"] for item in no_stopwords_data]).split()
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(20)
    word_frequencies = dict(common_words)

    # Extract top 5 trigrams per sentiment
    sentiment_texts = {0: [], 1: [], 2: [], 3: []}  # Group texts by sentiment
    for result in results:
        sentiment_texts[result["sentiment"]].append(result["text"])

    # Preprocess sentiment texts for trigrams
    trigram_texts = {sentiment: trigrams_preprocess(sentiment_texts[sentiment], keyword)
                     for sentiment in sentiment_texts}

    top_trigrams = {
        sentiment: get_top_trigrams(trigram_texts[sentiment], top_n=5)
        for sentiment in trigram_texts
    }

    return render_template(
        "results.html",
        keyword=keyword,
        results=results,
        sentiment_counts=sentiment_counts,
        word_frequencies=word_frequencies,
        top_trigrams=top_trigrams
    )

# CSV download route
@app.route("/download_csv")
def download_csv():
    # Load results from JSON file
    RESULTS_FILE = "sentiment_results.json"
    if not os.path.exists(RESULTS_FILE):
        return "No data available for download", 400

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Mapping sentiment numbers to words
    sentiment_map = {1: "Negative", 2: "Positive", 0: "Neutral", 3: "Mixed"}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Text", "Sentiment", "Source"])  # Updated header row

    for result in results:
        sentiment_text = sentiment_map.get(result["sentiment"], "Unknown")  # Convert number to word
        writer.writerow([result["text"], sentiment_text, result["source"]])

    output.seek(0)
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=sentiment_data.csv"
    return response


@app.route('/about')
def about():
    model_results = [
        {"Model": 1, "Batch Size": 16, "Learning Rate": "1e-4", "Accuracy": 0.42628, "F1 Score": 0.31714, "Precision": 0.36865, "Recall": 0.42628},
        {"Model": 2, "Batch Size": 16, "Learning Rate": "2e-5", "Accuracy": 0.86352, "F1 Score": 0.86074, "Precision": 0.86181, "Recall": 0.86352},
        {"Model": 3, "Batch Size": 16, "Learning Rate": "5e-5", "Accuracy": 0.86015, "F1 Score": 0.85824, "Precision": 0.85680, "Recall": 0.86015},
        {"Model": 4, "Batch Size": 32, "Learning Rate": "1e-4", "Accuracy": 0.83235, "F1 Score": 0.83267, "Precision": 0.83313, "Recall": 0.83235},
        {"Model": 5, "Batch Size": 32, "Learning Rate": "2e-5", "Accuracy": 0.84919, "F1 Score": 0.84721, "Precision": 0.84590, "Recall": 0.84919},
        {"Model": 6, "Batch Size": 32, "Learning Rate": "5e-5", "Accuracy": 0.87447, "F1 Score": 0.87285, "Precision": 0.87246, "Recall": 0.87447}
    ]
    return render_template('about.html', model_results=model_results)

# Run the ap
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
