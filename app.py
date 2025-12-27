from flask import Flask, render_template, request
import joblib
import re
import nltk
import requests
from nltk.corpus import stopwords

nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

# Load BERT model & Logistic Regression
bert_model = joblib.load("bert_vectorizer.pkl")
model = joblib.load("bert_model.pkl")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

# News API
NEWS_API_KEY = "90d1f75385f0411a9109550cfde5eb23"
NEWS_URL = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=5&apiKey={NEWS_API_KEY}"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    reason = None
    articles = []

    # Fetch live news
    try:
        response = requests.get(NEWS_URL)
        articles = response.json().get("articles", [])
    except:
        pass

    if request.method == "POST":
        news_text = request.form.get("news", "").strip()
        if len(news_text.split()) < 5:
            prediction = "⚠️ Provide more text (at least 5 words)"
        else:
            cleaned = clean_text(news_text)
            emb = bert_model.encode([cleaned])
            prob_real = model.predict_proba(emb)[0][1]

            if prob_real > 0.5:  # threshold
                prediction = "🟢 REAL NEWS"
                confidence = round(prob_real*100,2)
                reason = "The content is factual, neutral, and matches verified news patterns."
            else:
                prediction = "🔴 FAKE NEWS"
                confidence = round((1-prob_real)*100,2)
                reason = "The content contains sensational or misleading patterns."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        reason=reason,
        articles=articles
    )

if __name__ == "__main__":
    app.run(debug=True)
