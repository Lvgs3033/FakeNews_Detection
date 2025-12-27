import pandas as pd
import re
import nltk
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np

# -------------------------
# 1. NLTK stopwords
# -------------------------
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    """Clean and preprocess news text"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

# -------------------------
# 2. Load Dataset
# -------------------------
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

true_news["label"] = 1
fake_news["label"] = 0

data = pd.concat([true_news, fake_news]).sample(frac=1, random_state=42).reset_index(drop=True)
data["cleaned_text"] = data["text"].apply(clean_text)

# Optional: Use small sample for testing (remove for full dataset)
# data = data.sample(n=500, random_state=42)

# -------------------------
# 3. Load BERT model
# -------------------------
print("Loading BERT model...")
model_name = "all-MiniLM-L6-v2"  # small & fast
bert_model = SentenceTransformer(model_name)

# -------------------------
# 4. Generate embeddings
# -------------------------
print("Generating embeddings... (this may take a few minutes)")
embeddings = []
for text in tqdm(data["cleaned_text"].tolist()):
    emb = bert_model.encode(text)
    embeddings.append(emb)

embeddings = np.array(embeddings)
print("Embeddings generated successfully!")

# -------------------------
# 5. Split dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(embeddings, data["label"], test_size=0.2, random_state=42)

# -------------------------
# 6. Train Logistic Regression
# -------------------------
print("Training Logistic Regression on BERT embeddings...")
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)

# -------------------------
# 7. Evaluate
# -------------------------
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# -------------------------
# 8. Save model & BERT object
# -------------------------
joblib.dump(lr, "bert_model.pkl")
joblib.dump(bert_model, "bert_vectorizer.pkl")
print("Model and BERT vectorizer saved successfully!")
