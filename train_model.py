# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import re
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# LOAD DATA
# ===============================
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

true_news["label"] = 1
fake_news["label"] = 0

data = pd.concat([true_news, fake_news])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ===============================
# TEXT PREPROCESSING
# ===============================
nltk.download("stopwords")
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

data["cleaned_text"] = data["text"].apply(clean_text)

# ===============================
# TF-IDF VECTORIZATION
# ===============================
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(data["cleaned_text"])
y = data["label"]

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# TRAIN MODELS
# ===============================
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
lr_model.fit(X_train, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# ===============================
# MODEL SELECTION
# ===============================
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

best_model = lr_model if lr_acc >= nb_acc else nb_model

print("Logistic Regression Accuracy:", lr_acc)
print("Naive Bayes Accuracy:", nb_acc)

# ===============================
# EVALUATION
# ===============================
final_pred = best_model.predict(X_test)
print(classification_report(y_test, final_pred))

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, final_pred)
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"])
plt.title("Confusion Matrix")
plt.show()

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(best_model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model & Vectorizer saved successfully")

