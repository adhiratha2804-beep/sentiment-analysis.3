# -------------------------------
# Sentiment Analysis Project
# TF-IDF + Logistic Regression
# -------------------------------

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK data (run once)
nltk.download('stopwords')

# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("sentiment.csv")   # change path if needed
print(data.head())

# -------------------------------
# 2. Text Cleaning
# -------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

data['clean_text'] = data['text'].apply(clean_text)

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X = data['clean_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -------------------------------
# 5. Model Training
# -------------------------------
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# -------------------------------
# 6. Prediction
# -------------------------------
y_pred = model.predict(X_test_tfidf)

# -------------------------------
# 7. Evaluation Metrics
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# -------------------------------
# 8. Confusion Matrix Plot
# -------------------------------
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.show()

# -------------------------------
# 9. Inference on New Text
# -------------------------------
def predict_sentiment(text):
    text = clean_text(text)
    vector = tfidf.transform([text])
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"

# Example
print("\nSentiment:", predict_sentiment("The product quality is excellent"))