import streamlit as st
import requests
import pandas as pd
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
news_path = r"C:\Users\mamun\Downloads\news.csv"
df = pd.read_csv(news_path)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[{}]".format(string.punctuation), "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

df["text"] = df["text"].astype(str).apply(clean_text)

# Convert labels to numeric (0 = Fake, 1 = Real)
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

# Check for missing values
df.dropna(inplace=True)

# Prepare dataset for training
X = df["text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=2, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train model using Logistic Regression
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)

# Save trained model
model_path = r"C:\Users\mamun\OneDrive\codetech intership\projects\model.pkl"
vectorizer_path = r"C:\Users\mamun\OneDrive\codetech intership\projects\vectorizer.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

# Load model for Streamlit app
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)



# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# API Endpoint 
API_URL = "http://127.0.0.1:5000/predict"

# Streamlit Web App
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's real or fake.")

user_input = st.text_area("Paste News Article Here:")
if st.button("Check News"):
        if user_input:
            response = requests.post(API_URL, json={"text": user_input})
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['prediction']}**")
                
            else:
                st.error("Error: Could not connect to API")
        else:
             st.warning("Please enter news text first.")
