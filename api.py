import pickle
import re
import string
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model_path = r"C:\Users\mamun\OneDrive\codetech intership\projects\model.pkl"
vectorizer_path = r"C:\Users\mamun\OneDrive\codetech intership\projects\vectorizer.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[{}]".format(string.punctuation), "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake News Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    # Preprocess input text
    cleaned_text = clean_text(data["text"])
    transformed_text = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(transformed_text)[0]
    result = "Real" if prediction == 1 else "Fake"
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
