# app.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import streamlit as st

# Emotion labels
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Load model
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained("./saved_model")
model.eval()

# UI
st.title("Emotion Detector using BERT")
user_input = st.text_area("Enter a sentence to analyze emotions:")

if st.button("Predict"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            predicted = [emotion_labels[i] for i, p in enumerate(probs[0]) if p > 0.5]

        st.write("**Predicted Emotions:**")
        if predicted:
            st.success(", ".join(predicted))
        else:
            st.warning("No strong emotion detected.")
    else:
        st.warning("Please enter some text.")
