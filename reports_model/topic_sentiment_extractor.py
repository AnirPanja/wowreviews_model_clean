import pandas as pd
import numpy as np
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch

# Config
data_path = os.path.join(os.path.dirname(__file__), "data", "tripadvisor_hotel_reviews.csv")
sentiment_model_path = os.path.join(os.path.dirname(__file__), "tripadvisor_bert_model_chunked", "final")
output_path = os.path.join(os.path.dirname(__file__), "topic_sentiment_data.csv")

# Load data with flexible delimiter detection
df = pd.read_csv(data_path, sep=None, engine="python")  # Auto-detect delimiter
print(f"Detected columns: {df.columns.tolist()}")  # Debug: Show detected columns

# Take only first two columns (review and rating), ignoring extras
if len(df.columns) < 2:
    raise ValueError(f"Expected at least 2 columns, found {len(df.columns)}. Check the delimiter or file structure.")
df = df.iloc[:, :2]  # Take first two columns
df.columns = ["review", "rating"]  # Rename explicitly
df = df.dropna(subset=["review", "rating"])

# Map rating to sentiment
def map_rating_to_sentiment(rating):
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["rating"].apply(map_rating_to_sentiment)

# Load sentiment model with TensorFlow weights
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path, from_tf=True)

def get_sentiment_confidence(review):
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    predicted_class = np.argmax(probs)
    return probs[predicted_class]

df["sentiment_confidence"] = df["review"].apply(get_sentiment_confidence)

# Topic Discovery
topic_model = BERTopic(min_topic_size=50, verbose=True)
topics, probs = topic_model.fit_transform(df["review"].tolist())
df["topic"] = topics

# Save results
df[["review", "topic", "sentiment", "sentiment_confidence"]].to_csv(output_path, index=False)
print("Topic-Sentiment Data saved to:", output_path)