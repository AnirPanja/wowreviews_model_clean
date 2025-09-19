import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, gc

# Config
model_name = "bert-base-multilingual-cased"
data_path = os.path.join(os.path.dirname(__file__), "hotel_reviews_dataset.csv")  # ✅ new dataset
model_dir = os.path.join(os.path.dirname(__file__), "sentiment_bert_model")
batch_size = 4
epochs = 5
max_length = 64

# Load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["review", "label"])
    df["label"] = df["label"].astype(int)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    return train_df, test_df

# Tokenize
def tokenize_data(tokenizer, reviews, labels):
    encodings = tokenizer(
        reviews.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    return encodings, labels.values

def make_dataset(encodings, labels):
    return tf.data.Dataset.from_tensor_slices((dict(encodings), labels)).batch(batch_size)

def main():
    print("✨ Loading dataset...")
    train_df, test_df = load_and_preprocess_data()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encodings, train_labels = tokenize_data(tokenizer, train_df["review"], train_df["label"])
    test_encodings, test_labels = tokenize_data(tokenizer, test_df["review"], test_df["label"])

    train_dataset = make_dataset(train_encodings, train_labels)
    test_dataset = make_dataset(test_encodings, test_labels)

    print("✨ Loading model...")
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print("🚀 Starting training...")
    model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)

    print("📝 Evaluating model...")
    loss, acc = model.evaluate(test_dataset)
    print(f"✅ Final Test Loss: {loss}, Accuracy: {acc}")

    # Save model
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"💾 Model + tokenizer saved to {model_dir}")

    # Free memory
    gc.collect()
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
