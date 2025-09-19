import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, gc
import sys
import importlib.util

# Config
model_name = "bert-base-multilingual-cased"
data_path = os.path.join(os.path.dirname(__file__), "data", "tripadvisor_hotel_reviews.csv")
dynamic_data_py = os.path.join(os.path.dirname(__file__), "data", "review_data.py")  # Path to your PY file
model_dir = os.path.join(os.path.dirname(__file__), "tripadvisor_bert_model_chunked2")
chunk_size = 500   # process in chunks to avoid memory issues
batch_size = 8     # slightly bigger batch for stability
epochs = 2         # increase for better learning

# Load and preprocess data
def load_and_preprocess_data(use_csv=False):
    print("Hello Script2 called")
    if use_csv:
        # Load from CSV (default)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found at: {data_path}")
        df = pd.read_csv(data_path)
    else:
        # Load from PY file for dynamic real-time data
        print(f"Looking for review_data.py at: {dynamic_data_py}")
        if not os.path.exists(dynamic_data_py):
            raise FileNotFoundError(f"review_data.py not found at: {dynamic_data_py}")
        
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("review_data", dynamic_data_py)
            review_data = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(review_data)
            
            # Check if fetch_reviews exists
            if not hasattr(review_data, "fetch_reviews"):
                raise AttributeError("review_data.py must define a fetch_reviews function")
            
            df = review_data.fetch_reviews()  # Must return pd.DataFrame with 'Review' and 'Rating' columns
            if not isinstance(df, pd.DataFrame):
                raise ValueError("fetch_reviews() must return a pandas DataFrame")
            if not all(col in df.columns for col in ["Review", "Rating"]):
                raise ValueError("DataFrame must have 'Review' and 'Rating' columns")
        except Exception as e:
            print(f"Error loading data from review_data.py: {e}")
            raise

    df = df.rename(columns={"Review": "review", "Rating": "rating"})
    df = df.dropna(subset=["review", "rating"])
    
    # Label mapping: 0 = Negative, 1 = Neutral, 2 = Positive
    df["label"] = df["rating"].map(lambda r: 0 if r in [1, 2] else (1 if r == 3 else 2))

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Balance classes
    class_counts = df["label"].value_counts()
    min_class_size = class_counts.min()
    df_balanced = pd.concat([
        df[df["label"] == i].sample(min_class_size, replace=True, random_state=42) for i in range(3)
    ])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    train_df, test_df = train_test_split(
        df_balanced[["review", "label"]], test_size=0.2, random_state=42
    )
    print(f"✅ Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    return train_df, test_df

# Tokenize
def tokenize_data(tokenizer, reviews, labels, max_length=128):
    encodings = tokenizer(
        reviews.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    return encodings, labels.values

# Create dataset
def make_dataset(tokenizer, df, shuffle=True):
    encodings, labels = tokenize_data(tokenizer, df["review"], df["label"])
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    return dataset.batch(batch_size)

def main():
    os.makedirs(model_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    # Set use_csv=True to load from CSV, or False to load from PY file
    use_csv_data = False  # Change to False to use dynamic PY data
    train_df, test_df = load_and_preprocess_data(use_csv=use_csv_data)

    # Init model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Train in chunks
    total_chunks = (len(train_df) + chunk_size - 1) // chunk_size
    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch+1}/{epochs} ==========")
        for i in range(0, len(train_df), chunk_size):
            chunk_df = train_df.iloc[i:i+chunk_size]
            chunk_idx = f"{i+1}-{i+len(chunk_df)}"
            print(f"\n🔹 Training on chunk {chunk_idx} ({len(chunk_df)} samples)...")

            train_dataset = make_dataset(tokenizer, chunk_df, shuffle=True)
            model.fit(train_dataset, epochs=1, verbose=1)

            # Free memory
            del train_dataset
            gc.collect()

        # Evaluate at end of each epoch
        val_dataset = make_dataset(tokenizer, test_df, shuffle=False)
        loss_val, acc_val = model.evaluate(val_dataset)
        print(f"📊 Epoch {epoch+1} Validation - Loss: {loss_val:.4f}, Accuracy: {acc_val:.4f}")

    # Save final model
    final_model_dir = os.path.join(model_dir, "final")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"\n✅ Final model + tokenizer saved to {final_model_dir}")

if __name__ == "__main__":
    main()