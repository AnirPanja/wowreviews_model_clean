import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from datasets import Dataset
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# -------------------
# Load CSV
# -------------------
data_path = "data/phrase_sentiment_data.csv"
df = pd.read_csv(data_path, encoding="latin1")

# Ensure correct columns
assert all(col in df.columns for col in ["review", "phrase", "sentiment"])

# Map labels to integers
label_map = {"positive": 1, "negative": 0}
df["label"] = df["sentiment"].map(label_map)

# -------------------
# ✅ Clean invalid labels
# -------------------
print("Before cleaning:", df["label"].value_counts(dropna=False))

# Drop rows where label is NaN or invalid
df = df[df["label"].isin([0, 1])]
df = df.dropna(subset=["review", "phrase"])

print("After cleaning:", df["label"].value_counts(dropna=False))

# -------------------
# Train / Test Split
# -------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train_dataset = Dataset.from_pandas(train_df[["review", "phrase", "label"]])
test_dataset = Dataset.from_pandas(test_df[["review", "phrase", "label"]])

# -------------------
# Tokenizer
# -------------------
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    reviews = [str(r) if r is not None else "" for r in examples["review"]]
    phrases = [str(p) if p is not None else "" for p in examples["phrase"]]

    return tokenizer(
        text=reviews,
        text_pair=phrases,
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["review", "phrase"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["review", "phrase"])

# ✅ Rename labels column (required by HF models)
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

# -------------------
# Convert to TF dataset
# -------------------
train_dataset = train_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols="labels",
    shuffle=True,
    batch_size=16
)

test_dataset = test_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids"],
    label_cols="labels",
    shuffle=False,
    batch_size=16
)

# -------------------
# Model
# -------------------
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Optimizer
num_train_steps = len(train_dataset) * 3  # 3 epochs
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=num_train_steps)

# ✅ Use explicit loss
model.compile(
    optimizer=optimizer,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# -------------------
# Train
# -------------------
history = model.fit(train_dataset, validation_data=test_dataset, epochs=3)

# -------------------
# Evaluate
# -------------------
y_true, y_pred = [], []

for x, y in test_dataset:
    logits = model.predict(x, verbose=0).logits
    preds = tf.argmax(logits, axis=-1).numpy()
    y_pred.extend(preds)
    y_true.extend(y.numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

# -------------------
# Save Model
# -------------------
model.save_pretrained("phrase_sentiment_classifier")
tokenizer.save_pretrained("phrase_sentiment_classifier")

print("✅ Model trained and saved to phrase_sentiment_classifier/")

