import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import evaluate
import os

# Ensure TensorFlow backend
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # For compatibility

# Load dataset
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Preprocess function
def preprocess(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Convert to TensorFlow datasets
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

train_dataset = tokenized_dataset["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator,
)

test_dataset = tokenized_dataset["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    shuffle=False,
    batch_size=8,
    collate_fn=data_collator,
)

# Load model
model = TFAutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2,
    cache_dir='./cache'
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=1,  # Use 1 for quick setup; increase to 3 for better accuracy
)

# Save model
model.save_pretrained('my_model')
tokenizer.save_pretrained('my_model')
print("Model saved to ./my_model")