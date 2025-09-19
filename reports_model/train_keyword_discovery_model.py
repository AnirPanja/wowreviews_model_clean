import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

import tensorflow as tf
from transformers import (
    AutoTokenizer,
    TFAutoModelForTokenClassification,
    create_optimizer,
)

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 5e-5

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "phrase_sentiment_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "phrase_sentiment_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH, encoding="latin1")

print(f"Loaded {len(df)} rows from {DATA_PATH}")
print("Sample data:")
print(df.head())

# ----------------------------
# LABEL SETUP
# ----------------------------
label_list = ["O", "B-POS", "I-POS", "B-NEG", "I-NEG"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

print("Label mapping:", label2id)

# ----------------------------
# TOKENIZER
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ----------------------------
# HELPER: Find phrase span
# ----------------------------
def find_phrase_char_span(review: str, phrase: str, max_gap=2):
    review_low = review.lower()
    phrase_low = str(phrase).lower().strip()
    idx = review_low.find(phrase_low)
    if idx != -1:
        return idx, idx + len(phrase_low)

    # fallback subsequence match
    review_words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r'\w+', review_low)]
    phrase_words = re.findall(r'\w+', phrase_low)
    if not phrase_words or not review_words:
        return None

    for i in range(len(review_words)):
        if review_words[i][0] != phrase_words[0]:
            continue
        curr = i
        matched = True
        for pw in phrase_words[1:]:
            found = False
            for k in range(curr + 1, min(curr + 1 + max_gap + 1, len(review_words))):
                if review_words[k][0] == pw:
                    curr = k
                    found = True
                    break
            if not found:
                matched = False
                break
        if matched:
            return review_words[i][1], review_words[curr][2]
    return None


# ----------------------------
# ALIGN LABELS
# ----------------------------
def align_labels(review, phrases_with_sentiments):
    encoding = tokenizer(review, return_offsets_mapping=True, max_length=MAX_LEN, truncation=True)
    offset_mapping = encoding["offset_mapping"]
    token_labels = ["O"] * len(offset_mapping)

    for phrase, sentiment in phrases_with_sentiments:
        if not phrase or str(phrase).lower() == "nan":
            continue

        span = find_phrase_char_span(review, phrase, max_gap=2)
        if not span:
            continue

        start_char, end_char = span
        for i, (start, end) in enumerate(offset_mapping):
            if start is None or end is None:
                continue
            if end <= start_char:
                continue
            if start >= end_char:
                break

            if start <= start_char < end:
                token_labels[i] = f"B-{sentiment.upper()}"
            else:
                token_labels[i] = f"I-{sentiment.upper()}"

    return [label2id.get(lbl, 0) for lbl in token_labels]


# ----------------------------
# GROUP DATA BY REVIEW
# ----------------------------
grouped = df.groupby("review")
dataset = []

for review, group in tqdm(grouped, desc="Processing reviews"):
    phrases = list(zip(group["phrase"], group["sentiment"]))
    encoding = tokenizer(review, truncation=True, max_length=MAX_LEN, padding="max_length", return_offsets_mapping=True)
    labels = align_labels(review, phrases)

    if len(labels) < MAX_LEN:
        labels += [label2id["O"]] * (MAX_LEN - len(labels))
    labels = labels[:MAX_LEN]

    dataset.append((encoding, labels))

print(f"Prepared {len(dataset)} review samples")


# ----------------------------
# TF DATASET
# ----------------------------
def gen():
    for enc, labels in dataset:
        yield {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


output_signature = {
    "input_ids": tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32),
    "attention_mask": tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32),
    "labels": tf.TensorSpec(shape=(MAX_LEN,), dtype=tf.int32),
}

tf_dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
tf_dataset = tf_dataset.shuffle(1000).batch(BATCH_SIZE)

# ----------------------------
# MODEL
# ----------------------------
model = TFAutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

steps_per_epoch = len(dataset) // BATCH_SIZE
num_train_steps = steps_per_epoch * EPOCHS
optimizer, schedule = create_optimizer(
    init_lr=LR,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# ----------------------------
# CALLBACK for metrics
# ----------------------------
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        all_preds, all_labels = [], []
        for batch in tf_dataset.take(5):  # evaluate subset for speed
            logits = model(batch["input_ids"], attention_mask=batch["attention_mask"], training=False).logits
            preds = tf.argmax(logits, axis=-1).numpy()
            labels = batch["labels"].numpy()
            mask = batch["attention_mask"].numpy()

            for p, l, m in zip(preds, labels, mask):
                for pi, li, mi in zip(p, l, m):
                    if mi == 1:  # ignore padding
                        all_preds.append(pi)
                        all_labels.append(li)

        if all_labels:
            print("\nClassification Report (subset):")
            print(classification_report(
                all_labels,
                all_preds,
                labels=list(label2id.values()),
                target_names=label_list,
                zero_division=0
            ))


# ----------------------------
# TRAIN
# ----------------------------
model.fit(tf_dataset, epochs=EPOCHS, callbacks=[MetricsCallback()])

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")


# ----------------------------
# TEST SAMPLE
# ----------------------------
test_review = "The staff was helpful and the room was dirty"
enc = tokenizer(test_review, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="tf")
outputs = model(enc)
preds = tf.argmax(outputs.logits, axis=-1).numpy()[0]

tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
decoded = [(tok, id2label[p]) for tok, p in zip(tokens, preds) if tok not in ["[PAD]", "[CLS]", "[SEP]"]]
print("\n🔎 Sample Prediction:")
print(decoded)
