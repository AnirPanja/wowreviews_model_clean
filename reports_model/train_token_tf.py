# your_package/ml/train_token_tf.py
import json, os
from pathlib import Path
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "hf" / "aligned_examples.jsonl"
OUT_DIR = ROOT / "models" / "tf_token_bert"

MODEL_NAME = "bert-base-uncased"
label_list = ["O", "B-ASP", "I-ASP", "B-ADJ", "I-ADJ"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def load_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def build_dataset(examples, seq_len=128):
    X_ids = []
    X_mask = []
    y = []
    for ex in examples:
        input_ids = ex["input_ids"][:seq_len]
        attention_mask = ex["attention_mask"][:seq_len]
        labels = ex["labels"][:seq_len]
        # padding
        L = len(input_ids)
        if L < seq_len:
            pad_len = seq_len - L
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + ["O"] * pad_len
        lab_ids = [label2id.get(l, label2id["O"]) for l in labels]
        X_ids.append(input_ids)
        X_mask.append(attention_mask)
        y.append(lab_ids)
    return np.array(X_ids), np.array(X_mask), np.array(y)

def create_tf_dataset(X_ids, X_mask, y, batch_size=8, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(({"input_ids": X_ids, "attention_mask": X_mask}, y))
    if shuffle:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    examples = load_examples(DATA_FILE)
    if not examples:
        raise SystemExit("No aligned examples found. Run preprocess first.")
    train_ex, val_ex = train_test_split(examples, test_size=0.1, random_state=42)
    X_tr_ids, X_tr_mask, y_tr = build_dataset(train_ex, seq_len=128)
    X_va_ids, X_va_mask, y_va = build_dataset(val_ex, seq_len=128)
    train_ds = create_tf_dataset(X_tr_ids, X_tr_mask, y_tr, batch_size=8)
    val_ds = create_tf_dataset(X_va_ids, X_va_mask, y_va, batch_size=8, shuffle=False)

    model = TFAutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = str(OUT_DIR / "best_model")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, save_weights_only=False, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=4, callbacks=callbacks)
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print("Saved token model to", OUT_DIR)

if __name__ == "__main__":
    main()
