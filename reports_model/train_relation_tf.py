# reports_model/train_relation_tf.py
"""
Relation trainer that reads:
 - reports_model/data/hf/aligned_examples.jsonl
 - reports_model/data/hf/relations.jsonl
and trains a small TF MLP to predict whether an (asp_span, adj_span) pair is linked.
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parents[1]
HF_DIR = ROOT / "data" / "hf"
ALIGNED_FILE = HF_DIR / "aligned_examples.jsonl"
REL_FILE = HF_DIR / "relations.jsonl"
MODEL_NAME = "bert-base-uncased"
OUT_DIR = ROOT / "models" / "tf_relation"

print("Loading tokenizer + encoder...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
encoder = TFAutoModel.from_pretrained(MODEL_NAME)  # used to produce token embeddings

def load_aligned_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def load_relations(path):
    rels = []
    if not path.exists():
        return rels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rels.append(json.loads(line))
    return rels

def build_indexed_map(aligned_examples):
    """
    Build a dict: task_id (if available) or index -> example object
    Also extract span_token_map for each example, mapping span_id -> tok_start/tok_end/type
    """
    idx_map = {}
    for i, ex in enumerate(aligned_examples):
        # try to find a task id if present in original preprocess (some entries include no task id)
        # We can't rely on task id always; use index as fallback
        task_id = ex.get("task_id") or ex.get("id") or ex.get("data", {}).get("id") or i
        # span_token_map created during preprocess: mapping span_id -> {'tok_start','tok_end','label','text'}
        span_token_map = ex.get("span_token_map", {})
        # also build quick lists of spans by type
        asps = []
        adjs = []
        for sid, sm in span_token_map.items():
            label = sm.get("label", "").upper()
            if label == "ASP":
                asps.append((sid, sm["tok_start"], sm["tok_end"]))
            elif label == "ADJ":
                adjs.append((sid, sm["tok_start"], sm["tok_end"]))
            else:
                # fallback: infer by original span text heuristics
                # treat unknown as aspect
                asps.append((sid, sm["tok_start"], sm["tok_end"]))
        idx_map[str(task_id)] = {
            "index": i,
            "example": ex,
            "asps": asps,
            "adjs": adjs
        }
    return idx_map

def build_relation_training_pairs(aligned_examples, relations):
    """
    Returns lists:
      - inputs: a list of dicts {input_ids, attention_mask, asp_span (start,end), adj_span (start,end)}
      - labels: 1 for positive, 0 for negative
    Positive pairs come from relations.jsonl; negatives from pairing asp with other adj spans in same example
    """
    idx_map = build_indexed_map(aligned_examples)
    inputs = []
    labels = []

    # Build positives from relations file (if it contains token-aligned info)
    for r in relations:
        # relation entries may have either token mappings or only ids.
        task_id = r.get("task_id") or r.get("text_id") or r.get("task") or r.get("task_id")
        if task_id is None:
            # try picking the single example if only one exists
            if len(aligned_examples) == 1:
                task_id = str(list(idx_map.keys())[0])
            else:
                continue
        task_key = str(task_id)
        if task_key not in idx_map:
            # sometimes task ids in relations are numeric but keys are strings; try int->str
            task_key = str(int(task_id)) if isinstance(task_id, (float,int)) else task_key
            if task_key not in idx_map:
                # skip if we cannot find corresponding example
                continue
        example_entry = idx_map[task_key]
        ex = example_entry["example"]
        # Extract tokens/attention from example
        input_ids = ex["input_ids"][:128]
        attention_mask = ex["attention_mask"][:128]
        # Determine token spans for from/to
        from_tok = r.get("from_tok") or r.get("from_tok_span") or r.get("from")
        to_tok = r.get("to_tok") or r.get("to_tok_span") or r.get("to")
        # sometimes preprocess saved 'from_tok' as a simple dict with 'tok_start'/'tok_end'
        def tokpair_from_map(m):
            if not m:
                return None
            if isinstance(m, dict):
                # m might be {'tok_start':..,'tok_end':..} or {'tok_start':..,'tok_end':..,'text':..}
                a = m.get("tok_start") or m.get("start") or m.get("from_tok_start")
                b = m.get("tok_end") or m.get("end") or m.get("from_tok_end")
                if a is not None and b is not None:
                    return (int(a), int(b))
            # if m is a mapping like {'tok_start':..}
            return None

        fr = tokpair_from_map(from_tok)
        to = tokpair_from_map(to_tok)

        # If token spans weren't present in relations.jsonl, try to map from span ids using example.span_token_map
        if fr is None or to is None:
            span_map = ex.get("span_token_map", {})
            fr_id = r.get("from_id")
            to_id = r.get("to_id")
            if fr_id and str(fr_id) in span_map:
                fr = (span_map[str(fr_id)]["tok_start"], span_map[str(fr_id)]["tok_end"])
            if to_id and str(to_id) in span_map:
                to = (span_map[str(to_id)]["tok_start"], span_map[str(to_id)]["tok_end"])

        if fr and to:
            inputs.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "asp": fr,   # note: from_id typically is adjective -> aspect mapping in your export; adjust if reversed
                "adj": to
            })
            labels.append(1)

    # If we found no positives via relations (fallback), try to create positives using adjacency heuristic:
    if len(labels) == 0:
        print("No explicit positives found in relations.jsonl — building positives using heuristic (closest adj to each asp).")
        for k, v in idx_map.items():
            ex = v["example"]
            input_ids = ex["input_ids"][:128]
            attention_mask = ex["attention_mask"][:128]
            asps = v["asps"]
            adjs = v["adjs"]
            for a_sid, a_st, a_ed in asps:
                # find nearest adj by token distance
                if not adjs:
                    continue
                best = None; bestd = 9999
                for j_sid, j_st, j_ed in adjs:
                    d = abs(j_st - a_st)
                    if d < bestd:
                        bestd = d; best = (j_st, j_ed)
                if best is not None:
                    inputs.append({"input_ids": input_ids, "attention_mask": attention_mask, "asp": (a_st, a_ed), "adj": best})
                    labels.append(1)

    # Build negatives: for each positive, pair the same asp with other adjs (if any) in same example that aren't the true adj.
    # Also supplement with random negative sampling.
    neg_count = 0
    max_neg_per_example = 3
    # Build a quick map for positives to avoid accidental duplication
    pos_pairs = set()
    for inp, lab in zip(inputs, labels):
        if lab == 1:
            pos_pairs.add((tuple(inp["asp"]), tuple(inp["adj"])))
    # Now generate negatives:
    for k, v in idx_map.items():
        ex = v["example"]
        input_ids = ex["input_ids"][:128]
        attention_mask = ex["attention_mask"][:128]
        asps = v["asps"]
        adjs = v["adjs"]
        for a_sid, a_st, a_ed in asps:
            cnt = 0
            for j_sid, j_st, j_ed in adjs:
                pair = ((a_st, a_ed), (j_st, j_ed))
                if pair in pos_pairs:
                    continue
                inputs.append({"input_ids": input_ids, "attention_mask": attention_mask, "asp": (a_st, a_ed), "adj": (j_st, j_ed)})
                labels.append(0)
                cnt += 1
                neg_count += 1
                if cnt >= max_neg_per_example:
                    break

    print(f"Built {sum(1 for l in labels if l==1)} positive and {sum(1 for l in labels if l==0)} negative examples (total {len(labels)})")
    return inputs, labels

def make_tf_dataset_from_pooled(pooled_vectors, labels, batch_size=32):
    X = np.array(pooled_vectors)
    y = np.array(labels)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=42)
    train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def pool_spans_batch(encoder_out_np, spans_batch):
    """
    encoder_out_np: (B, L, H) numpy array
    spans_batch: list of (asp_start, asp_end, adj_start, adj_end) for each element in batch
    returns pooled vectors shape (B, 2*H)
    """
    pooled = []
    for i, (a_st, a_ed, j_st, j_ed) in enumerate(spans_batch):
        seq = encoder_out_np[i]
        asp_vec = seq[a_st:a_ed+1].mean(axis=0)
        adj_vec = seq[j_st:j_ed+1].mean(axis=0)
        pooled.append(np.concatenate([asp_vec, adj_vec], axis=0))
    return np.stack(pooled)

def build_and_train_model(pooled_vectors, labels):
    input_dim = pooled_vectors.shape[1]
    inp = tf.keras.Input(shape=(input_dim,), name="pooled_input")
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    X_tr, X_va, y_tr, y_va = train_test_split(pooled_vectors, labels, test_size=0.1, random_state=42)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=6, batch_size=32)
    return model

def main():
    if not ALIGNED_FILE.exists():
        raise FileNotFoundError(f"Aligned examples not found: {ALIGNED_FILE}")
    aligned_examples = load_aligned_examples(ALIGNED_FILE)
    relations = load_relations(REL_FILE)
    inputs, labels = build_relation_training_pairs(aligned_examples, relations)
    if len(labels) == 0:
        print("No training pairs created. Exiting.")
        return

    # We'll encode inputs in batches and pool span vectors
    batch_size = 16
    pooled_vectors = []
    final_labels = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        # prepare input tensors
        input_ids_batch = [b["input_ids"][:128] for b in batch]
        attention_batch = [b["attention_mask"][:128] for b in batch]
        # pad to seq len
        seq_len = 128
        input_ids_batch_padded = [ids + [tokenizer.pad_token_id]*(seq_len - len(ids)) if len(ids)<seq_len else ids[:seq_len] for ids in input_ids_batch]
        attention_batch_padded = [mask + [0]*(seq_len - len(mask)) if len(mask)<seq_len else mask[:seq_len] for mask in attention_batch]
        input_ids_arr = np.array(input_ids_batch_padded)
        attention_arr = np.array(attention_batch_padded)
        # encoder forward pass (batch)
        enc_out = encoder(input_ids_arr, attention_mask=attention_arr)[0].numpy()  # (B,L,H)
        # build span list for pooling
        spans_batch = []
        for j, b in enumerate(batch):
            a_st, a_ed = b["asp"]
            j_st, j_ed = b["adj"]
            # clamp to seq_len
            if a_st >= seq_len or j_st >= seq_len:
                # skip problematic pair
                continue
            spans_batch.append((a_st, a_ed, j_st, j_ed))
            final_labels.append(labels[i+j])
        if not spans_batch:
            continue
        pooled = pool_spans_batch(enc_out, spans_batch)
        pooled_vectors.append(pooled)
    if not pooled_vectors:
        print("No pooled vectors created (maybe token spans were out of range). Exiting.")
        return
    pooled_vectors = np.vstack(pooled_vectors)
    final_labels = np.array(final_labels)
    print("Pooled vector shape:", pooled_vectors.shape, "Labels shape:", final_labels.shape)

    # Train MLP
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # model = build_and_train_model(pooled_vectors, final_labels)
    # model.save(str(OUT_DIR / "relation_model"))
    # print("Saved relation model to", OUT_DIR / "relation_model")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_and_train_model(pooled_vectors, final_labels)

    # Save as single-file Keras format (.keras) to avoid SavedModel graph-serialization issues
    # out_path = OUT_DIR / "relation_model.keras"
    # model.save(str(out_path))
    # print("Saved relation model to", out_path)
    out_path = OUT_DIR / "relation_model"
    model.export(out_path)   # ✅ TF SavedModel export
    print("Saved relation model to", out_path)
    
if __name__ == "__main__":
    main()