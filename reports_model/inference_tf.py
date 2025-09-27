# reports_model/inference_tf.py
"""
Inference utilities (TF) for token-level extraction + relation linking.
Replaces piece-level span extraction with word-level aggregation to avoid subword fragment outputs.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, TFAutoModel

ROOT = Path(__file__).resolve().parents[1]
TOKEN_MODEL_DIR = ROOT / "models" / "tf_token_bert"
REL_DIR = ROOT / "models" / "tf_relation" / "relation_model"
SEQ_LEN = 128

# ----------------- Load token classifier -----------------
print("Loading token classifier...")
tokenizer = AutoTokenizer.from_pretrained(str(TOKEN_MODEL_DIR), use_fast=True)
token_model = TFAutoModelForTokenClassification.from_pretrained(str(TOKEN_MODEL_DIR))
id2label = {int(k): v for k, v in token_model.config.id2label.items()}

# ----------------- Load encoder (for relation pooling) -----------------
try:
    encoder = TFAutoModel.from_pretrained(str(TOKEN_MODEL_DIR))
except Exception:
    encoder = TFAutoModel.from_pretrained("bert-base-uncased")

# ----------------- Load relation model -----------------
def load_relation_model():
    REL_KERAS = ROOT / "models" / "tf_relation" / "relation_model.keras"
    REL_DIR_LOCAL = REL_DIR
    REL_H5 = ROOT / "models" / "tf_relation" / "relation_model.h5"
    # prefer keras/h5 if present; otherwise try savedmodel dir
    if REL_H5.exists():
        try:
            m = tf.keras.models.load_model(str(REL_H5))
            print("Loaded relation model (H5):", REL_H5)
            return m
        except Exception as e:
            print("Failed to load relation .h5:", e)
    if REL_KERAS.exists():
        try:
            m = tf.keras.models.load_model(str(REL_KERAS))
            print("Loaded relation model (.keras):", REL_KERAS)
            return m
        except Exception as e:
            print("Failed to load relation .keras:", e)
    if REL_DIR_LOCAL.exists():
        try:
            m = tf.keras.models.load_model(str(REL_DIR_LOCAL))
            print("Loaded relation model (SavedModel):", REL_DIR_LOCAL)
            return m
        except Exception as e:
            # try tf.saved_model.load
            try:
                m2 = tf.saved_model.load(str(REL_DIR_LOCAL))
                print("Loaded relation model via tf.saved_model.load:", REL_DIR_LOCAL)
                return m2
            except Exception as e2:
                print("Failed to load relation model dir:", e, e2)
    print("No relation model found.")
    return None

REL_MODEL = load_relation_model()

# ----------------- Utilities: piece->word aggregation -----------------
def _aggregate_piece_labels_to_words(encodings, piece_labels, id2label):
    """
    encodings: tokenizer BatchEncoding for a single example (with offsets)
    piece_labels: numpy/list of label ids (len == token_count)
    id2label: mapping int->string
    Returns:
      list of word-level dicts: {"word_id", "start", "end", "tokens", "label"}
    """
    # wrapper to call .word_ids() robustly
    class _EncWrap:
        def __init__(self, enc):
            self._enc = enc
        def word_ids(self):
            try:
                return self._enc.word_ids(batch_index=0)
            except TypeError:
                return self._enc.word_ids()

        def __getitem__(self, key):
            return self._enc[key][0]

    enc_wrap = _EncWrap(encodings)
    try:
        word_ids = enc_wrap.word_ids()
    except Exception:
        # fallback: build pseudo word ids from offsets
        offsets = encodings["offset_mapping"][0].tolist()
        word_ids = []
        cur_w = 0
        prev_end = None
        for (s, e) in offsets:
            if s == e == 0:
                word_ids.append(None)
            else:
                if prev_end is None or s > prev_end:
                    word_ids.append(cur_w)
                    prev_end = e
                    cur_w += 1
                else:
                    word_ids.append(cur_w - 1)
                    prev_end = e

    offsets = encodings["offset_mapping"][0].tolist()
    word_map = defaultdict(lambda: {"tokens": [], "start": None, "end": None, "piece_labels": []})

    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        s, e = offsets[i]
        if word_map[wid]["start"] is None or s < word_map[wid]["start"]:
            word_map[wid]["start"] = s
        if word_map[wid]["end"] is None or e > word_map[wid]["end"]:
            word_map[wid]["end"] = e
        word_map[wid]["tokens"].append(i)
        word_map[wid]["piece_labels"].append(int(piece_labels[i]))

    words = []
    for w, info in sorted(word_map.items(), key=lambda x: x[0]):
        piece_lab_names = [id2label.get(lid, "O") for lid in info["piece_labels"]]
        chosen = "O"
        # prefer first B-*
        for pl in piece_lab_names:
            if pl.startswith("B-"):
                chosen = pl
                break
        else:
            non_o = [pl for pl in piece_lab_names if pl != "O"]
            if non_o:
                chosen = non_o[0]
            else:
                chosen = "O"
        words.append({
            "word_id": int(w),
            "start": int(info["start"]) if info["start"] is not None else 0,
            "end": int(info["end"]) if info["end"] is not None else 0,
            "tokens": list(info["tokens"]),
            "label": chosen
        })
    return words

# ----------------- Predict token labels (word-level spans) -----------------
def predict_token_labels(text: str) -> List[Dict]:
    """
    Robust pipeline:
     - tokenize with offsets
     - run token classifier -> piece-level preds
     - aggregate to word-level labels
     - convert word-level BIO -> spans with character offsets and tok indices
     - merge tiny separations (punctuation/hyphen)
    Returns: list of spans dicts {"label": "ASP"|"ADJ", "text", "start","end","tok_start","tok_end"}
    """
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    padding="max_length", max_length=SEQ_LEN, return_tensors="np")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # run token model (TF)
    logits = token_model(tf.constant(input_ids), attention_mask=tf.constant(attention_mask)).logits  # (1,L,C)
    preds = tf.math.argmax(logits, axis=-1).numpy()[0]  # (L,)

    # aggregate pieces -> words
    word_level = _aggregate_piece_labels_to_words(enc, preds, id2label)

    # build spans from word-level BIO
    spans = []
    cur_label = None
    cur_start = None
    cur_end = None
    cur_tok_start = None
    cur_tok_end = None

    for w in word_level:
        lab = w["label"] or "O"
        if lab.startswith("B-"):
            if cur_label is not None:
                span_text = text[cur_start:cur_end] if cur_end > cur_start else ""
                spans.append({
                    "label": "ASP" if "ASP" in cur_label else "ADJ",
                    "text": span_text,
                    "start": int(cur_start),
                    "end": int(cur_end),
                    "tok_start": int(cur_tok_start),
                    "tok_end": int(cur_tok_end)
                })
            cur_label = lab[2:]
            cur_start = w["start"]
            cur_end = w["end"]
            cur_tok_start = w["tokens"][0] if w["tokens"] else 0
            cur_tok_end = w["tokens"][-1] if w["tokens"] else 0
        elif lab.startswith("I-") and cur_label is not None:
            cur_end = max(cur_end, w["end"])
            cur_tok_end = w["tokens"][-1] if w["tokens"] else cur_tok_end
        else:
            if cur_label is not None:
                span_text = text[cur_start:cur_end] if cur_end > cur_start else ""
                spans.append({
                    "label": "ASP" if "ASP" in cur_label else "ADJ",
                    "text": span_text,
                    "start": int(cur_start),
                    "end": int(cur_end),
                    "tok_start": int(cur_tok_start),
                    "tok_end": int(cur_tok_end)
                })
                cur_label = None
                cur_start = None
                cur_end = None
                cur_tok_start = None
                cur_tok_end = None

    # flush trailing
    if cur_label is not None:
        span_text = text[cur_start:cur_end] if cur_end > cur_start else ""
        spans.append({
            "label": "ASP" if "ASP" in cur_label else "ADJ",
            "text": span_text,
            "start": int(cur_start),
            "end": int(cur_end),
            "tok_start": int(cur_tok_start),
            "tok_end": int(cur_tok_end)
        })

    # merge adjacent spans of same label separated only by punctuation/hyphen/space
    merged = []
    i = 0
    while i < len(spans):
        cur = spans[i].copy()
        j = i + 1
        while j < len(spans):
            nxt = spans[j]
            if nxt["label"] == cur["label"]:
                gap_text = text[cur["end"]:nxt["start"]]
                if re.match(r"^[\s\-–—,\.]*$", gap_text):
                    # merge
                    cur["end"] = nxt["end"]
                    cur["text"] = text[cur["start"]:cur["end"]].strip()
                    cur["tok_end"] = nxt["tok_end"]
                    j += 1
                    continue
            break
        merged.append(cur)
        i = j

    # final cleanup
    for sp in merged:
        sp["text"] = sp["text"].strip()
    return merged

# ----------------- Pooling utility -----------------
def pool_span_vector(sequence_output: np.ndarray, tok_start: int, tok_end: int) -> np.ndarray:
    L = sequence_output.shape[0]
    ts = max(0, min(tok_start, L - 1))
    te = max(0, min(tok_end, L - 1))
    if te < ts:
        te = ts
    vec = sequence_output[ts:te + 1].mean(axis=0)
    return vec

# ----------------- Relation linking (uses REL_MODEL if present) -----------------
def _link_with_relation_model(text: str, spans: List[Dict], threshold: float = 0.5) -> List[Tuple[str, str]]:
    if REL_MODEL is None:
        return []
    # re-tokenize to get encoder input & offsets
    enc = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=SEQ_LEN, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    enc_out = encoder(input_ids, attention_mask=attention_mask)[0].numpy()[0]  # (L,H)
    pool_vecs = []
    pair_info = []
    for asp in [s for s in spans if s["label"] == "ASP"]:
        for adj in [s for s in spans if s["label"] == "ADJ"]:
            a_ts = int(asp["tok_start"]); a_te = int(asp["tok_end"])
            j_ts = int(adj["tok_start"]); j_te = int(adj["tok_end"])
            L = enc_out.shape[0]
            if a_ts >= L or j_ts >= L:
                continue
            a_vec = pool_span_vector(enc_out, a_ts, a_te)
            j_vec = pool_span_vector(enc_out, j_ts, j_te)
            pool_vecs.append(np.concatenate([a_vec, j_vec], axis=0))
            pair_info.append((asp, adj))

    if not pool_vecs:
        return []

    X = np.vstack(pool_vecs).astype(np.float32)
    probs = None

    # Try keras .predict
    try:
        if hasattr(REL_MODEL, "predict"):
            preds = REL_MODEL.predict(X, batch_size=32)
            probs = preds[:, 1] if preds.ndim == 2 and preds.shape[1] > 1 else preds.ravel()
    except Exception:
        probs = None

    # Try saved_model signature
    if probs is None:
        try:
            sig = None
            if hasattr(REL_MODEL, "signatures"):
                sig = REL_MODEL.signatures.get("serving_default") or (list(REL_MODEL.signatures.values())[0] if REL_MODEL.signatures else None)
            if sig is not None:
                inp_name = list(sig.structured_input_signature[1].keys())[0] if len(sig.structured_input_signature[1]) else None
                if inp_name:
                    out = sig(**{inp_name: tf.constant(X)})
                else:
                    out = sig(tf.constant(X))
                first = list(out.values())[0].numpy()
                probs = first[:, 1] if first.ndim == 2 and first.shape[1] > 1 else first.ravel()
        except Exception:
            probs = None

    # Try calling as a TF module
    if probs is None:
        try:
            out = REL_MODEL(tf.constant(X))
            if isinstance(out, dict):
                first = list(out.values())[0].numpy()
            else:
                first = out.numpy()
            probs = first[:, 1] if first.ndim == 2 and first.shape[1] > 1 else first.ravel()
        except Exception as e:
            print("Relation model scoring failed (predict/signature/call):", e)
            return []

    results = []
    for (asp, adj), p in zip(pair_info, probs):
        try:
            score = float(p)
        except Exception:
            score = float(np.array(p).item())
        if score >= threshold:
            results.append((asp["text"].strip(), adj["text"].strip()))
    return results

# ----------------- Full pipeline wrapper -----------------
def extract_aspect_adjs(text: str, rel_threshold: float = 0.5) -> List[str]:
    spans = predict_token_labels(text)
    # If REL_MODEL exists try it; otherwise fallback to nearest-adj heuristic
    pairs = []
    if REL_MODEL is not None:
        try:
            pairs = _link_with_relation_model(text, spans, threshold=rel_threshold)
        except Exception as e:
            print("Relation linking failed, falling back to heuristic:", e)
            pairs = []
    if not pairs:
        aspects = [s for s in spans if s["label"] == "ASP"]
        adjs = [s for s in spans if s["label"] == "ADJ"]
        for asp in aspects:
            best = None; bestd = 1_000_000
            for adj in adjs:
                d = min(abs(adj["tok_start"] - asp["tok_start"]), abs(adj["tok_end"] - asp["tok_end"]))
                if d < bestd:
                    bestd = d; best = adj
            if best:
                pairs.append((asp["text"].strip(), best["text"].strip()))
    outputs = []
    for asp_text, adj_text in pairs:
        outputs.append(f"{adj_text} {asp_text}")
    return outputs

# ----------------- Debug helper (verbose) -----------------
def predict_debug(text: str, rel_threshold: float = 0.5):
    print("=== MODULE COMPONENTS ===")
    print("tokenizer:", "YES" if tokenizer is not None else "NO")
    print("token_model:", "YES" if token_model is not None else "NO")
    print("encoder:", "YES" if encoder is not None else "NO")
    print("REL_MODEL:", type(REL_MODEL).__name__ if REL_MODEL is not None else "None")
    print("predict_token_labels: YES")
    print("extract_aspect_adjs: YES")
    print("=========================")
    print()

    # Step 1 - tokenization
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=SEQ_LEN, return_tensors="np")
    input_ids = enc["input_ids"][0].tolist()
    attention = enc["attention_mask"][0].tolist()
    offsets = enc["offset_mapping"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("1) Tokenization + offsets")
    print(" - input_ids (len={}): {}".format(len(input_ids), input_ids))
    print(" - attention_mask:", attention)
    print(" - offsets (first 64):", offsets[:64])
    print(" - tokens (first 64):", tokens[:64])
    print(" - Reconstructed text from offsets (non-zero spans):")
    pieces = []
    for i, (s, e) in enumerate(offsets):
        tok = tokens[i]
        if s == e == 0:
            pieces.append(f"[{tok}: (0,0)]")
        else:
            substr = text[s:e]
            pieces.append(f"[{tok}: {repr(substr)}]")
    print("  ", " ".join(pieces[:64]))
    print()

    # Step 2 - run token classifier and show piece labels
    logits = token_model(tf.constant(enc["input_ids"]), attention_mask=tf.constant(enc["attention_mask"])).logits  # shape (1,L,C)
    log_np = logits.numpy()[0]
    argmax_labels = np.argmax(log_np, axis=-1).tolist()
    print("2) Running token classifier (logits -> pred labels)")
    print(" - logits shape:", log_np.shape)
    print(" - argmax labels (per token):", argmax_labels)
    print(" - token_model id2label:", token_model.config.id2label)
    pred_names = [id2label.get(int(l), "O") for l in argmax_labels]
    print(" - predicted labels (per token):", pred_names)
    print()

    # Step 3 - call predict_token_labels to get spans
    spans = predict_token_labels(text)
    print("3) Calling predict_token_labels(TEXT) to get spans (label, start-end, text)")
    for i, sp in enumerate(spans):
        print(f"   {i}. {sp['label']} [{sp['start']}:{sp['end']}] -> {repr(sp['text'])} (tok {sp['tok_start']}-{sp['tok_end']})")
    print()

    # Step 4 - encoder forward to pool vectors (if any spans)
    if spans:
        enc2 = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=SEQ_LEN)
        enc_out = encoder(enc2["input_ids"], attention_mask=enc2["attention_mask"])[0].numpy()[0]  # (L,H)
        print("4) Encoder forward to get token embeddings, then pool span vectors (if spans available)")
        for sp in spans:
            a_st = int(sp["tok_start"]); a_ed = int(sp["tok_end"])
            vec = pool_span_vector(enc_out, a_st, a_ed)
            norm = float(np.linalg.norm(vec))
            print(f"   span {repr(sp['text'])} tok [{a_st}:{a_ed}] vec shape {vec.shape} norm {norm:.4f}")
    else:
        print("4) No spans to pool.")
    print()

    # Step 5 - try relation link
    try:
        rel_pairs = _link_with_relation_model(text, spans, threshold=rel_threshold) if REL_MODEL is not None else []
        print("5) Trying _link_with_relation_model(TEXT, spans)")
        print(" - _link_with_relation_model returned pairs:", rel_pairs)
    except Exception as e:
        print("5) Relation linking raised exception:", e)
    print()

    # REL_MODEL introspection
    print("6) REL_MODEL introspection")
    if REL_MODEL is None:
        print(" - REL_MODEL is None")
    else:
        print(" - type:", type(REL_MODEL))
        print(" - has predict:", hasattr(REL_MODEL, "predict"))
        print(" - has signatures:", hasattr(REL_MODEL, "signatures"))
        print(" - has __call__:", callable(REL_MODEL))
        try:
            if hasattr(REL_MODEL, "signatures"):
                keys = list(REL_MODEL.signatures.keys()) if REL_MODEL.signatures else []
                print(" - REL_MODEL.signatures keys:", keys)
                if keys:
                    sig = REL_MODEL.signatures.get(keys[0])
                    if sig:
                        try:
                            dummy = np.zeros((1, enc_out.shape[1]*2), dtype=np.float32)
                            out = sig(tf.constant(dummy))
                            print(" - signature returned dict keys:", list(out.keys()))
                            first = list(out.values())[0].numpy()
                            print(" - first tensor shape:", first.shape)
                        except Exception as e:
                            print(" - signature call failed:", e)
        except Exception:
            pass
    print()

    # high-level pipeline outputs
    extracted_pairs = extract_aspect_adjs(text, rel_threshold=rel_threshold)
    print("=== High-level pipeline: extract_aspect_adjs / predict_debug (if available) ===")
    print("Calling extract_aspect_adjs(TEXT):")
    print(" -> extract_aspect_adjs output:", extracted_pairs)
    print()
    print("Calling predict_debug(TEXT) (more verbose pipeline debug if provided):")
    print("Spans (label, start-end, text):")
    for s in spans:
        print(f"  {s['label']} [{s['start']}:{s['end']}] -> {repr(s['text'])} (tok {s['tok_start']}-{s['tok_end']})")
    print("Pairs:", extracted_pairs)
    return extracted_pairs

# Quick demo when run directly
if __name__ == "__main__":
    demo = "The Wi-Fi kept disconnecting and was too slow to browse."
    print("Input:", demo)
    print("Output:", extract_aspect_adjs(demo))
