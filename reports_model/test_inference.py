# reports_model/ml/test_inference.py
"""
Debug script for relation/token inference pipeline.

This script will:
 - tokenize input
 - show offset mappings and token list
 - run the token classifier and show logits -> predicted labels
 - convert token labels -> spans (if predict_token_labels exists) and print spans
 - run encoder forward to obtain token embeddings and show pooled vectors for spans
 - attempt to call relation model (REL_MODEL) using the three methods we support:
    .predict(X)
    REL_MODEL.signatures['serving_default'](...)
    REL_MODEL(tf.constant(X))
 and print returned shapes/values (or exceptions).

Run:
  python reports_model/ml/test_inference.py
"""

import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

# input text to debug
TEXT = "The Wi-Fi kept disconnecting and was too slow to browse."

# try importing from package-style path first, then local
IMPORT_NAMES = [
    ("reports_model.inference_tf", "reports_model.inference_tf"),
    ("inference_tf", "inference_tf")
]

inference_mod = None
for mod_name, readable in IMPORT_NAMES:
    try:
        inference_mod = __import__(mod_name, fromlist=["*"])
        print(f"Imported inference module: {mod_name}")
        break
    except Exception as e:
        # ignore, try next
        # print(f"Failed to import {mod_name}: {e}")
        inference_mod = None

if inference_mod is None:
    print("ERROR: Could not import inference_tf module (tried package and local). Exiting.")
    sys.exit(1)

# helper to safe-get attribute
def get_attr(obj, name):
    return getattr(obj, name) if hasattr(obj, name) else None

# Extract commonly used objects / functions if present
tokenizer = get_attr(inference_mod, "tokenizer")
token_model = get_attr(inference_mod, "token_model")
encoder = get_attr(inference_mod, "encoder")
REL_MODEL = get_attr(inference_mod, "REL_MODEL")
predict_token_labels = get_attr(inference_mod, "predict_token_labels")
tokens_to_spans = get_attr(inference_mod, "tokens_to_spans")
pool_span_vector = get_attr(inference_mod, "pool_span_vector")
_link_with_relation_model = get_attr(inference_mod, "_link_with_relation_model")
extract_aspect_adjs = get_attr(inference_mod, "extract_aspect_adjs")
predict_debug = get_attr(inference_mod, "predict_debug")

print("\n=== MODULE COMPONENTS ===")
print("tokenizer:", "YES" if tokenizer is not None else "NO")
print("token_model:", "YES" if token_model is not None else "NO")
print("encoder:", "YES" if encoder is not None else "NO")
print("REL_MODEL:", type(REL_MODEL).__name__ if REL_MODEL is not None else "None")
print("predict_token_labels:", "YES" if predict_token_labels is not None else "NO")
print("tokens_to_spans:", "YES" if tokens_to_spans is not None else "NO")
print("_link_with_relation_model:", "YES" if _link_with_relation_model is not None else "NO")
print("extract_aspect_adjs:", "YES" if extract_aspect_adjs is not None else "NO")
print("predict_debug:", "YES" if predict_debug is not None else "NO")
print("=========================\n")

# 1) Tokenizer step
if tokenizer is None:
    print("ERROR: tokenizer missing in inference module; cannot proceed.")
    sys.exit(1)

print("1) Tokenization + offsets")
enc = tokenizer(TEXT, return_offsets_mapping=True, truncation=True, padding=True, max_length=128, return_tensors="tf")
input_ids = enc["input_ids"].numpy()[0].tolist()
attention_mask = enc["attention_mask"].numpy()[0].tolist()
offsets = enc["offset_mapping"].numpy()[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(" - input_ids (len={}):".format(len(input_ids)), input_ids[:64])
print(" - attention_mask:", attention_mask[:64])
print(" - offsets (first 64):", offsets[:64])
print(" - tokens (first 64):", tokens[:64])
print(" - Reconstructed text from offsets (non-zero spans):")
recon = []
for (s,e), tok in zip(offsets, tokens):
    if s==0 and e==0:
        recon.append(f"[{tok}: (0,0)]")
    else:
        snippet = TEXT[s:e]
        recon.append(f"[{tok}: '{snippet}']")
print("   "+ " ".join(recon[:64]))
print()

# 2) Token classifier logits / predictions
if token_model is None:
    print("WARNING: token_model missing — skipping token classifier step.")
else:
    print("2) Running token classifier (logits -> pred labels)")
    # run model
    try:
        tf_input_ids = tf.constant(enc["input_ids"])
        tf_attention = tf.constant(enc["attention_mask"])
        outputs = token_model(tf_input_ids, attention_mask=tf_attention)
        logits = outputs.logits.numpy()[0]  # (L, C)
        print(" - logits shape:", logits.shape)
        # show top few tokens logits + argmax
        argmax = np.argmax(logits, axis=-1)
        print(" - argmax labels (per token):", argmax.tolist()[:64])
        # if model has config.id2label mapping, show labels
        id2label = None
        try:
            id2label = token_model.config.id2label
            print(" - token_model id2label:", id2label)
            label_names = [id2label.get(str(int(i)), id2label.get(i, "O")) for i in argmax.tolist()]
            print(" - predicted labels (per token):", label_names[:64])
        except Exception as e:
            print(" - no id2label mapping or failed to read it:", e)
    except Exception as e:
        print("ERROR running token_model:", e)

print()

# 3) Use predict_token_labels (if available) which does offset-based span reconstruction
if predict_token_labels is None:
    print("predict_token_labels() not available — skipping span extraction using that function.")
else:
    print("3) Calling predict_token_labels(TEXT) to get spans (label, start-end, text)")
    try:
        spans = predict_token_labels(TEXT)
        print(" - spans returned (count={}):".format(len(spans)))
        for i, s in enumerate(spans):
            print(f"   {i}. {s['label']} [{s['start']}:{s['end']}] -> {repr(s['text'])} (tok {s.get('tok_start')}-{s.get('tok_end')})")
    except Exception as e:
        print("ERROR in predict_token_labels:", e)
        spans = []

print()

# 4) encoder outputs and pooled vectors for spans (if encoder exists)
if encoder is None:
    print("encoder not available — can't compute pooled span vectors.")
else:
    print("4) Encoder forward to get token embeddings, then pool span vectors (if spans available)")
    try:
        enc_out = encoder(enc["input_ids"], attention_mask=enc["attention_mask"])[0].numpy()[0]  # (L,H)
        print(" - encoder output shape (L,H):", enc_out.shape)
    except Exception as e:
        print(" - encoder forward failed:", e)
        enc_out = None

    if enc_out is not None and spans:
        print(" - pooling vectors for each span and printing vector norms")
        pooled_list = []
        for s in spans:
            ts = int(s.get("tok_start", 0))
            te = int(s.get("tok_end", ts))
            if ts >= enc_out.shape[0]:
                print(f"   span tok_start {ts} out of range (L={enc_out.shape[0]}) — skipping")
                continue
            te = min(te, enc_out.shape[0]-1)
            vec = enc_out[ts:te+1].mean(axis=0)
            pooled_list.append((s, vec))
            print(f"   span '{s['text']}' tok [{ts}:{te}] vec shape {vec.shape} norm {np.linalg.norm(vec):.4f}")
    else:
        print(" - no spans to pool (or encoder unavailable).")

print()

# 5) Try relation model scoring with pooled vectors using _link_with_relation_model (if available)
if _link_with_relation_model is not None:
    print("5) Trying _link_with_relation_model(TEXT, spans)")
    try:
        rel_pairs = _link_with_relation_model(TEXT, spans, threshold=0.1)
        print(" - _link_with_relation_model returned pairs:", rel_pairs)
    except Exception as e:
        print(" - _link_with_relation_model raised exception:", e)
else:
    print("_link_with_relation_model not available — trying lower-level REL_MODEL checks")

# 6) Inspect REL_MODEL object (if present) and try different invocation methods
if REL_MODEL is not None:
    print("\n6) REL_MODEL introspection")
    try:
        print(" - type:", type(REL_MODEL))
        # list top-level attributes we care about
        attrs = ["predict", "signatures", "__call__", "save", "variables"]
        for a in attrs:
            print(f"   has {a}: {hasattr(REL_MODEL, a)}")
        # if keras-style predict available, attempt a small dummy call with random data
        dummy_X = None
        if enc_out is not None and len(enc_out.shape) == 2:
            H = enc_out.shape[1]
            dummy_X = np.random.randn(1, 2 * H).astype(np.float32)
        else:
            dummy_X = np.random.randn(1, 1536).astype(np.float32)
        # Try .predict
        if hasattr(REL_MODEL, "predict"):
            try:
                print("   Trying REL_MODEL.predict(dummy_X)...")
                out = REL_MODEL.predict(dummy_X, batch_size=1)
                print("   => predict output shape/type:", type(out), getattr(out, "shape", None))
            except Exception as e:
                print("   => predict failed:", e)
        # Try signatures
        if hasattr(REL_MODEL, "signatures"):
            try:
                print("   REL_MODEL.signatures keys:", list(REL_MODEL.signatures.keys()))
                sig = REL_MODEL.signatures.get("serving_default") or (list(REL_MODEL.signatures.values())[0] if REL_MODEL.signatures else None)
                if sig is not None:
                    print("   Trying signature call with tf.constant(dummy_X)...")
                    inp = tf.constant(dummy_X)
                    try:
                        out = sig(inp)
                        print("   => signature returned dict keys:", list(out.keys()))
                        first = list(out.values())[0].numpy()
                        print("   => first tensor shape:", first.shape)
                    except Exception as e:
                        print("   => signature call failed:", e)
            except Exception as e:
                print("   reading signatures failed:", e)
        # Try calling object directly
        try:
            print("   Trying direct call REL_MODEL(tf.constant(dummy_X)) ...")
            out = REL_MODEL(tf.constant(dummy_X))
            if isinstance(out, dict):
                key = list(out.keys())[0]
                arr = out[key].numpy()
                print("   => returned dict; first key:", key, "shape:", arr.shape)
            else:
                arr = out.numpy()
                print("   => returned tensor shape:", arr.shape)
        except Exception as e:
            print("   => direct call failed:", e)
    except Exception as e:
        print("REL_MODEL introspection failed:", e)
else:
    print("No REL_MODEL present (None)")

print("\n=== High-level pipeline: extract_aspect_adjs / predict_debug (if available) ===")
if extract_aspect_adjs is not None:
    try:
        print("Calling extract_aspect_adjs(TEXT):")
        out = extract_aspect_adjs(TEXT)
        print(" -> extract_aspect_adjs output:", out)
    except Exception as e:
        print(" -> extract_aspect_adjs raised:", e)
else:
    print("extract_aspect_adjs not available.")

if predict_debug is not None:
    try:
        print("\nCalling predict_debug(TEXT) (more verbose pipeline debug if provided):")
        dbg = predict_debug(TEXT)
        print(" -> predict_debug returned:", dbg)
    except Exception as e:
        print(" -> predict_debug raised:", e)

print("\nDone. Use the printed information above to inspect token pieces, offsets, predicted BIO labels, spans and relation model behaviour.")
