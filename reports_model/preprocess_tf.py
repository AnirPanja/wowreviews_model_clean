# reports_model/preprocess_tf.py
import json
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPORT_PATH = DATA_DIR / "labelstudio_export.json"   # label studio export - can be .json or .jsonl
OUT_DIR = DATA_DIR / "hf"
MODEL_NAME = "bert-base-uncased"  # change if you want different backbone

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def load_labelstudio_export(path: Path):
    text = path.read_text(encoding="utf-8")
    text = text.strip()
    # if starts with '[' it's a JSON array; if not treat as JSONL
    if text.startswith("["):
        items = json.loads(text)
        for it in items:
            yield it
    else:
        # JSON Lines: one JSON object per line
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def normalize_span_text(s: str) -> str:
    # strip leading/trailing whitespace and fix some common invisible chars
    if s is None:
        return s
    return s.strip()

def parse_task(task: Dict[str, Any]):
    """
    Extracts review text and annotations (spans + relations) from one Label Studio task object.
    Returns:
      text (str), spans (list of dicts like {id, start, end, text, labels}), relations (list of dicts {from_id,to_id})
    """
    data = task.get("data") or task.get("task") or {}
    # Label Studio sometimes stores text under different keys; mostly we used "review"
    # Fallback to any string field present
    text = None
    if isinstance(data, dict):
        if "review" in data:
            text = data["review"]
        else:
            # try first string field
            for k,v in data.items():
                if isinstance(v, str) and len(v) > 20:
                    text = v
                    break
    if text is None:
        # sometimes review is at top-level 'text'
        text = task.get("text") or ""
    # gather the annotations array (pick first completed annotation)
    annotations = task.get("annotations") or task.get("results") or []
    if not annotations:
        # some exports put predictions/annotations at top-level 'annotations' as empty list and 'predictions' present
        annotations = task.get("predictions") or []
    spans = []
    relations = []
    # We expect annotations to be a list — pick the first non-empty 'result'
    chosen = None
    for ann in annotations:
        if ann and ann.get("result"):
            chosen = ann
            break
    if chosen is None:
        # fallback: maybe the task itself contains 'result'
        if task.get("result"):
            chosen = {"result": task["result"]}
    if chosen:
        results = chosen.get("result", [])
        # map of id -> span dict
        span_by_id = {}
        for r in results:
            # label annotations: type 'labels'
            if r.get("type") in ("labels","label","choices"):
                v = r.get("value", {})
                # Label Studio value uses 'start'/'end' for spans; sometimes nested under 'text' or 'choices'
                start = v.get("start")
                end = v.get("end")
                text_snip = v.get("text") or v.get("labels") or v.get("name") or ""
                # If offsets missing but the object has 'original' keys:
                if start is None and "from_name" in r and r.get("origin") in ("manual", None):
                    # attempt to read 'value' keys that may be nested differently:
                    start = v.get("start_offset") or v.get("startChar") or None
                    end = v.get("end_offset") or v.get("endChar") or None
                label_names = v.get("labels") or v.get("label") or v.get("choice") or []
                # normalize
                if isinstance(label_names, str):
                    label_names = [label_names]
                label_names = [ln.strip() for ln in label_names if ln]
                text_snip = normalize_span_text(text_snip)
                span = {
                    "id": r.get("id"),
                    "start": start,
                    "end": end,
                    "text": text_snip,
                    "labels": label_names,
                    "raw": r
                }
                span_by_id[span["id"]] = span
            elif r.get("type") == "relation":
                # relation object referencing span ids
                fr = r.get("from_id")
                to = r.get("to_id")
                if fr and to:
                    relations.append({"from_id": fr, "to_id": to})
        # collect spans
        spans = list(span_by_id.values())
    # return cleaned text and spans/relations
    return text or "", spans, relations

def align_spans_to_tokens(text: str, spans: List[dict]):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True)
    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # initialize labels per token with "O"
    labels = ["O"] * len(offsets)
    # we will build a simple mapping from span id -> token start/end indices
    span_token_map = {}
    for span in spans:
        st = span.get("start")
        ed = span.get("end")
        if st is None or ed is None:
            # skip spans lacking char offsets
            continue
        tok_st = None
        tok_end = None
        for i, (a,b) in enumerate(offsets):
            # skip special tokens with (0,0)
            if a == 0 and b == 0:
                continue
            # overlap check
            if b > st and a < ed:
                if tok_st is None:
                    tok_st = i
                tok_end = i
        if tok_st is None:
            # alignment failed for this span — log and continue
            print(f"WARN: couldn't align span text='{span.get('text')}' start={st} end={ed} on review snippet: {text[:80]!r}")
            continue
        # choose label type: Aspect -> B-ASP/I-ASP ; Adjective -> B-ADJ/I-ADJ
        # Label Studio labels may be e.g. "Aspect" or "aspect" — normalize
        label_name = None
        for ln in span.get("labels", []):
            nl = ln.strip().lower()
            if "asp" in nl or "aspect" in nl or "service" in nl or "food" in nl:
                label_name = "ASP"
                break
            if "adj" in nl or "adject" in nl or nl in ("good","great","bad","dirty","clean","attentive","rude","fast","slow"):
                label_name = "ADJ"
                break
        if label_name is None:
            # if not determined, heuristically use POS? For now try infer by simple heuristics:
            txt = (span.get("text") or "").strip().lower()
            if len(txt.split()) <= 3 and txt.endswith(("ing","ed","ive","ful","less","y","ous","able")):
                label_name = "ADJ"
            else:
                label_name = "ASP"
        # assign BIO tokens
        if label_name == "ASP":
            labels[tok_st] = "B-ASP"
            for k in range(tok_st+1, tok_end+1):
                labels[k] = "I-ASP"
        else:
            labels[tok_st] = "B-ADJ"
            for k in range(tok_st+1, tok_end+1):
                labels[k] = "I-ADJ"
        # store mapping
        span_token_map[span["id"]] = {"tok_start": tok_st, "tok_end": tok_end, "label": label_name, "text": span.get("text")}
    # build example info
    example = {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokens": tokens,
        "offsets": offsets,
        "labels": labels,
        "span_token_map": span_token_map,
        "spans": spans
    }
    return example

def main():
    EXPORT = Path(EXPORT_PATH)
    if not EXPORT.exists():
        raise FileNotFoundError(f"Label Studio export not found: {EXPORT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aligned_path = OUT_DIR / "aligned_examples.jsonl"
    relations_out = OUT_DIR / "relations.jsonl"
    aligned_examples = []
    relations_export = []
    count = 0
    for task in load_labelstudio_export(EXPORT):
        text, spans, relations = parse_task(task)
        if not text:
            continue
        ex = align_spans_to_tokens(text, spans)
        # store span relation pairs as token indices if possible
        for rel in relations:
            fr = rel.get("from_id")
            to = rel.get("to_id")
            # try to map to token spans
            fr_map = ex["span_token_map"].get(fr)
            to_map = ex["span_token_map"].get(to)
            if fr_map and to_map:
                relations_export.append({
                    "task_id": task.get("id"),
                    "from_id": fr,
                    "to_id": to,
                    "from_tok": fr_map,
                    "to_tok": to_map
                })
            else:
                # save relation but mark as unresolved if tokens not found
                relations_export.append({
                    "task_id": task.get("id"),
                    "from_id": fr,
                    "to_id": to,
                    "note": "token_align_failed"
                })
        aligned_examples.append(ex)
        count += 1
    # write aligned examples
    with open(aligned_path, "w", encoding="utf-8") as wf:
        for ex in aligned_examples:
            wf.write(json.dumps(ex) + "\n")
    with open(relations_out, "w", encoding="utf-8") as wf:
        for r in relations_export:
            wf.write(json.dumps(r) + "\n")
    print(f"Wrote {len(aligned_examples)} aligned examples to {aligned_path}")
    print(f"Wrote {len(relations_export)} relations to {relations_out}")

if __name__ == "__main__":
    main()
