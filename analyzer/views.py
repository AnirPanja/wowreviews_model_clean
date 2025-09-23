# views.py
import os
import json
import logging
import re
import string
from typing import List, Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Optional: language-tool for grammar/punctuation fixing
USE_LANGTOOL = False
try:
    import language_tool_python  # type: ignore
    USE_LANGTOOL = True
except Exception:
    language_tool_python = None

# Try import spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    spacy = None
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Hugging Face / torch for neural GEC
try:
    from transformers import pipeline as hf_pipeline
    import torch
    HF_AVAILABLE = True
except Exception:
    hf_pipeline = None
    torch = None
    HF_AVAILABLE = False

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
REVIEW_MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../wowreviews_model_clean/reports_model/tripadvisor_bert_model_chunked/final")
)

REVIEW_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ----------------------------
# GLOBAL MODELS
# ----------------------------
_review_tokenizer: Optional[AutoTokenizer] = None
_review_model: Optional[TFAutoModelForSequenceClassification] = None
_spacy_nlp = None
_lang_tool = None

# Neural GEC config
USE_NEURAL_GEC = False
GEC_FORCE_ACCEPT = False
GEC_MODEL_NAME = "vennify/t5-base-grammar-correction"
_GEC_PIPELINE = None

# ----------------------------
# Constants
# ----------------------------
ADJ_TRUST = {
    "good", "bad", "clean", "dirty", "smelly", "friendly", "nice", "lovely", "terrible",
    "excellent", "great", "slow", "fast", "cheap", "expensive", "comfortable",
    "helpful", "rude", "loud", "quiet", "amazing", "missing", "poor", "delicious",
    "broken", "noisy", "spacious", "small", "hot", "cold", "wonderful", "horrible", "worst"
}

NOUN_BLACKLIST = {"thing", "things", "something", "way", "ways", "everything", "anything", "mention"}

PHRASE_REJECT_TOKENS = {"and", "or", "but", ",", ".", "the", "a", "an", "to", "for", "with", "that", "this", "these", "those", "mention", "special"}

PHRASE_CONF_THRESHOLD = 0.50  # Lowered to retain valid phrases

LEXICON_POS = {"good", "clean", "friendly", "nice", "lovely", "excellent", "great", "amazing", "delicious", "comfortable"}
LEXICON_NEG = {"bad", "dirty", "terrible", "rude", "slow", "noisy", "smelly", "missing", "horrible", "worst", "loud"}

SMELL_TERMS = {"smell", "smelly", "odor", "odour", "stink", "stinky"}

# ----------------------------
# Model loading utilities
# ----------------------------
def load_review_model():
    global _review_tokenizer, _review_model
    if _review_model is not None and _review_tokenizer is not None:
        return
    logger.info("Loading review-level model from: %s", REVIEW_MODEL_DIR)
    try:
        _review_tokenizer = AutoTokenizer.from_pretrained(REVIEW_MODEL_DIR)
        _review_model = TFAutoModelForSequenceClassification.from_pretrained(REVIEW_MODEL_DIR)
        logger.info("Loaded review-level model successfully.")
    except Exception as e:
        logger.exception("Failed to load review model from disk: %s", e)
        raise

def try_load_spacy():
    global _spacy_nlp
    if not SPACY_AVAILABLE:
        logger.info("spaCy not installed; phrase extraction will use heuristic fallback.")
        return
    if _spacy_nlp is not None:
        return
    try:
        _spacy_nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy en_core_web_sm.")
    except Exception as e:
        logger.warning("Could not load en_core_web_sm: %s. Phrase extraction will be limited.", e)
        _spacy_nlp = None

def try_load_langtool():
    global _lang_tool
    if not USE_LANGTOOL or language_tool_python is None:
        return
    if _lang_tool is not None:
        return
    try:
        _lang_tool = language_tool_python.LanguageTool("en-US")
        logger.info("Loaded LanguageTool for grammar/punctuation fixing.")
    except Exception as e:
        logger.warning("Could not initialize LanguageTool: %s", e)
        _lang_tool = None

# ----------------------------
# CLEANING
# ----------------------------
def clean_review_text(text: str) -> str:
    if not isinstance(text, str):
        logger.warning("Invalid review text type: %s", type(text))
        return ""
    
    MANUAL_FIXES = {
        "warwck": "Warwick",
        "warwickk": "Warwick",
        "throught": "through",
        "throughts": "through",
        "al ": "all ",
        "al.": "all.",
        "tapp": "tap",
        "phonecalls": "phone calls",
        "phonecalls.": "phone calls.",
        "pick 20 minutes": "waited 20 minutes",
        "checking tap tap tap": "kept tapping",
        "tap tap tap": "tapping",
        "nice hotel not nice staff hotel lovely staff quite rude": "Nice hotel but not nice staff. Hotel lovely but staff quite rude",
        "waited pick": "waited to pick",
        "al friendly": "all friendly",
        "friendly faces tiring day airport": "friendly staff after a tiring day at the airport",
    }

    def apply_manual_fixes(s: str) -> str:
        s2 = s
        for k, v in MANUAL_FIXES.items():
            if re.match(r"^[\w']+$", k.strip()):
                s2 = re.sub(rf"\b{re.escape(k)}\b", v, s2, flags=re.I)
            else:
                s2 = re.sub(re.escape(k), v, s2, flags=re.I)
        return s2

    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("—", "-").replace("–", "-").replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    t = re.sub(r'(?<=\d)\s*,\s*(?=\d)', ',', t)
    t = re.sub(r'(?<!\d)\s*,\s*(?!\d)', ', ', t)
    t = re.sub(r'(?<!\d)\s*\.\s*(?!\d)', '. ', t)

    t = re.sub(r"[!?]{2,}", "!", t)
    t = re.sub(r"\.{3,}", "...", t)
    t = re.sub(r"\.{2}", ".", t)

    t = apply_manual_fixes(t)

    # Improved sentence splitting
    sentences = []
    current = []
    words = t.split()
    for i, word in enumerate(words):
        current.append(word)
        if word.endswith(('.', '!', '?')) or (i < len(words) - 1 and words[i+1].lower() in {'and', 'but', 'however', 'then', 'so', 'because', 'also', 'for', 'while', 'though'}):
            sentence = ' '.join(current).strip(', ')
            if sentence:
                sentences.append(sentence)
            current = []
    if current:
        sentence = ' '.join(current).strip(', ')
        if sentence:
            sentences.append(sentence)

    t = '. '.join(s.capitalize() for s in sentences if s)
    if t and not t.endswith('.'):
        t += '.'

    words = t.split()
    deduped = []
    for i, w in enumerate(words):
        w_stripped = w.strip(string.punctuation).lower()
        skip = False
        if i > 3:
            if w_stripped == words[i-3].strip(string.punctuation).lower() or w_stripped == words[i-4].strip(string.punctuation).lower():
                skip = True
        if not skip:
            deduped.append(w)
    t = ' '.join(deduped)

    def capitalize_sentences(s):
        res = []
        for sent in re.split(r"(?<=[.!?])\s+", s):
            sent = sent.strip()
            if not sent:
                continue
            if sent.upper() == sent and len(sent) > 1:
                sent = sent.capitalize()
            sent = sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper()
            res.append(sent)
        return " ".join(res)

    t = capitalize_sentences(t)
    t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t, flags=re.I)

    try_load_langtool()
    if _lang_tool is not None:
        try:
            matches = _lang_tool.check(t)
            fixed = language_tool_python.utils.correct(t, matches)
            if fixed and isinstance(fixed, str):
                t = fixed
                logger.debug("LanguageTool applied corrections.")
        except Exception as e:
            logger.warning("LanguageTool check failed: %s", e)

    try:
        if USE_NEURAL_GEC:
            load_gec_model()
            if _GEC_PIPELINE is not None:
                logger.info("Calling neural GEC (words=%d, chars=%d)", len(t.split()), len(t))
                gec_out = neural_grammar_correct(t)
                if gec_out and isinstance(gec_out, str) and gec_out.strip():
                    if GEC_FORCE_ACCEPT or gec_out.strip() != t.strip():
                        t = gec_out
                        logger.info("Neural GEC output applied (len=%d).", len(gec_out))
                    else:
                        logger.debug("Neural GEC produced output but it matched input (no replace).")
    except Exception as e:
        logger.exception("Neural GEC step failed: %s", e)

    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not re.search(r"[.!?]$", t) and len(t.split()) > 6:
        t = t + "."

    return t

# ----------------------------
# Neural GEC helpers
# ----------------------------
def load_gec_model():
    global _GEC_PIPELINE
    if not USE_NEURAL_GEC:
        return
    if not HF_AVAILABLE:
        logger.warning("Hugging Face/torch not available; neural GEC disabled.")
        return
    if _GEC_PIPELINE is not None:
        return
    try:
        logger.info("Loading neural GEC model: %s", GEC_MODEL_NAME)
        device = 0 if torch and torch.cuda.is_available() else -1
        logger.debug("GEC device: %s", "cuda" if device == 0 else "cpu")
        _GEC_PIPELINE = hf_pipeline("text2text-generation", model=GEC_MODEL_NAME, device=device)
        logger.info("Neural GEC model loaded.")
    except Exception as e:
        logger.exception("Failed to load GEC model %s: %s", GEC_MODEL_NAME, e)
        _GEC_PIPELINE = None

def neural_grammar_correct(text: str, max_length: int = 512) -> str:
    if not USE_NEURAL_GEC or not HF_AVAILABLE or _GEC_PIPELINE is None:
        return text
    try:
        if len(text.split()) < 300:
            out = _GEC_PIPELINE(text, max_length=max_length, num_return_sequences=1)
            corrected = out[0].get("generated_text", "").strip()
            return corrected if corrected else text
        parts = re.split(r'(?<=[.!?])\s+', text)
        out_parts = []
        cur = ""
        for p in parts:
            if len((cur + " " + p).split()) <= 250:
                cur = (cur + " " + p).strip()
            else:
                if cur:
                    res = _GEC_PIPELINE(cur, max_length=max_length, num_return_sequences=1)
                    out_parts.append(res[0].get("generated_text", "").strip())
                cur = p
        if cur:
            res = _GEC_PIPELINE(cur, max_length=max_length, num_return_sequences=1)
            out_parts.append(res[0].get("generated_text", "").strip())
        corrected = " ".join(out_parts).strip()
        return corrected if corrected else text
    except Exception as e:
        logger.exception("Neural GEC failed: %s", e)
        return text

# ----------------------------
# Scoring helpers
# ----------------------------
def softmax_probs_from_logits(logits: np.ndarray) -> np.ndarray:
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    return probs

def score_phrase_with_review_model(phrase: str) -> Tuple[str, float, Dict[str, float]]:
    if not isinstance(phrase, str):
        logger.warning("Invalid phrase type: %s", type(phrase))
        return "Neutral", 0.0, {"0": 0.0, "1": 0.0, "2": 0.0}
    if _review_tokenizer is None or _review_model is None:
        raise RuntimeError("Review model not loaded")
    enc = _review_tokenizer(phrase, return_tensors="tf", truncation=True, padding="max_length", max_length=64)
    outputs = _review_model(enc)
    probs = softmax_probs_from_logits(outputs.logits)[0]
    pred_idx = int(np.argmax(probs))
    label = REVIEW_LABEL_MAP.get(pred_idx, "Neutral")
    confidence = float(probs[pred_idx])
    prob_map = {str(i): float(probs[i]) for i in range(probs.shape[0])}
    return label, confidence, prob_map

def _is_simple_negation(token, doc) -> bool:
    try:
        if any(child.dep_ == "neg" for child in token.children):
            return True
        left_i = token.i - 1
        if left_i >= 0 and doc[left_i].lower_ in {"not", "n't", "never"}:
            return True
        if any(child.dep_ == "neg" for child in token.head.children):
            return True
        for i in range(max(0, token.i - 5), min(len(doc), token.i + 5)):  # Wider window
            if doc[i].lower_ in {"not", "n't", "never"}:
                return True
    except Exception:
        return False
    return False

def strict_heuristic_phrases(review: str, max_phrases: int = 6) -> List[str]:
    if not isinstance(review, str):
        logger.warning("Invalid review type for phrase extraction: %s", type(review))
        return []
    
    review_l = review.strip().lower()
    candidates = []
    pat = re.compile(
        r"([\w\-']{1,30})\s+(?:was|is|were|are|seemed|felt|looked)\s+(?:very|quite|so|really|too|a little|rather)?\s*(not\s+)?([a-zA-Z\-']{2,30})",
        flags=re.I
    )
    for m in pat.finditer(review_l):
        subj_text = m.group(1).strip().lower()
        neg = bool(m.group(2))
        adj = m.group(3).lower()
        if adj in ADJ_TRUST and subj_text.isalpha() and subj_text not in NOUN_BLACKLIST:
            phrase = f"{'not ' if neg else ''}{adj} {subj_text}"
            candidates.append(phrase)

    sense_pat = re.compile(
        r"([\w\-']{1,30})\s+(?:smell|look|feel|taste|sound)\s+(?:very|quite|so|really|too|a little|rather)?\s*(not\s+)?([a-zA-Z\-']{2,30})",
        flags=re.I
    )
    for m in sense_pat.finditer(review_l):
        subj_text = m.group(1).strip().lower()
        neg = bool(m.group(2))
        adj = m.group(3).lower()
        if adj in ADJ_TRUST and subj_text.isalpha() and subj_text not in NOUN_BLACKLIST:
            phrase = f"{'not ' if neg else ''}{adj} {subj_text}"
            candidates.append(phrase)

    try_load_spacy()
    if _spacy_nlp is not None:
        doc = _spacy_nlp(review)
        for sent in doc.sents:
            for token in sent:
                if token.lemma_.lower() in ADJ_TRUST and token.pos_ in ("ADJ", "ADV"):
                    if token.dep_ == "amod" and token.head.pos_ in ("NOUN", "PROPN") and token.head.lemma_.lower() not in NOUN_BLACKLIST:
                        if abs(token.i - token.head.i) <= 2:
                            phrase = f"{token.text} {token.head.text}"
                            if _is_simple_negation(token, doc) or _is_simple_negation(token.head, doc):
                                phrase = "not " + phrase
                            candidates.append(phrase)
                    elif token.dep_ in ("acomp", "attr", "oprd", "advmod") and token.head.pos_ == "VERB":
                        subj_token = None
                        for child in token.head.children:
                            if child.dep_ in ("nsubj", "nsubjpass"):
                                if child.pos_ in ("NOUN", "PROPN") and child.lemma_.lower() not in NOUN_BLACKLIST:
                                    subj_token = child
                                    break
                        if subj_token and abs(token.i - subj_token.i) <= 2:  # Tighter proximity
                            phrase = f"{token.text} {subj_token.text}"
                            if _is_simple_negation(token, doc) or _is_simple_negation(subj_token, doc):
                                phrase = "not " + phrase
                            candidates.append(phrase)
        candidates = list(set(candidates))[:max_phrases]
        return [c.title() for c in candidates if c]

    words = re.findall(r"\w+", review_l)
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 in ADJ_TRUST and w2.isalpha() and w2 not in NOUN_BLACKLIST:
            candidates.append(f"{w1} {w2}")
        if w1.isalpha() and w1 not in NOUN_BLACKLIST and w2 in ADJ_TRUST:
            candidates.append(f"{w2} {w1}")
    for i in range(len(words) - 2):
        if words[i] in {"not", "no", "never"} and words[i+1] in ADJ_TRUST and words[i+2].isalpha() and words[i+2] not in NOUN_BLACKLIST:
            candidates.append(f"{words[i]} {words[i+1]} {words[i+2]}")

    out = []
    seen = set()
    for c in candidates:
        norm = " ".join(c.split())
        if norm in seen:
            continue
        seen.add(norm)
        parts = norm.split()
        if len(parts) < 2 or len(parts) > 3:
            continue
        title = " ".join(p.capitalize() for p in parts)
        out.append(title)
        if len(out) >= max_phrases:
            break
    return out[:max_phrases]

def normalize_phrase_to_adj_noun(phrase: str, review_text: str = None) -> str:
    if not isinstance(phrase, str):
        logger.warning("Invalid phrase type for normalization: %s", type(phrase))
        return ""
    
    p = phrase.strip().lower()
    tokens = re.findall(r"\w+", p)
    if not tokens:
        return ""
    if len(tokens) > 6:
        return ""
    
    if tokens[0] in {"not", "no", "never"} and len(tokens) >= 3:
        adj = tokens[1].capitalize()
        noun = tokens[2].capitalize()
        return f"Not {adj} {noun}"
    
    if any(t in SMELL_TERMS for t in tokens):
        noun_candidate = None
        for i, tk in enumerate(tokens):
            if tk in SMELL_TERMS:
                if i - 1 >= 0 and tokens[i - 1].isalpha() and tokens[i - 1] not in ADJ_TRUST:
                    noun_candidate = tokens[i - 1]
                elif i + 1 < len(tokens) and tokens[i + 1].isalpha() and tokens[i + 1] not in NOUN_BLACKLIST:
                    noun_candidate = tokens[i + 1]
                break
        if not noun_candidate and review_text and SPACY_AVAILABLE and _spacy_nlp:
            doc = _spacy_nlp(review_text)
            for sent in doc.sents:
                if any(s in sent.text.lower() for s in SMELL_TERMS):
                    nouns = [t.text for t in sent if t.pos_ in ("NOUN", "PROPN") and t.lemma_.lower() not in NOUN_BLACKLIST]
                    if nouns:
                        noun_candidate = nouns[0].lower()
                        break
            if not noun_candidate:
                nouns = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN") and t.lemma_.lower() not in NOUN_BLACKLIST]
                if nouns:
                    noun_candidate = nouns[0].lower()
        if noun_candidate:
            if "smelly" in tokens or any(t in SMELL_TERMS for t in tokens):
                return f"Smelly {noun_candidate.capitalize()}"
            return f"Bad {noun_candidate.capitalize()}"
        return "Bad Smell"
    
    if len(tokens) >= 2 and tokens[0] in ADJ_TRUST and tokens[1].isalpha():
        return " ".join([tokens[0].capitalize(), tokens[1].capitalize()])
    
    adj_found = None
    noun_found = None
    for i, tk in enumerate(tokens):
        if adj_found is None and tk in ADJ_TRUST:
            adj_found = tk
            if i + 1 < len(tokens) and tokens[i + 1].isalpha():
                noun_found = tokens[i + 1]
                break
            if i - 1 >= 0 and tokens[i - 1].isalpha():
                noun_found = tokens[i - 1]
                break
    if adj_found and noun_found:
        return f"{adj_found.capitalize()} {noun_found.capitalize()}"
    
    if review_text and SPACY_AVAILABLE and _spacy_nlp:
        doc = _spacy_nlp(review_text)
        for tok in doc:
            if tok.pos_ == "ADJ" and tok.lemma_.lower() in ADJ_TRUST:
                head = tok.head
                if head is not None and head.pos_ in ("NOUN", "PROPN") and head.lemma_.lower() not in NOUN_BLACKLIST:
                    if abs(tok.i - head.i) <= 2:
                        phrase = f"{tok.text} {head.text}"
                        if _is_simple_negation(tok, doc) or _is_simple_negation(head, doc):
                            phrase = f"not {phrase}"
                        return phrase.capitalize()
    
    parts = [w for w in re.findall(r"[A-Za-z']+", phrase) if w.lower() not in PHRASE_REJECT_TOKENS]
    meaningful = [w for w in parts if w.lower() not in {"special", "mention"}]
    if not meaningful:
        return ""
    return " ".join([w.capitalize() for w in meaningful[:2]])

def lexicon_label(phrase: str) -> Optional[str]:
    if not isinstance(phrase, str):
        logger.warning("Invalid phrase for lexicon label: %s", type(phrase))
        return None
    tokens = set(re.findall(r"\w+", phrase.lower()))
    has_neg = any(t in {"not", "no", "never"} for t in tokens)
    pos = tokens & LEXICON_POS
    neg = tokens & LEXICON_NEG
    if pos and not has_neg:
        return "Positive"
    if neg or (pos and has_neg):
        return "Negative"
    return None

def ensure_has_noun(phrase: str) -> bool:
    if not isinstance(phrase, str):
        logger.warning("Invalid phrase for noun check: %s", type(phrase))
        return False
    tokens = re.findall(r"\w+", phrase.lower())
    if not tokens:
        return False
    if SPACY_AVAILABLE and _spacy_nlp:
        doc = _spacy_nlp(phrase)
        return any(t.pos_ in ("NOUN", "PROPN") for t in doc)
    last = tokens[-1]
    return last.isalpha() and last not in ADJ_TRUST and last not in NOUN_BLACKLIST

def validate_phrase_in_context(phrase: str, review_text: str, phrase_sentiment: str) -> bool:
    if not isinstance(phrase, str) or not isinstance(review_text, str):
        logger.warning("Invalid types for context validation: phrase=%s, review_text=%s", type(phrase), type(review_text))
        return False
    if not SPACY_AVAILABLE or not _spacy_nlp:
        return True
    
    phrase_lower = phrase.lower()
    phrase_noun = re.findall(r"\w+", phrase_lower)[-1]  # Last word is typically the noun
    doc = _spacy_nlp(review_text)
    
    # Check entire review for conflicting sentiments about the same noun
    conflicting_phrases = []
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        sent_tokens = set(t.text.lower() for t in sent)
        has_neg = any(t in {"not", "n't", "never"} for t in sent_tokens)
        has_neg_adj = any(t in LEXICON_NEG for t in sent_tokens)
        has_pos_adj = any(t in LEXICON_POS for t in sent_tokens)
        
        # If phrase mentions the same noun, check for conflicts
        if phrase_noun in sent_lower:
            if phrase_sentiment == "Positive" and (has_neg or has_neg_adj):
                conflicting_phrases.append((sent.text, "negative context"))
                logger.debug("Dropping positive phrase '%s' due to negative context in sentence: %s", phrase, sent.text)
            elif phrase_sentiment == "Negative" and has_pos_adj and not has_neg:
                conflicting_phrases.append((sent.text, "positive context"))
                logger.debug("Dropping negative phrase '%s' due to positive context in sentence: %s", phrase, sent.text)
    
    if conflicting_phrases:
        logger.debug("Phrase '%s' conflicts with: %s", phrase, conflicting_phrases)
        return False
    return True

# ----------------------------
# MAIN VIEW
# ----------------------------
@csrf_exempt
def predict_sentiment(request):
    try:
        load_review_model()
        try_load_spacy()
        try_load_langtool()

        if request.method != "POST":
            return JsonResponse({"error": "Use POST method"}, status=405)

        body = json.loads(request.body.decode("utf-8"))
        reviews = body.get("reviews", None)
        if not reviews or not isinstance(reviews, list):
            return JsonResponse({"error": "Field 'reviews' (list of strings) is required"}, status=400)

        results = []
        for review in reviews:
            if not isinstance(review, str):
                logger.warning("Invalid review type: %s", type(review))
                return JsonResponse({"error": "All reviews must be strings"}, status=400)

            # ---- CLEAN REVIEW ----
            cleaned = clean_review_text(review)
            logger.debug("Cleaned review: %s", cleaned)

            # ---- Review-level sentiment ----
            enc = _review_tokenizer(cleaned, return_tensors="tf", truncation=True, padding="max_length", max_length=128)
            out = _review_model(enc)
            probs = softmax_probs_from_logits(out.logits)[0]
            pred_idx = int(np.argmax(probs))
            review_sentiment = REVIEW_LABEL_MAP.get(pred_idx, "Neutral")
            review_conf = float(probs[pred_idx])
            prob_map = {str(i): float(probs[i]) for i in range(probs.shape[0])}

            # ---- Phrase extraction ----
            raw_candidates = strict_heuristic_phrases(cleaned, max_phrases=12)
            logger.debug("Raw candidates: %s", raw_candidates)

            debug_raw = list(raw_candidates)
            kept = []
            dropped = []

            normalized = []
            for cand in raw_candidates:
                norm = normalize_phrase_to_adj_noun(cand, review_text=cleaned)
                if not isinstance(norm, str) or not norm:
                    dropped.append((cand, "normalize_empty"))
                    logger.debug("Dropped '%s': normalize_empty", cand)
                    continue
                norm_tokens = [t.lower() for t in re.findall(r"\w+", norm)]
                if any(t in PHRASE_REJECT_TOKENS for t in norm_tokens):
                    dropped.append((cand, "reject_token"))
                    logger.debug("Dropped '%s': reject_token", cand)
                    continue
                norm_length = len(norm.split())
                cleaned_length = len(cleaned.split()) if isinstance(cleaned, str) else 0
                if norm_length < 2 or norm_length > 3:
                    dropped.append((cand, "bad_length"))
                    logger.debug("Dropped '%s': bad_length", cand)
                    continue
                if not ensure_has_noun(norm):
                    dropped.append((cand, "no_noun"))
                    logger.debug("Dropped '%s': no_noun", cand)
                    continue
                if norm_length >= max(3, int(cleaned_length * 0.55)):
                    dropped.append((cand, "too_long"))
                    logger.debug("Dropped '%s': too_long", cand)
                    continue
                normalized.append(norm)
                kept.append(norm)

            if not normalized:
                normalized = strict_heuristic_phrases(cleaned, max_phrases=6)
                normalized = [normalize_phrase_to_adj_noun(x, cleaned) for x in normalized]
                normalized = [n for n in normalized if isinstance(n, str) and n and ensure_has_noun(n)]
                kept.extend(normalized)

            # Score phrases
            scored = []
            seen = set()
            normalized_unique = []
            for n in normalized:
                if isinstance(n, str) and n and n.lower() not in seen:
                    normalized_unique.append(n)
                    seen.add(n.lower())

            for phrase in normalized_unique[:12]:
                try:
                    lbl, conf, _pm = score_phrase_with_review_model(phrase)
                    logger.debug("Scored phrase '%s': %s, %.3f", phrase, lbl, conf)
                    lex = lexicon_label(phrase)
                    if lex:
                        if lbl == "Neutral" and conf < 0.85:
                            lbl = lex
                            conf = max(conf, 0.6)
                        elif lbl == "Negative" and lex == "Positive" and conf < 0.85:
                            lbl = "Positive"
                            conf = max(conf, 0.6)
                        elif lbl == "Positive" and lex == "Negative" and conf < 0.85:
                            lbl = "Negative"
                            conf = max(conf, 0.6)
                    if not validate_phrase_in_context(phrase, cleaned, lbl):
                        dropped.append((phrase, "invalid_context"))
                        logger.debug("Dropped '%s': invalid_context", phrase)
                        continue
                except Exception as e:
                    logger.exception("Scoring phrase '%s' failed: %s", phrase, e)
                    dropped.append((phrase, "scoring_failed"))
                    continue

                if not isinstance(conf, (int, float)) or conf < PHRASE_CONF_THRESHOLD:
                    # Keep low-confidence phrases if they are valid
                    if ensure_has_noun(phrase) and lexicon_label(phrase):
                        scored.append({"phrase": phrase, "sentiment": lbl, "confidence": float(conf)})
                    else:
                        dropped.append((phrase, f"low_conf_{conf:.3f}" if isinstance(conf, (int, float)) else "invalid_conf"))
                        logger.debug("Dropped '%s': low_conf_%.3f", phrase, conf)
                    continue

                tokens = set(re.findall(r"\w+", phrase.lower()))
                if any(w.lower() in {"mention", "special"} for w in tokens):
                    dropped.append((phrase, "generic_mention"))
                    logger.debug("Dropped '%s': generic_mention", phrase)
                    continue

                scored.append({"phrase": phrase, "sentiment": lbl, "confidence": float(conf)})

            scored_sorted = sorted(scored, key=lambda x: x["confidence"], reverse=True)[:6]
            keywords = scored_sorted

            if not keywords:
                # Use highest-confidence valid phrases even if below threshold
                fallback_candidates = [
                    {"phrase": p, "sentiment": lbl, "confidence": float(conf)}
                    for p, lbl, conf in [
                        score_phrase_with_review_model(p)
                        for p in normalized_unique
                        if p and ensure_has_noun(p)
                    ]
                    if lexicon_label(p) and validate_phrase_in_context(p, cleaned, lbl)
                ]
                keywords = sorted(fallback_candidates, key=lambda x: x["confidence"], reverse=True)[:6]
                if not keywords:
                    preview = cleaned.strip().split(".")[0].title() if isinstance(cleaned, str) else ""
                    keywords = [{"phrase": preview, "sentiment": review_sentiment, "confidence": review_conf}]
                    kept.append(preview)

            results.append({
                "review": review,
                "cleaned_review": cleaned,
                "sentiment": review_sentiment,
                "confidence": float(review_conf),
                "probabilities": prob_map,
                "keywords": keywords,
                "debug": {"raw": debug_raw, "kept": kept, "dropped": dropped},
            })

        return JsonResponse({"predictions": results}, status=200)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return JsonResponse({"error": f"Prediction failed: {str(e)}"}, status=500)