

# import os
# import json
# import re
# import string
# import ast
# from typing import List, Tuple, Dict, Optional

# import numpy as np
# import tensorflow as tf
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# # ----------------------------
# # OpenAI config (use env var - safer)
# # ----------------------------
# # Set OPENAI_API_KEY in your environment for real calls:
# #   export OPENAI_API_KEY="sk-..."
# # If not set, OpenAI calls will be disabled and deterministic fallback will be used.
# OPENAI_API_KEY = 'Your API Key'  # prefer env var
# OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
# USE_OPENAI = bool(OPENAI_API_KEY)
# openai_client = None
# if USE_OPENAI:
#     try:
#         # new v1+ client
#         from openai import OpenAI  # type: ignore
#         openai_client = OpenAI(api_key=OPENAI_API_KEY)
#         print("DEBUG: OpenAI client initialized.")
#     except Exception as e:
#         print("DEBUG: OpenAI client init failed:", e)
#         openai_client = None
#         USE_OPENAI = True
# else:
#     print("DEBUG: OPENAI_API_KEY not set — OpenAI disabled, using deterministic fallback.")

# # ----------------------------
# # Config / model paths
# # ----------------------------
# BASE_DIR = os.path.dirname(__file__)
# REVIEW_MODEL_DIR = os.path.abspath(
#     os.path.join(BASE_DIR, "../../wowreviews_model_clean/reports_model/tripadvisor_bert_model_chunked/final")
# )
# REVIEW_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# # ----------------------------
# # Globals
# # ----------------------------
# _review_tokenizer: Optional[AutoTokenizer] = None
# _review_model: Optional[TFAutoModelForSequenceClassification] = None
# _spacy_nlp = None
# _lang_tool = None

# # ----------------------------
# # Tunables/constants
# # ----------------------------
# ADJ_TRUST = {
#     "good", "bad", "clean", "dirty", "smelly", "friendly", "nice", "lovely", "terrible",
#     "excellent", "great", "slow", "fast", "cheap", "expensive", "comfortable",
#     "helpful", "rude", "loud", "quiet", "amazing", "missing", "poor", "delicious",
#     "broken", "noisy", "spacious", "small", "hot", "cold", "wonderful", "horrible", "worst", "faulty", "incorrect", "limited", "awful", "decent", "well"
# }
# NOUN_BLACKLIST = {"thing", "things", "something", "way", "ways", "everything", "anything", "mention", "special"}
# PHRASE_REJECT_TOKENS = {"and", "or", "but", ",", ".", "the", "a", "an", "to", "for", "with", "that", "this", "these", "those"}
# DETS_AND_CONJ = {"the", "a", "an", "this", "that", "these", "those", "and", "but", "or"}
# PHRASE_CONF_THRESHOLD = 0.35
# LEXICON_POS = {"good", "clean", "friendly", "nice", "lovely", "excellent", "great", "amazing", "delicious", "comfortable", "well", "decent"}
# LEXICON_NEG = {"bad", "dirty", "terrible", "rude", "slow", "noisy", "smelly", "missing", "poor", "horrible", "worst", "loud", "faulty", "incorrect", "limited", "awful"}
# IMPLICIT_NEGATION_TERMS = {"needs to improve", "should not be", "lacking", "needs correction", "not available", "needs to be revamped", "needs attention", "more options need"}
# SMELL_TERMS = {"smell", "smelly", "odor", "odour", "stink", "stinky"}

# CANONICAL_MAP = {
#     "billing": "billing", "bill": "billing", "payment": "billing", "pay": "billing",
#     "staff": "staff", "service": "staff", "desk clerk": "staff", "desk": "staff",
#     "restroom": "restrooms", "restrooms": "restrooms", "toilet": "restrooms",
#     "stall": "restrooms", "stalls": "restrooms",
#     "room": "rooms", "rooms": "rooms",
#     "food quality": "food", "food": "food", "menu": "food", "breakfast": "breakfast",
#     "parking": "parking", "music": "music", "noise": "noise",
#     "reservation": "reservation", "tea": "beverages", "coffee": "beverages",
#     "movie": "movie", "film": "movie", "luxury vibe": "luxury vibe", "luxury": "luxury vibe"
# }
# STOP_TOPIC_TOKENS = {"is", "not", "got", "home", "in", "the", "a", "an", "and", "for", "of", "to", "it", "this", "that", "still", "please", "very", "first", "day", "mention", "special", "isnt", "wasnt", "ok"}

# # ----------------------------
# # Loaders
# # ----------------------------
# def load_review_model():
#     global _review_tokenizer, _review_model
#     if _review_model is not None and _review_tokenizer is not None:
#         return
#     print(f"DEBUG: Loading review-level model from: {REVIEW_MODEL_DIR}")
#     _review_tokenizer = AutoTokenizer.from_pretrained(REVIEW_MODEL_DIR)
#     _review_model = TFAutoModelForSequenceClassification.from_pretrained(REVIEW_MODEL_DIR)
#     print("DEBUG: Review model loaded.")

# # optional spaCy/langtool loaders (kept simple; not mandatory)
# def try_load_spacy():
#     global _spacy_nlp
#     try:
#         import spacy
#         if _spacy_nlp is None:
#             _spacy_nlp = spacy.load("en_core_web_sm")
#             print("DEBUG: spaCy loaded.")
#     except Exception as e:
#         _spacy_nlp = None
#         print("DEBUG: spaCy not available:", e)

# def try_load_langtool():
#     global _lang_tool
#     try:
#         import language_tool_python
#         if _lang_tool is None:
#             _lang_tool = language_tool_python.LanguageTool("en-US")
#             print("DEBUG: LanguageTool loaded.")
#     except Exception as e:
#         _lang_tool = None
#         print("DEBUG: language_tool_python not available:", e)

# # ----------------------------
# # Cleaning (same as original)
# # ----------------------------
# def clean_review_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     MANUAL_FIXES = {
#         "warwck": "Warwick", "warwickk": "Warwick",
#         "throught": "through", "throughts": "through",
#         "al ": "all ", "al.": "all.", "tapp": "tap",
#         "phonecalls": "phone calls", "phonecalls.": "phone calls.",
#         "pick 20 minutes": "waited 20 minutes",
#         "checking tap tap tap": "kept tapping", "tap tap tap": "tapping",
#         "nice hotel not nice staff hotel lovely staff quite rude": "Nice hotel but not nice staff. Hotel lovely but staff quite rude",
#         "waited pick": "waited to pick", "al friendly": "all friendly",
#         "friendly faces tiring day airport": "friendly staff after a tiring day at the airport",
#         "phone numbers booking confirmation email needs to be corrected": "Phone numbers in booking confirmation email need correction",
#         "needs to be revamped": "needs revamping"
#     }
#     def apply_manual_fixes(s):
#         s2 = s
#         for k, v in MANUAL_FIXES.items():
#             if re.match(r"^[\w']+$", k.strip()):
#                 s2 = re.sub(rf"\b{re.escape(k)}\b", v, s2, flags=re.I)
#             else:
#                 s2 = re.sub(re.escape(k), v, s2, flags=re.I)
#         return s2

#     t = text.strip()
#     t = re.sub(r"\s+", " ", t)
#     t = t.replace("—", "-").replace("–", "-").replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
#     t = re.sub(r'(?<=\d)\s*,\s*(?=\d)', ',', t)
#     t = re.sub(r'(?<!\d)\s*,\s*(?!\d)', ', ', t)
#     t = re.sub(r'(?<!\d)\s*\.\s*(?!\d)', '. ', t)
#     t = re.sub(r"[!?]{2,}", "!", t)
#     t = re.sub(r"\.{3,}", "...", t)
#     t = apply_manual_fixes(t)

#     if re.search(r'\d+\.', t):
#         parts = re.split(r'(\d+\.\s*)', t)
#         sentences = []
#         for i, part in enumerate(parts):
#             if re.match(r'\d+\.\s*', part):
#                 continue
#             if part.strip():
#                 p = part.strip()
#                 if not p.endswith('.'):
#                     p += '.'
#                 sentences.append(p.capitalize())
#         t = ': '.join(sentences)
#     else:
#         sentences = []
#         cur = []
#         words = t.split()
#         for i, w in enumerate(words):
#             cur.append(w)
#             if w.endswith(('.', '!', '?')) or (i < len(words)-1 and words[i+1].lower() in {'and','but','however','then','so','because','also','for','while','though'}):
#                 s = ' '.join(cur).strip(', ')
#                 if s:
#                     sentences.append(s)
#                 cur = []
#         if cur:
#             s = ' '.join(cur).strip(', ')
#             if s:
#                 sentences.append(s)
#         t = '. '.join(s.capitalize() for s in sentences if s)

#     words = t.split()
#     deduped = []
#     for i, w in enumerate(words):
#         w_stripped = w.strip(string.punctuation).lower()
#         skip = False
#         if i > 3:
#             if w_stripped == words[i-3].strip(string.punctuation).lower() or w_stripped == words[i-4].strip(string.punctuation).lower():
#                 skip = True
#         if not skip:
#             deduped.append(w)
#     t = ' '.join(deduped)

#     def cap_sentences(s):
#         res = []
#         for sent in re.split(r"(?<=[.!?])\s+", s):
#             sent = sent.strip()
#             if not sent:
#                 continue
#             if sent.upper() == sent and len(sent) > 1:
#                 sent = sent.capitalize()
#             sent = sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper()
#             res.append(sent)
#         return " ".join(res)
#     t = cap_sentences(t)
#     t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t, flags=re.I)

#     try_load_langtool()
#     if _lang_tool is not None:
#         try:
#             matches = _lang_tool.check(t)
#             fixed = __import__("language_tool_python").utils.correct(t, matches)
#             if fixed and isinstance(fixed, str):
#                 t = fixed
#         except Exception:
#             pass

#     t = re.sub(r"\s+([.,;:!?])", r"\1", t)
#     t = re.sub(r"\s+", " ", t).strip()
#     if not re.search(r"[.!?]$", t) and len(t.split()) > 6:
#         t = t + "."
#     return t

# # ----------------------------
# # Model scoring helpers
# # ----------------------------
# def softmax_probs_from_logits(logits: np.ndarray) -> np.ndarray:
#     return tf.nn.softmax(logits, axis=-1).numpy()

# def score_phrase_with_review_model(phrase: str) -> Tuple[str, float, Dict[str, float]]:
#     if _review_tokenizer is None or _review_model is None:
#         raise RuntimeError("Review model not loaded")
#     enc = _review_tokenizer(phrase, return_tensors="tf", truncation=True, padding="max_length", max_length=64)
#     outputs = _review_model(enc)
#     probs = softmax_probs_from_logits(outputs.logits)[0]
#     pred_idx = int(np.argmax(probs))
#     label = REVIEW_LABEL_MAP.get(pred_idx, "Neutral")
#     confidence = float(probs[pred_idx])
#     prob_map = {str(i): float(probs[i]) for i in range(probs.shape[0])}
#     return label, confidence, prob_map

# # ----------------------------
# # Phrase extraction (kept similar to original)
# # ----------------------------
# def strict_heuristic_phrases(review: str, max_phrases: int = 8) -> List[str]:
#     if not isinstance(review, str):
#         return []
#     review_l = review.strip().lower()
#     candidates = []
#     pat = re.compile(
#         r"([\w\s\-']{1,30})\s+(?:was|is|were|are|seemed|felt|looked)\s+(?:very|quite|so|really|too|a little|rather)?\s*(not\s+)?([a-zA-Z\-']{2,30})",
#         flags=re.I
#     )
#     for m in pat.finditer(review_l):
#         subj_text = m.group(1).strip().lower()
#         neg = bool(m.group(2))
#         adj = m.group(3).lower()
#         if adj in ADJ_TRUST and all(w not in NOUN_BLACKLIST for w in subj_text.split()):
#             phrase = f"{'not ' if neg else ''}{adj} {subj_text}"
#             candidates.append(phrase)
#     # additional patterns...
#     sense_pat = re.compile(
#         r"([\w\s\-']{1,30})\s+(?:smell|look|feel|taste|sound)\s+(?:very|quite|so|really|too|a little|rather)?\s*(not\s+)?([a-zA-Z\-']{2,30})",
#         flags=re.I
#     )
#     for m in sense_pat.finditer(review_l):
#         subj_text = m.group(1).strip().lower()
#         neg = bool(m.group(2))
#         adj = m.group(3).lower()
#         if adj in ADJ_TRUST and all(w not in NOUN_BLACKLIST for w in subj_text.split()):
#             phrase = f"{'not ' if neg else ''}{adj} {subj_text}"
#             candidates.append(phrase)

#     # simple fallback token pattern
#     words = re.findall(r"\w+", review_l)
#     seen = set()
#     for i in range(len(words)-1):
#         w1, w2 = words[i], words[i+1]
#         if w1 in ADJ_TRUST and w2.isalpha() and w2 not in NOUN_BLACKLIST:
#             phrase = f"{w1} {w2}"
#             if phrase not in seen:
#                 candidates.append(phrase); seen.add(phrase)
#         if w1.isalpha() and w1 not in NOUN_BLACKLIST and w2 in ADJ_TRUST:
#             phrase = f"{w2} {w1}"
#             if phrase not in seen:
#                 candidates.append(phrase); seen.add(phrase)
#     out = []
#     seen2 = set()
#     for c in candidates:
#         norm = " ".join(c.split())
#         if norm in seen2:
#             continue
#         seen2.add(norm)
#         parts = norm.split()
#         if len(parts) < 2 or len(parts) > 4:
#             continue
#         title = " ".join(p.capitalize() for p in parts)
#         out.append(title)
#         if len(out) >= max_phrases:
#             break
#     return out[:max_phrases]

# # ----------------------------
# # Normalization / lexicon / context helpers (kept similar)
# # ----------------------------
# def normalize_phrase_to_adj_noun(phrase: str, review_text: str = None) -> str:
#     if not isinstance(phrase, str):
#         return ""
#     p = phrase.strip().lower()
#     tokens = re.findall(r"\w+", p)
#     if not tokens:
#         return ""
#     if len(tokens) > 6:
#         return ""
#     if tokens[0] in {"not","no","never"} and len(tokens) >= 3:
#         adj = tokens[1].capitalize()
#         noun_words = [w for w in tokens[2:] if w not in DETS_AND_CONJ]
#         if not noun_words: return ""
#         noun = ' '.join(noun_words).capitalize()
#         return f"Not {adj} {noun}"
#     if any(t in SMELL_TERMS for t in tokens):
#         noun_candidate = None
#         for i, tk in enumerate(tokens):
#             if tk in SMELL_TERMS:
#                 if i-1 >= 0 and tokens[i-1].isalpha() and tokens[i-1] not in ADJ_TRUST:
#                     noun_candidate = tokens[i-1]
#                 elif i+1 < len(tokens) and tokens[i+1].isalpha() and tokens[i+1] not in NOUN_BLACKLIST:
#                     noun_candidate = tokens[i+1]
#                 break
#         if noun_candidate:
#             if "smelly" in tokens or any(t in SMELL_TERMS for t in tokens):
#                 return f"Smelly {noun_candidate.capitalize()}"
#             return f"Bad {noun_candidate.capitalize()}"
#         return "Bad Smell"
#     if len(tokens) >= 2 and tokens[0] in ADJ_TRUST:
#         noun_words = [w for w in tokens[1:] if w not in DETS_AND_CONJ]
#         if not noun_words: return ""
#         noun = ' '.join(noun_words).capitalize()
#         return f"{tokens[0].capitalize()} {noun}"
#     parts = [w for w in re.findall(r"[A-Za-z']+", phrase) if w.lower() not in PHRASE_REJECT_TOKENS]
#     meaningful = [w for w in parts if w.lower() not in {"special","mention"}]
#     if not meaningful: return ""
#     if review_text and any(term in review_text.lower() for term in IMPLICIT_NEGATION_TERMS):
#         noun = ' '.join(meaningful).capitalize()
#         if "quality" in noun.lower() or "menu" in noun.lower():
#             return f"Poor {noun}"
#         elif "option" in noun.lower():
#             return f"Limited {noun}"
#         elif "correction" in review_text.lower():
#             return f"Incorrect {noun}"
#         return f"Poor {noun}"
#     return " ".join([w.capitalize() for w in meaningful])

# def lexicon_label(phrase: str) -> Optional[str]:
#     tokens = set(re.findall(r"\w+", (phrase or "").lower()))
#     if not tokens:
#         return None
#     has_neg = any(t in {"not","no","never"} for t in tokens)
#     pos = tokens & LEXICON_POS
#     neg = tokens & LEXICON_NEG
#     if pos and not has_neg:
#         return "Positive"
#     if neg or (pos and has_neg):
#         return "Negative"
#     if any(term in phrase.lower() for term in IMPLICIT_NEGATION_TERMS):
#         return "Negative"
#     return None

# def ensure_has_noun(phrase: str) -> bool:
#     tokens = re.findall(r"\w+", (phrase or "").lower())
#     if not tokens:
#         return False
#     return tokens[-1].isalpha() and tokens[-1] not in ADJ_TRUST and tokens[-1] not in NOUN_BLACKLIST

# def validate_phrase_in_context(phrase: str, review_text: str, phrase_sentiment: str) -> bool:
#     # Simple default: return True (spaCy-based validation is optional)
#     return True

# # ----------------------------
# # Aggregation + deterministic fallback
# # ----------------------------
# def _priority_for_topic(neg_pct: float, total_mentions: int) -> str:
#     if neg_pct >= 30.0 or total_mentions >= 10:
#         return "High"
#     if neg_pct >= 15.0 or total_mentions >= 4:
#         return "Medium"
#     return "Low"

# def _deterministic_insight_for_topic(topic: str, total_m: int, neg_m: int, total_reviews: int) -> Dict:
#     pct_all = (total_m / total_reviews * 100.0) if total_reviews else 0.0
#     pct_neg_of_mentions = (neg_m / max(1, total_m) * 100.0) if total_m else 0.0
#     actions = []
#     t = topic.lower()
#     if "billing" in t or "payment" in t:
#         actions = [
#             "Pilot tableside/QR/mobile payments and measure checkout time",
#             "Train staff on expedited checkout and POS flows",
#             "Monitor peak-hour queue times and reassign staff"
#         ]
#     elif "staff" in t or "service" in t or "desk" in t:
#         actions = [
#             "Run customer-service training and role-play scenarios",
#             "Adjust staffing during heavy windows (hire part-timers)",
#             "Add supervision/quality-checks and gather staff feedback"
#         ]
#     elif "restroom" in t or "stall" in t:
#         actions = [
#             "Increase cleaning rounds and add checklists with timestamps",
#             "Schedule spot inspections during peak hours",
#             "Add easy reporting (QR/hotline) for maintenance"
#         ]
#     elif "food" in t or "menu" in t or "breakfast" in t:
#         actions = [
#             "Run taste tests and review suppliers",
#             "Add 3-5 vegetarian/desired options as a pilot",
#             "Train kitchen staff on consistency and daily checks"
#         ]
#     elif "parking" in t:
#         actions = [
#             "List nearby parking options and add to booking emails",
#             "Negotiate shared parking with nearby businesses",
#             "Consider valet during peak times"
#         ]
#     elif "wifi" in t or "wi-fi" in t or "connect" in t or "connectivity" in t:
#         actions = [
#             "Audit network hardware and ISP performance; run speed tests",
#             "Add redundancy or boost Wi-Fi capacity in key areas",
#             "Provide clear signage for Wi-Fi and reset instructions"
#         ]
#     elif "elevator" in t or "lift" in t:
#         actions = [
#             "Schedule immediate maintenance for elevator reliability",
#             "Add temporary signage and staff assistance when elevator is down",
#             "Track downtime metrics and escalate to engineering"
#         ]
#     else:
#         actions = [
#             "Investigate root causes and collect more representative examples",
#             "Open ops tickets and assign owners with due dates",
#             "Track a small set of metrics to verify whether fixes improve customer sentiment"
#         ]
#     priority = _priority_for_topic(pct_neg_of_mentions, total_m)
#     return {
#         "issue": topic,
#         "total_mentions": total_m,
#         "negative_mentions": neg_m,
#         "percent_of_reviews": round(pct_all, 1),
#         "percent_negative_of_neg_reviews": round(pct_neg_of_mentions, 1),
#         "priority": priority,
#         "recommended_actions": actions,
#         "source": "deterministic"
#     }

# def _canonical_topic_from_phrase(phrase: str) -> str:
#     if not phrase: return ""
#     s = phrase.lower()
#     s = re.sub(r"[^a-z0-9\s]", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     if not s: return ""
#     for k, v in CANONICAL_MAP.items():
#         if k in s:
#             return v
#     tokens = [t for t in s.split() if t not in STOP_TOPIC_TOKENS]
#     if not tokens: return ""
#     return tokens[-1]

# def _build_aggregate_topic_stats(results: List[dict]) -> Dict[str, Dict]:
#     total_reviews = len(results)
#     raw_topics = {}
#     for r in results:
#         cleaned = r.get("cleaned_review") or r.get("review") or ""
#         for kw in r.get("keywords", []):
#             phrase = (kw.get("phrase") or "").strip()
#             sentiment = kw.get("sentiment")
#             topic = _canonical_topic_from_phrase(phrase)
#             if not topic:
#                 continue
#             entry = raw_topics.setdefault(topic, {"topic": topic, "total_mentions": 0, "negative_mentions": 0, "examples": []})
#             entry["total_mentions"] += 1
#             if sentiment == "Negative":
#                 entry["negative_mentions"] += 1
#             snippet = cleaned.strip() or (r.get("review") or "").strip()
#             if snippet and snippet not in entry["examples"]:
#                 entry["examples"].append(snippet if len(snippet) <= 300 else snippet[:300] + "...")
#     # fallback scanning raw text
#     if not raw_topics:
#         for r in results:
#             txt = (r.get("cleaned_review") or r.get("review") or "").lower()
#             for k in CANONICAL_MAP.keys():
#                 if k in txt:
#                     topic = CANONICAL_MAP[k]
#                     entry = raw_topics.setdefault(topic, {"topic": topic, "total_mentions": 0, "negative_mentions": 0, "examples": []})
#                     entry["total_mentions"] += 1
#                     if r.get("sentiment") == "Negative":
#                         entry["negative_mentions"] += 1
#                     s = (r.get("cleaned_review") or r.get("review")).strip()
#                     if s and s not in entry["examples"]:
#                         entry["examples"].append(s)
#     return raw_topics

# # ----------------------------
# # OpenAI integration (robust parsing + single retry)
# # ----------------------------
# def _parse_json_from_text(text: str):
#     if not text or not isinstance(text, str):
#         return None
#     # try direct json
#     try:
#         return json.loads(text)
#     except Exception:
#         pass
#     # try to find a JSON array substring
#     start = text.find('[')
#     if start != -1:
#         end = text.rfind(']')
#         if end != -1 and end > start:
#             sub = text[start:end+1]
#             try:
#                 return json.loads(sub)
#             except Exception:
#                 try:
#                     return ast.literal_eval(sub)
#                 except Exception:
#                     pass
#     # try to find a JSON object substring
#     start = text.find('{')
#     if start != -1:
#         end = text.rfind('}')
#         if end != -1 and end > start:
#             sub = text[start:end+1]
#             try:
#                 return json.loads(sub)
#             except Exception:
#                 try:
#                     return ast.literal_eval(sub)
#                 except Exception:
#                     pass
#     return None

# def _extract_text_from_response(response):
#     """
#     Accepts multiple OpenAI response shapes and extracts textual content.
#     """
#     try:
#         # Try ChatCompletion-like shape
#         if hasattr(response, "choices") and isinstance(response.choices, list) and len(response.choices) > 0:
#             choice = response.choices[0]
#             # new client might have .message
#             if hasattr(choice, "message") and isinstance(choice.message, dict):
#                 return choice.message.get("content") or str(choice.message)
#             # older style
#             if hasattr(choice, "text"):
#                 return choice.text
#             # sometimes content is nested
#             if hasattr(choice, "message") and hasattr(choice.message, "get"):
#                 return choice.message.get("content")
#             # fallback str
#             return str(choice)
#         # direct dict style
#         if isinstance(response, dict):
#             choices = response.get("choices")
#             if isinstance(choices, list) and len(choices) > 0:
#                 ch = choices[0]
#                 if isinstance(ch, dict) and "message" in ch and isinstance(ch["message"], dict):
#                     return ch["message"].get("content")
#                 if isinstance(ch, dict) and "text" in ch:
#                     return ch.get("text")
#         # fallback string
#         return str(response)
#     except Exception as e:
#         print("DEBUG: _extract_text_from_response error:", e)
#         return None

# def generate_aggregated_insights_via_openai(topic_stats: Dict[str, Dict], total_reviews: int, max_topics: int = 8) -> Optional[List[Dict]]:
#     """
#     Patched OpenAI caller that is robust to truncated/non-JSON responses.
#     Returns list of insight dicts or None on failure.
#     """

#     if not USE_OPENAI or openai_client is None:
#         print("OpenAI not enabled or client missing; skipping OpenAI generation.")
#         return None

#     # Build items list to send
#     items = []
#     for t, data in list(topic_stats.items())[:max_topics]:
#         items.append({
#             "issue": data["topic"],
#             "total_mentions": data.get("total_mentions", 0),
#             "negative_mentions": data.get("negative_mentions", 0),
#             "examples": data.get("examples", [])[:3]
#         })

#     few_shot = [
#         {
#             "issue": "billing",
#             "percent_of_reviews": 12,
#             "percent_negative_of_neg_reviews": 68,
#             "priority": "High",
#             "recommended_actions": [
#                 "Pilot mobile/QR/tableside payments and measure checkout time",
#                 "Train staff on expedited checkout and POS flows",
#                 "Add temporary staff during lunch hours"
#             ],
#             "short_explanation": "68% of negative reviews referenced slow billing; long wait times during lunch."
#         },
#         {
#             "issue": "vegetarian options",
#             "percent_of_reviews": 6,
#             "percent_negative_of_neg_reviews": 23,
#             "priority": "Medium",
#             "recommended_actions": [
#                 "Add 3-5 vegetarian dishes as a pilot",
#                 "Highlight vegetarian options on menu",
#                 "Run a 'Meatless Monday' promotion"
#             ],
#             "short_explanation": "23% of negative feedback mentioned limited vegetarian choices."
#         }
#     ]

#     prompt = (
#         "You are an operations consultant. Given aggregated topic stats, return EXACTLY a JSON array of objects.\n"
#         "Each object must have these fields: issue (string), percent_of_reviews (number), percent_negative_of_neg_reviews (number), "
#         "priority (High/Medium/Low), recommended_actions (array of 3 concise action strings), short_explanation (1 sentence).\n"
#         "Do NOT write any extra text. Only return the JSON array.\n\n"
#         "Few-shot examples:\n" + json.dumps(few_shot, indent=2) + "\n\n"
#         "Now analyze the provided topics and stats and produce the JSON. Topics input:\n" + json.dumps(items, indent=2) + "\n"
#         f"Total reviews: {total_reviews}\n"
#     )

#     def _extract_text_and_meta(resp):
#         """
#         Tries several ways to extract text and finish_reason from the response
#         Accepts dict-like and SDK object-like responses.
#         Returns tuple (text:str or None, finish_reason:str or None)
#         """
#         try:
#             # dict-like (classic openai.ChatCompletion.create)
#             if isinstance(resp, dict):
#                 choices = resp.get("choices", [])
#                 if choices:
#                     c0 = choices[0]
#                     # message content
#                     if isinstance(c0.get("message"), dict):
#                         return c0["message"].get("content"), c0.get("finish_reason")
#                     # text fallback
#                     if "text" in c0:
#                         return c0.get("text"), c0.get("finish_reason")
#             # object-like (openai types)
#             choices = getattr(resp, "choices", None)
#             if choices and len(choices) > 0:
#                 first = choices[0]
#                 # message object
#                 msg = getattr(first, "message", None)
#                 if msg:
#                     # message might be dict-like or object-like
#                     if isinstance(msg, dict):
#                         return msg.get("content"), getattr(first, "finish_reason", None)
#                     # object-like
#                     content = getattr(msg, "get", None)
#                     if callable(content):
#                         # msg behaves like mapping
#                         try:
#                             return msg.get("content"), getattr(first, "finish_reason", None)
#                         except Exception:
#                             pass
#                     # try attribute
#                     cont = getattr(msg, "content", None)
#                     if cont is not None:
#                         return cont, getattr(first, "finish_reason", None)
#                 # fallback to text attr
#                 txt = getattr(first, "text", None)
#                 if txt is not None:
#                     return txt, getattr(first, "finish_reason", None)
#         except Exception as e:
#            print("Extraction helper failed: %s", e)
#         # last resort str()
#         try:
#             return str(resp), None
#         except Exception:
#             return None, None

#     def _call_openai(max_tokens_local=800, temperature_local=0.0):
#         """
#         Encapsulated call so we can control parameters.
#         Uses new client shape if available but falls back to legacy dict-style.
#         """
#         try:
#             # If using the new OpenAI client object where openai_client.chat.completions.create exists
#             if hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
#                 return openai_client.chat.completions.create(
#                     model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
#                     messages=[
#                         {"role": "system", "content": "You are concise and return strict JSON arrays only."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=max_tokens_local,
#                     temperature=temperature_local,
#                 )
#             # fallback: old openai package style (dict returned)
#             import openai as old_openai  # type: ignore
#             old_openai.api_key = OPENAI_API_KEY
#             return old_openai.ChatCompletion.create(
#                 model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
#                 messages=[
#                     {"role": "system", "content": "You are concise and return strict JSON arrays only."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=max_tokens_local,
#                 temperature=temperature_local,
#             )
#         except Exception as e:
#             print("OpenAI call failed: %s", e)
#             return None

#     # --- Attempts: initial, retry with larger max_tokens, then continuation ---
#     attempts = [
#         {"max_tokens": 800, "temperature": 0.0},
#         {"max_tokens": 1400, "temperature": 0.0},
#         {"max_tokens": 2000, "temperature": 0.0}
#     ]

#     for i, cfg in enumerate(attempts):
#         print("Calling OpenAI model=%s max_tokens=%d attempt=%d", os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"), cfg["max_tokens"], i+1)
#         print(f"DEBUG: Calling OpenAI attempt {i+1} max_tokens={cfg['max_tokens']}")
#         resp = _call_openai(max_tokens_local=cfg["max_tokens"], temperature_local=cfg["temperature"])
#         if resp is None:
#             print("OpenAI call returned None on attempt %d", i+1)
#             continue

#         text, finish_reason = _extract_text_and_meta(resp)
#         print("raw OpenAI response type: %s", type(resp))
#         print("finish_reason: %s", finish_reason)
#         print("DEBUG: raw response type:", type(resp), "finish_reason:", finish_reason)
#         if not text:
#             print("OpenAI returned empty text on attempt %d", i+1)
#             continue

#         # try parse JSON
#         parsed = _parse_json_from_text(text)
#         if isinstance(parsed, list):
#             out = []
#             for o in parsed:
#                 if not isinstance(o, dict):
#                     continue
#                 if "issue" in o and "recommended_actions" in o:
#                     recs = o.get("recommended_actions") or []
#                     if isinstance(recs, str):
#                         recs = [s.strip() for s in re.split(r'[\n;]', recs) if s.strip()][:3]
#                     recs = [str(x).strip() for x in recs][:3]
#                     out.append({
#                         "issue": str(o.get("issue")),
#                         "percent_of_reviews": float(o.get("percent_of_reviews", 0)),
#                         "percent_negative_of_neg_reviews": float(o.get("percent_negative_of_neg_reviews", 0)),
#                         "priority": str(o.get("priority", "Low")),
#                         "recommended_actions": recs,
#                         "short_explanation": str(o.get("short_explanation", ""))[:300],
#                         # "source": "openai"
#                     })
#             if out:
#                 print("OpenAI returned %d insights on attempt %d.", len(out), i+1)
#                 return out

#         # If not parseable, and the finish_reason indicates truncation, try to continue
#         print("Attempt %d: response not parseable as JSON.", i+1)
#         print("DEBUG: first response not parseable as JSON, finish_reason:", finish_reason)
#         # If finish_reason suggests length, try continuation step (only once)
#         if finish_reason == "length" or (text and not text.strip().endswith(("]", "}"))):
#             print("Response likely truncated (finish_reason=%s). Attempting continuation.", finish_reason)
#             # Ask model to continue and return only the JSON array. Send the partial output back to the model.
#             cont_prompt = (
#                 "The previous response was cut off. You returned a partial JSON array. "
#                 "Please continue and return ONLY the JSON array (no explanation). Previous partial output:\n\n"
#                 + text[-4000:]  # include last chunk (safe-guard)
#             )
#             # continuation call - we send minimal context to avoid exceeding token limits
#             try:
#                 if hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
#                     cont_resp = openai_client.chat.completions.create(
#                         model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
#                         messages=[
#                             {"role": "system", "content": "You are concise and return strict JSON arrays only."},
#                             {"role": "user", "content": cont_prompt}
#                         ],
#                         max_tokens=1200,
#                         temperature=0.0
#                     )
#                 else:
#                     import openai as old_openai  # type: ignore
#                     old_openai.api_key = OPENAI_API_KEY
#                     cont_resp = old_openai.ChatCompletion.create(
#                         model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
#                         messages=[
#                             {"role": "system", "content": "You are concise and return strict JSON arrays only."},
#                             {"role": "user", "content": cont_prompt}
#                         ],
#                         max_tokens=1200,
#                         temperature=0.0
#                     )
#                 cont_text, _ = _extract_text_and_meta(cont_resp)
#                 print("Continuation text (first 1000 chars): %s", repr(cont_text[:1000]) if cont_text else "None")
#                 parsed = _parse_json_from_text((text or "") + (cont_text or ""))
#                 if isinstance(parsed, list):
#                     out = []
#                     for o in parsed:
#                         if not isinstance(o, dict): continue
#                         if "issue" in o and "recommended_actions" in o:
#                             recs = o.get("recommended_actions") or []
#                             if isinstance(recs, str):
#                                 recs = [s.strip() for s in re.split(r'[\n;]', recs) if s.strip()][:3]
#                             recs = [str(x).strip() for x in recs][:3]
#                             out.append({
#                                 "issue": str(o.get("issue")),
#                                 "percent_of_reviews": float(o.get("percent_of_reviews", 0)),
#                                 "percent_negative_of_neg_reviews": float(o.get("percent_negative_of_neg_reviews", 0)),
#                                 "priority": str(o.get("priority", "Low")),
#                                 "recommended_actions": recs,
#                                 "short_explanation": str(o.get("short_explanation", ""))[:300],
#                                 "source": "openai"
#                             })
#                     if out:
#                         print("OpenAI continuation returned %d insights.", len(out))
#                         return out
#             except Exception as e:
#                 print("Continuation attempt failed: %s", e)

#     # All attempts failed
#     print("OpenAI aggregated insights failed after retries: falling back to deterministic.")
#     return None

# def build_insights_and_text(results: List[dict], max_topics: int = 8) -> Tuple[List[dict], str, bool]:
#     total_reviews = len(results)
#     raw_topics = _build_aggregate_topic_stats(results)
#     ordered = sorted(raw_topics.values(), key=lambda x: (x.get("negative_mentions", 0), x.get("total_mentions", 0)), reverse=True)

#     ai_generated = None
#     if USE_OPENAI:
#         try:
#             ai_generated = generate_aggregated_insights_via_openai({d["topic"]: d for d in ordered}, total_reviews, max_topics=max_topics)
#         except Exception as e:
#             print("DEBUG: generate_aggregated_insights_via_openai exception:", e)
#             ai_generated = None

#     insights = []
#     openai_used_flag = False
#     if ai_generated:
#         openai_used_flag = True
#         for g in ai_generated:
#             insights.append({
#                 "issue": g.get("issue"),
#                 "summary": f"{g.get('percent_of_reviews',0):.0f}% of reviews mention {g.get('issue')}.",
#                 "priority": g.get("priority"),
#                 "recommended_actions": g.get("recommended_actions"),
#                 "short_explanation": g.get("short_explanation", ""),
#                 "source": g.get("source", "openai")
#             })
#     else:
#         # deterministic fallback
#         for d in ordered[:max_topics]:
#             topic = d["topic"]
#             total_m = d.get("total_mentions", 0)
#             neg_m = d.get("negative_mentions", 0)
#             entry = _deterministic_insight_for_topic(topic, total_m, neg_m, total_reviews)
#             insights.append({
#                 "issue": entry["issue"],
#                 "summary": f"{entry['percent_of_reviews']:.0f}% of reviews mention {entry['issue']}.",
#                 "priority": entry["priority"],
#                 "recommended_actions": entry["recommended_actions"],
#                 "short_explanation": "",
#                 "source": entry.get("source", "deterministic")
#             })

#     # human-readable formatted text (clean)
#     lines = []
#     for ins in insights:
#         lines.append(f"{ins['issue'].title()}")
#         summary = ins.get("summary", "")
#         if summary:
#             lines.append(summary)
#         lines.append(f"{ins.get('priority', 'Low')} priority")
#         lines.append("Recommended Actions:")
#         for a in ins.get("recommended_actions", []):
#             lines.append(f"- {a}")
#         if ins.get("source"):
#             lines.append(f"(source: {ins.get('source')})")
#         lines.append("")  # blank line
#     ai_insights_text = "\n".join(lines).strip()
#     return insights, ai_insights_text, openai_used_flag

# # ----------------------------
# # MAIN view
# # ----------------------------
# @csrf_exempt
# def predict_sentiment(request):
#     try:
#         load_review_model()
#         try_load_spacy()
#         try_load_langtool()

#         if request.method != "POST":
#             return JsonResponse({"error": "Use POST method"}, status=405)
#         try:
#             body = json.loads(request.body.decode("utf-8"))
#         except Exception:
#             return JsonResponse({"error": "Invalid JSON body"}, status=400)

#         reviews = body.get("reviews")
#         if not isinstance(reviews, list):
#             return JsonResponse({"error": "Field 'reviews' (list of strings) is required"}, status=400)

#         results = []
#         for review in reviews:
#             if not isinstance(review, str):
#                 return JsonResponse({"error": "All reviews must be strings"}, status=400)
#             cleaned = clean_review_text(review)
#             # review-level sentiment
#             enc = _review_tokenizer(cleaned, return_tensors="tf", truncation=True, padding="max_length", max_length=128)
#             out = _review_model(enc)
#             probs = softmax_probs_from_logits(out.logits)[0]
#             pred_idx = int(np.argmax(probs))
#             review_sentiment = REVIEW_LABEL_MAP.get(pred_idx, "Neutral")
#             review_conf = float(probs[pred_idx])
#             prob_map = {str(i): float(probs[i]) for i in range(probs.shape[0])}

#             # phrase extraction
#             raw_candidates = strict_heuristic_phrases(cleaned, max_phrases=8)
#             debug_raw = list(raw_candidates)
#             kept = []; dropped = []
#             normalized = []
#             for cand in raw_candidates:
#                 norm = normalize_phrase_to_adj_noun(cand, review_text=cleaned)
#                 if not norm:
#                     dropped.append((cand, "normalize_empty")); continue
#                 norm_tokens = [t.lower() for t in re.findall(r"\w+", norm)]
#                 if any(t in PHRASE_REJECT_TOKENS for t in norm_tokens):
#                     dropped.append((cand, "reject_token")); continue
#                 norm_length = len(norm.split()); cleaned_length = len(cleaned.split())
#                 if norm_length < 2 or norm_length > 4:
#                     dropped.append((cand, "bad_length")); continue
#                 if not ensure_has_noun(norm):
#                     dropped.append((cand, "no_noun")); continue
#                 if norm_length >= max(4, int(cleaned_length * 0.55)):
#                     dropped.append((cand, "too_long")); continue
#                 normalized.append(norm); kept.append(norm)

#             if not normalized:
#                 normalized = strict_heuristic_phrases(cleaned, max_phrases=8)
#                 normalized = [normalize_phrase_to_adj_noun(x, cleaned) for x in normalized]
#                 normalized = [n for n in normalized if n and ensure_has_noun(n)]
#                 kept.extend(normalized)

#             # scoring
#             scored = []
#             seen = set()
#             normalized_unique = []
#             for n in normalized:
#                 if isinstance(n, str) and n and n.lower() not in seen:
#                     normalized_unique.append(n); seen.add(n.lower())
#             for phrase in normalized_unique[:8]:
#                 try:
#                     lbl, conf, _pm = score_phrase_with_review_model(phrase)
#                     lex = lexicon_label(phrase)
#                     if lex:
#                         if lbl == "Neutral" and conf < 0.85:
#                             lbl = lex; conf = max(conf, 0.6)
#                         elif lbl == "Negative" and lex == "Positive" and conf < 0.85:
#                             lbl = "Positive"; conf = max(conf, 0.6)
#                         elif lbl == "Positive" and lex == "Negative" and conf < 0.85:
#                             lbl = "Negative"; conf = max(conf, 0.6)
#                     if not validate_phrase_in_context(phrase, cleaned, lbl):
#                         dropped.append((phrase, "invalid_context")); continue
#                 except Exception as e:
#                     print("DEBUG: Scoring failed:", e)
#                     dropped.append((phrase, "scoring_failed")); continue
#                 if not isinstance(conf, (int,float)) or conf < PHRASE_CONF_THRESHOLD:
#                     if ensure_has_noun(phrase) and lexicon_label(phrase):
#                         scored.append({"phrase": phrase, "sentiment": lbl, "confidence": float(conf)})
#                     else:
#                         dropped.append((phrase, f"low_conf_{conf:.3f}"))
#                     continue
#                 if any(w.lower() in {"mention","special"} for w in re.findall(r"\w+", phrase.lower())):
#                     dropped.append((phrase,"generic_mention")); continue
#                 scored.append({"phrase": phrase, "sentiment": lbl, "confidence": float(conf)})

#             scored_sorted = sorted(scored, key=lambda x: x["confidence"], reverse=True)[:8]
#             keywords = scored_sorted
#             if not keywords:
#                 preview = cleaned.strip().split(".")[0].title() if cleaned else ""
#                 keywords = [{"phrase": preview, "sentiment": review_sentiment, "confidence": review_conf}]

#             results.append({
#                 "review": review,
#                 "cleaned_review": cleaned,
#                 "sentiment": review_sentiment,
#                 "confidence": float(review_conf),
#                 "probabilities": prob_map,
#                 "keywords": keywords,
#                 "debug": {"raw": debug_raw, "kept": kept, "dropped": dropped}
#             })

#         # Build aggregated insights + formatted text
#         insights, ai_insights_text, openai_used_flag = build_insights_and_text(results, max_topics=12)
#         resp_payload = {"predictions": results, "insights": insights, "ai_insights_text": ai_insights_text, "openai_used": bool(openai_used_flag)}
#         return JsonResponse(resp_payload, status=200)

#     except Exception as e:
#         print("DEBUG: Prediction failed:", e)
#         return JsonResponse({"error": f"Prediction failed: {str(e)}"}, status=500)


"""
views.py — full, replaceable file.

Replaces heuristic keyword extraction with the token-classifier + relation-linker pipeline
if available (expects reports_model/inference_tf.extract_aspect_adjs).

Keeps the review-level sentiment model, OpenAI helpers, deterministic fallback, and aggregation.
"""

import os
import json
import re
import string
import ast
from typing import List, Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ----------------------------
# OpenAI config (use env var - safer)
# ----------------------------
# Set OPENAI_API_KEY in your environment for real calls:
#   export OPENAI_API_KEY="sk-..."
OPENAI_API_KEY = 'Your API Key'  # if empty OpenAI disabled
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
USE_OPENAI = bool(OPENAI_API_KEY)
openai_client = None
if USE_OPENAI:
    try:
        # new v1+ client
        from openai import OpenAI  # type: ignore
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("DEBUG: OpenAI client initialized.")
    except Exception as e:
        print("DEBUG: OpenAI client init failed:", e)
        openai_client = None
        USE_OPENAI = True
else:
    print("DEBUG: OPENAI_API_KEY not set — OpenAI disabled, using deterministic fallback.")

# ----------------------------
# Config / model paths
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
REVIEW_MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../wowreviews_model_clean/reports_model/tripadvisor_bert_model_chunked/final")
)
REVIEW_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ----------------------------
# Try to import the token-level inference pipeline
# ----------------------------
try:
    from reports_model.inference_tf import extract_aspect_adjs, predict_debug
except Exception:
    try:
        # fallback local import (if running inside reports_model folder)
        from inference_tf import extract_aspect_adjs, predict_debug
    except Exception:
        extract_aspect_adjs = None
        predict_debug = None
        print("DEBUG: Could not import extract_aspect_adjs/predict_debug from inference_tf. Phrase extraction model will be disabled.")

# ----------------------------
# Globals
# ----------------------------
_review_tokenizer: Optional[AutoTokenizer] = None
_review_model: Optional[TFAutoModelForSequenceClassification] = None
_spacy_nlp = None
_lang_tool = None

# ----------------------------
# Tunables/constants
# ----------------------------
ADJ_TRUST = {
    "good", "bad", "clean", "dirty", "smelly", "friendly", "nice", "lovely", "terrible",
    "excellent", "great", "slow", "fast", "cheap", "expensive", "comfortable",
    "helpful", "rude", "loud", "quiet", "amazing", "missing", "poor", "delicious",
    "broken", "noisy", "spacious", "small", "hot", "cold", "wonderful", "horrible", "worst", "faulty", "incorrect", "limited", "awful", "decent", "well"
}
NOUN_BLACKLIST = {"thing", "things", "something", "way", "ways", "everything", "anything", "mention", "special"}
PHRASE_REJECT_TOKENS = {"and", "or", "but", ",", ".", "the", "a", "an", "to", "for", "with", "that", "this", "these", "those"}
DETS_AND_CONJ = {"the", "a", "an", "this", "that", "these", "those", "and", "but", "or"}
PHRASE_CONF_THRESHOLD = 0.35
LEXICON_POS = {"good", "clean", "friendly", "nice", "lovely", "excellent", "great", "amazing", "delicious", "comfortable", "well", "decent"}
LEXICON_NEG = {"bad", "dirty", "terrible", "rude", "slow", "noisy", "smelly", "missing", "poor", "horrible", "worst", "loud", "faulty", "incorrect", "limited", "awful"}
IMPLICIT_NEGATION_TERMS = {"needs to improve", "should not be", "lacking", "needs correction", "not available", "needs to be revamped", "needs attention", "more options need"}
SMELL_TERMS = {"smell", "smelly", "odor", "odour", "stink", "stinky"}

CANONICAL_MAP = {
    "billing": "billing", "bill": "billing", "payment": "billing", "pay": "billing",
    "staff": "staff", "service": "staff", "desk clerk": "staff", "desk": "staff",
    "restroom": "restrooms", "restrooms": "restrooms", "toilet": "restrooms",
    "stall": "restrooms", "stalls": "restrooms",
    "room": "rooms", "rooms": "rooms",
    "food quality": "food", "food": "food", "menu": "food", "breakfast": "breakfast",
    "parking": "parking", "music": "music", "noise": "noise",
    "reservation": "reservation", "tea": "beverages", "coffee": "beverages",
    "movie": "movie", "film": "movie", "luxury vibe": "luxury vibe", "luxury": "luxury vibe"
}
STOP_TOPIC_TOKENS = {"is", "not", "got", "home", "in", "the", "a", "an", "and", "for", "of", "to", "it", "this", "that", "still", "please", "very", "first", "day", "mention", "special", "isnt", "wasnt", "ok"}

# ----------------------------
# Loaders
# ----------------------------
def load_review_model():
    global _review_tokenizer, _review_model
    if _review_model is not None and _review_tokenizer is not None:
        return
    print(f"DEBUG: Loading review-level model from: {REVIEW_MODEL_DIR}")
    _review_tokenizer = AutoTokenizer.from_pretrained(REVIEW_MODEL_DIR)
    _review_model = TFAutoModelForSequenceClassification.from_pretrained(REVIEW_MODEL_DIR)
    print("DEBUG: Review model loaded.")

def try_load_spacy():
    global _spacy_nlp
    try:
        import spacy
        if _spacy_nlp is None:
            _spacy_nlp = spacy.load("en_core_web_sm")
            print("DEBUG: spaCy loaded.")
    except Exception as e:
        _spacy_nlp = None
        print("DEBUG: spaCy not available:", e)

def try_load_langtool():
    global _lang_tool
    try:
        import language_tool_python
        if _lang_tool is None:
            _lang_tool = language_tool_python.LanguageTool("en-US")
            print("DEBUG: LanguageTool loaded.")
    except Exception as e:
        _lang_tool = None
        print("DEBUG: language_tool_python not available:", e)

# ----------------------------
# Cleaning
# ----------------------------
def clean_review_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    MANUAL_FIXES = {
        "warwck": "Warwick", "warwickk": "Warwick",
        "throught": "through", "throughts": "through",
        "al ": "all ", "al.": "all.", "tapp": "tap",
        "phonecalls": "phone calls", "phonecalls.": "phone calls.",
        "pick 20 minutes": "waited 20 minutes",
        "checking tap tap tap": "kept tapping", "tap tap tap": "tapping",
        "nice hotel not nice staff hotel lovely staff quite rude": "Nice hotel but not nice staff. Hotel lovely but staff quite rude",
        "waited pick": "waited to pick", "al friendly": "all friendly",
        "friendly faces tiring day airport": "friendly staff after a tiring day at the airport",
        "phone numbers booking confirmation email needs to be corrected": "Phone numbers in booking confirmation email need correction",
        "needs to be revamped": "needs revamping"
    }
    def apply_manual_fixes(s):
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
    t = apply_manual_fixes(t)

    if re.search(r'\d+\.', t):
        parts = re.split(r'(\d+\.\s*)', t)
        sentences = []
        for i, part in enumerate(parts):
            if re.match(r'\d+\.\s*', part):
                continue
            if part.strip():
                p = part.strip()
                if not p.endswith('.'):
                    p += '.'
                sentences.append(p.capitalize())
        t = ': '.join(sentences)
    else:
        sentences = []
        cur = []
        words = t.split()
        for i, w in enumerate(words):
            cur.append(w)
            if w.endswith(('.', '!', '?')) or (i < len(words)-1 and words[i+1].lower() in {'and','but','however','then','so','because','also','for','while','though'}):
                s = ' '.join(cur).strip(', ')
                if s:
                    sentences.append(s)
                cur = []
        if cur:
            s = ' '.join(cur).strip(', ')
            if s:
                sentences.append(s)
        t = '. '.join(s.capitalize() for s in sentences if s)

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

    def cap_sentences(s):
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
    t = cap_sentences(t)
    t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t, flags=re.I)

    try_load_langtool()
    if _lang_tool is not None:
        try:
            matches = _lang_tool.check(t)
            fixed = __import__("language_tool_python").utils.correct(t, matches)
            if fixed and isinstance(fixed, str):
                t = fixed
        except Exception:
            pass

    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not re.search(r"[.!?]$", t) and len(t.split()) > 6:
        t = t + "."
    return t

# ----------------------------
# Model scoring helpers
# ----------------------------
def softmax_probs_from_logits(logits: np.ndarray) -> np.ndarray:
    return tf.nn.softmax(logits, axis=-1).numpy()

def score_phrase_with_review_model(phrase: str) -> Tuple[str, float, Dict[str, float]]:
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

# ----------------------------
# Phrase normalization + lexicon helpers
# ----------------------------
def normalize_phrase_to_adj_noun(phrase: str, review_text: str = None) -> str:
    if not isinstance(phrase, str):
        return ""
    p = phrase.strip().lower()
    tokens = re.findall(r"\w+", p)
    if not tokens:
        return ""
    if len(tokens) > 6:
        return ""
    if tokens[0] in {"not","no","never"} and len(tokens) >= 3:
        adj = tokens[1].capitalize()
        noun_words = [w for w in tokens[2:] if w not in DETS_AND_CONJ]
        if not noun_words: return ""
        noun = ' '.join(noun_words).capitalize()
        return f"Not {adj} {noun}"
    if any(t in SMELL_TERMS for t in tokens):
        noun_candidate = None
        for i, tk in enumerate(tokens):
            if tk in SMELL_TERMS:
                if i-1 >= 0 and tokens[i-1].isalpha() and tokens[i-1] not in ADJ_TRUST:
                    noun_candidate = tokens[i-1]
                elif i+1 < len(tokens) and tokens[i+1].isalpha() and tokens[i+1] not in NOUN_BLACKLIST:
                    noun_candidate = tokens[i+1]
                break
        if noun_candidate:
            if "smelly" in tokens or any(t in SMELL_TERMS for t in tokens):
                return f"Smelly {noun_candidate.capitalize()}"
            return f"Bad {noun_candidate.capitalize()}"
        return "Bad Smell"
    if len(tokens) >= 2 and tokens[0] in ADJ_TRUST:
        noun_words = [w for w in tokens[1:] if w not in DETS_AND_CONJ]
        if not noun_words: return ""
        noun = ' '.join(noun_words).capitalize()
        return f"{tokens[0].capitalize()} {noun}"
    parts = [w for w in re.findall(r"[A-Za-z']+", phrase) if w.lower() not in PHRASE_REJECT_TOKENS]
    meaningful = [w for w in parts if w.lower() not in {"special","mention"}]
    if not meaningful: return ""
    if review_text and any(term in review_text.lower() for term in IMPLICIT_NEGATION_TERMS):
        noun = ' '.join(meaningful).capitalize()
        if "quality" in noun.lower() or "menu" in noun.lower():
            return f"Poor {noun}"
        elif "option" in noun.lower():
            return f"Limited {noun}"
        elif "correction" in review_text.lower():
            return f"Incorrect {noun}"
        return f"Poor {noun}"
    return " ".join([w.capitalize() for w in meaningful])

def lexicon_label(phrase: str) -> Optional[str]:
    tokens = set(re.findall(r"\w+", (phrase or "").lower()))
    if not tokens:
        return None
    has_neg = any(t in {"not","no","never"} for t in tokens)
    pos = tokens & LEXICON_POS
    neg = tokens & LEXICON_NEG
    if pos and not has_neg:
        return "Positive"
    if neg or (pos and has_neg):
        return "Negative"
    if any(term in phrase.lower() for term in IMPLICIT_NEGATION_TERMS):
        return "Negative"
    return None

def ensure_has_noun(phrase: str) -> bool:
    tokens = re.findall(r"\w+", (phrase or "").lower())
    if not tokens:
        return False
    return tokens[-1].isalpha() and tokens[-1] not in ADJ_TRUST and tokens[-1] not in NOUN_BLACKLIST

def validate_phrase_in_context(phrase: str, review_text: str, phrase_sentiment: str) -> bool:
    # Placeholder - you can add spaCy checks here
    return True

# ----------------------------
# Aggregation + deterministic fallback
# ----------------------------
def _priority_for_topic(neg_pct: float, total_mentions: int) -> str:
    if neg_pct >= 30.0 or total_mentions >= 10:
        return "High"
    if neg_pct >= 15.0 or total_mentions >= 4:
        return "Medium"
    return "Low"

def _deterministic_insight_for_topic(topic: str, total_m: int, neg_m: int, total_reviews: int) -> Dict:
    pct_all = (total_m / total_reviews * 100.0) if total_reviews else 0.0
    pct_neg_of_mentions = (neg_m / max(1, total_m) * 100.0) if total_m else 0.0
    actions = []
    t = topic.lower()
    if "billing" in t or "payment" in t:
        actions = [
            "Pilot tableside/QR/mobile payments and measure checkout time",
            "Train staff on expedited checkout and POS flows",
            "Monitor peak-hour queue times and reassign staff"
        ]
    elif "staff" in t or "service" in t or "desk" in t:
        actions = [
            "Run customer-service training and role-play scenarios",
            "Adjust staffing during heavy windows (hire part-timers)",
            "Add supervision/quality-checks and gather staff feedback"
        ]
    elif "restroom" in t or "stall" in t:
        actions = [
            "Increase cleaning rounds and add checklists with timestamps",
            "Schedule spot inspections during peak hours",
            "Add easy reporting (QR/hotline) for maintenance"
        ]
    elif "food" in t or "menu" in t or "breakfast" in t:
        actions = [
            "Run taste tests and review suppliers",
            "Add 3-5 vegetarian/desired options as a pilot",
            "Train kitchen staff on consistency and daily checks"
        ]
    elif "parking" in t:
        actions = [
            "List nearby parking options and add to booking emails",
            "Negotiate shared parking with nearby businesses",
            "Consider valet during peak times"
        ]
    elif "wifi" in t or "wi-fi" in t or "connect" in t or "connectivity" in t:
        actions = [
            "Audit network hardware and ISP performance; run speed tests",
            "Add redundancy or boost Wi-Fi capacity in key areas",
            "Provide clear signage for Wi-Fi and reset instructions"
        ]
    elif "elevator" in t or "lift" in t:
        actions = [
            "Schedule immediate maintenance for elevator reliability",
            "Add temporary signage and staff assistance when elevator is down",
            "Track downtime metrics and escalate to engineering"
        ]
    else:
        actions = [
            "Investigate root causes and collect more representative examples",
            "Open ops tickets and assign owners with due dates",
            "Track a small set of metrics to verify whether fixes improve customer sentiment"
        ]
    priority = _priority_for_topic(pct_neg_of_mentions, total_m)
    return {
        "issue": topic,
        "total_mentions": total_m,
        "negative_mentions": neg_m,
        "percent_of_reviews": round(pct_all, 1),
        "percent_negative_of_neg_reviews": round(pct_neg_of_mentions, 1),
        "priority": priority,
        "recommended_actions": actions,
        "source": "deterministic"
    }

def _canonical_topic_from_phrase(phrase: str) -> str:
    if not phrase: return ""
    s = phrase.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s: return ""
    for k, v in CANONICAL_MAP.items():
        if k in s:
            return v
    tokens = [t for t in s.split() if t not in STOP_TOPIC_TOKENS]
    if not tokens: return ""
    return tokens[-1]

def _build_aggregate_topic_stats(results: List[dict]) -> Dict[str, Dict]:
    total_reviews = len(results)
    raw_topics = {}
    for r in results:
        cleaned = r.get("cleaned_review") or r.get("review") or ""
        for kw in r.get("keywords", []):
            phrase = (kw.get("phrase") or "").strip()
            sentiment = kw.get("sentiment")
            topic = _canonical_topic_from_phrase(phrase)
            if not topic:
                continue
            entry = raw_topics.setdefault(topic, {"topic": topic, "total_mentions": 0, "negative_mentions": 0, "examples": []})
            entry["total_mentions"] += 1
            if sentiment == "Negative":
                entry["negative_mentions"] += 1
            snippet = cleaned.strip() or (r.get("review") or "").strip()
            if snippet and snippet not in entry["examples"]:
                entry["examples"].append(snippet if len(snippet) <= 300 else snippet[:300] + "...")
    # fallback scanning raw text
    if not raw_topics:
        for r in results:
            txt = (r.get("cleaned_review") or r.get("review") or "").lower()
            for k in CANONICAL_MAP.keys():
                if k in txt:
                    topic = CANONICAL_MAP[k]
                    entry = raw_topics.setdefault(topic, {"topic": topic, "total_mentions": 0, "negative_mentions": 0, "examples": []})
                    entry["total_mentions"] += 1
                    if r.get("sentiment") == "Negative":
                        entry["negative_mentions"] += 1
                    s = (r.get("cleaned_review") or r.get("review")).strip()
                    if s and s not in entry["examples"]:
                        entry["examples"].append(s)
    return raw_topics

# ----------------------------
# OpenAI robust helpers (full)
# ----------------------------
def _parse_json_from_text(text: str):
    if not text or not isinstance(text, str):
        return None
    # try direct json
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to find a JSON array substring
    start = text.find('[')
    if start != -1:
        end = text.rfind(']')
        if end != -1 and end > start:
            sub = text[start:end+1]
            try:
                return json.loads(sub)
            except Exception:
                try:
                    return ast.literal_eval(sub)
                except Exception:
                    pass
    # try to find a JSON object substring
    start = text.find('{')
    if start != -1:
        end = text.rfind('}')
        if end != -1 and end > start:
            sub = text[start:end+1]
            try:
                return json.loads(sub)
            except Exception:
                try:
                    return ast.literal_eval(sub)
                except Exception:
                    pass
    return None

def _extract_text_from_response(response):
    """
    Accepts multiple OpenAI response shapes and extracts textual content.
    """
    try:
        # Try ChatCompletion-like shape
        if hasattr(response, "choices") and isinstance(response.choices, list) and len(response.choices) > 0:
            choice = response.choices[0]
            # new client might have .message
            if hasattr(choice, "message") and isinstance(choice.message, dict):
                return choice.message.get("content") or str(choice.message)
            # older style
            if hasattr(choice, "text"):
                return choice.text
            # sometimes content is nested
            if hasattr(choice, "message") and hasattr(choice.message, "get"):
                return choice.message.get("content")
            # fallback str
            return str(choice)
        # direct dict style
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                ch = choices[0]
                if isinstance(ch, dict) and "message" in ch and isinstance(ch["message"], dict):
                    return ch["message"].get("content")
                if isinstance(ch, dict) and "text" in ch:
                    return ch.get("text")
        # fallback string
        return str(response)
    except Exception as e:
        print("DEBUG: _extract_text_from_response error:", e)
        return None

def generate_aggregated_insights_via_openai(topic_stats: Dict[str, Dict], total_reviews: int, max_topics: int = 8) -> Optional[List[Dict]]:
    """
    Patched OpenAI caller that is robust to truncated/non-JSON responses.
    Returns list of insight dicts or None on failure.
    """

    if not USE_OPENAI or openai_client is None:
        print("OpenAI not enabled or client missing; skipping OpenAI generation.")
        return None

    # Build items list to send
    items = []
    for t, data in list(topic_stats.items())[:max_topics]:
        items.append({
            "issue": data["topic"],
            "total_mentions": data.get("total_mentions", 0),
            "negative_mentions": data.get("negative_mentions", 0),
            "examples": data.get("examples", [])[:3]
        })

    few_shot = [
        {
            "issue": "billing",
            "percent_of_reviews": 12,
            "percent_negative_of_neg_reviews": 68,
            "priority": "High",
            "recommended_actions": [
                "Pilot mobile/QR/tableside payments and measure checkout time",
                "Train staff on expedited checkout and POS flows",
                "Add temporary staff during lunch hours"
            ],
            "short_explanation": "68% of negative reviews referenced slow billing; long wait times during lunch."
        },
        {
            "issue": "vegetarian options",
            "percent_of_reviews": 6,
            "percent_negative_of_neg_reviews": 23,
            "priority": "Medium",
            "recommended_actions": [
                "Add 3-5 vegetarian dishes as a pilot",
                "Highlight vegetarian options on menu",
                "Run a 'Meatless Monday' promotion"
            ],
            "short_explanation": "23% of negative feedback mentioned limited vegetarian choices."
        }
    ]

    prompt = (
        "You are an operations consultant. Given aggregated topic stats, return EXACTLY a JSON array of objects.\n"
        "Each object must have these fields: issue (string), percent_of_reviews (number), percent_negative_of_neg_reviews (number), "
        "priority (High/Medium/Low), recommended_actions (array of 3 concise action strings), short_explanation (1 sentence).\n"
        "Do NOT write any extra text. Only return the JSON array.\n\n"
        "Few-shot examples:\n" + json.dumps(few_shot, indent=2) + "\n\n"
        "Now analyze the provided topics and stats and produce the JSON. Topics input:\n" + json.dumps(items, indent=2) + "\n"
        f"Total reviews: {total_reviews}\n"
    )

    def _extract_text_and_meta(resp):
        """
        Tries several ways to extract text and finish_reason from the response
        Accepts dict-like and SDK object-like responses.
        Returns tuple (text:str or None, finish_reason:str or None)
        """
        try:
            # dict-like (classic openai.ChatCompletion.create)
            if isinstance(resp, dict):
                choices = resp.get("choices", [])
                if choices:
                    c0 = choices[0]
                    # message content
                    if isinstance(c0.get("message"), dict):
                        return c0["message"].get("content"), c0.get("finish_reason")
                    # text fallback
                    if "text" in c0:
                        return c0.get("text"), c0.get("finish_reason")
            # object-like (openai types)
            choices = getattr(resp, "choices", None)
            if choices and len(choices) > 0:
                first = choices[0]
                # message object
                msg = getattr(first, "message", None)
                if msg:
                    # message might be dict-like or object-like
                    if isinstance(msg, dict):
                        return msg.get("content"), getattr(first, "finish_reason", None)
                    # object-like
                    content = getattr(msg, "get", None)
                    if callable(content):
                        # msg behaves like mapping
                        try:
                            return msg.get("content"), getattr(first, "finish_reason", None)
                        except Exception:
                            pass
                    # try attribute
                    cont = getattr(msg, "content", None)
                    if cont is not None:
                        return cont, getattr(first, "finish_reason", None)
                # fallback to text attr
                txt = getattr(first, "text", None)
                if txt is not None:
                    return txt, getattr(first, "finish_reason", None)
        except Exception as e:
           print("Extraction helper failed: %s", e)
        # last resort str()
        try:
            return str(resp), None
        except Exception:
            return None, None

    def _call_openai(max_tokens_local=800, temperature_local=0.0):
        """
        Encapsulated call so we can control parameters.
        Uses new client shape if available but falls back to legacy dict-style.
        """
        try:
            # If using the new OpenAI client object where openai_client.chat.completions.create exists
            if hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
                return openai_client.chat.completions.create(
                    model=os.environ.get("OPENAI_MODEL", OPENAI_MODEL),
                    messages=[
                        {"role": "system", "content": "You are concise and return strict JSON arrays only."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens_local,
                    temperature=temperature_local,
                )
            # fallback: old openai package style (dict returned)
            import openai as old_openai  # type: ignore
            old_openai.api_key = OPENAI_API_KEY
            return old_openai.ChatCompletion.create(
                model=os.environ.get("OPENAI_MODEL", OPENAI_MODEL),
                messages=[
                    {"role": "system", "content": "You are concise and return strict JSON arrays only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens_local,
                temperature=temperature_local,
            )
        except Exception as e:
            print("OpenAI call failed: %s", e)
            return None

    # --- Attempts: initial, retry with larger max_tokens, then continuation ---
    attempts = [
        {"max_tokens": 800, "temperature": 0.0},
        {"max_tokens": 1400, "temperature": 0.0},
        {"max_tokens": 2000, "temperature": 0.0}
    ]

    for i, cfg in enumerate(attempts):
        print("Calling OpenAI model=%s max_tokens=%d attempt=%d", os.environ.get("OPENAI_MODEL", OPENAI_MODEL), cfg["max_tokens"], i+1)
        print(f"DEBUG: Calling OpenAI attempt {i+1} max_tokens={cfg['max_tokens']}")
        resp = _call_openai(max_tokens_local=cfg["max_tokens"], temperature_local=cfg["temperature"])
        if resp is None:
            print("OpenAI call returned None on attempt %d", i+1)
            continue

        text, finish_reason = _extract_text_and_meta(resp)
        print("raw OpenAI response type: %s", type(resp))
        print("finish_reason: %s", finish_reason)
        print("DEBUG: raw response type:", type(resp), "finish_reason:", finish_reason)
        if not text:
            print("OpenAI returned empty text on attempt %d", i+1)
            continue

        # try parse JSON
        parsed = _parse_json_from_text(text)
        if isinstance(parsed, list):
            out = []
            for o in parsed:
                if not isinstance(o, dict):
                    continue
                if "issue" in o and "recommended_actions" in o:
                    recs = o.get("recommended_actions") or []
                    if isinstance(recs, str):
                        recs = [s.strip() for s in re.split(r'[\n;]', recs) if s.strip()][:3]
                    recs = [str(x).strip() for x in recs][:3]
                    out.append({
                        "issue": str(o.get("issue")),
                        "percent_of_reviews": float(o.get("percent_of_reviews", 0)),
                        "percent_negative_of_neg_reviews": float(o.get("percent_negative_of_neg_reviews", 0)),
                        "priority": str(o.get("priority", "Low")),
                        "recommended_actions": recs,
                        "short_explanation": str(o.get("short_explanation", ""))[:300],
                    })
            if out:
                print("OpenAI returned %d insights on attempt %d.", len(out), i+1)
                return out

        # If not parseable, and the finish_reason indicates truncation, try to continue
        print("Attempt %d: response not parseable as JSON.", i+1)
        print("DEBUG: first response not parseable as JSON, finish_reason:", finish_reason)
        if finish_reason == "length" or (text and not text.strip().endswith(("]", "}"))):
            print("Response likely truncated (finish_reason=%s). Attempting continuation.", finish_reason)
            cont_prompt = (
                "The previous response was cut off. You returned a partial JSON array. "
                "Please continue and return ONLY the JSON array (no explanation). Previous partial output:\n\n"
                + (text[-4000:] if text else "")
            )
            try:
                if hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
                    cont_resp = openai_client.chat.completions.create(
                        model=os.environ.get("OPENAI_MODEL", OPENAI_MODEL),
                        messages=[
                            {"role": "system", "content": "You are concise and return strict JSON arrays only."},
                            {"role": "user", "content": cont_prompt}
                        ],
                        max_tokens=1200,
                        temperature=0.0
                    )
                else:
                    import openai as old_openai  # type: ignore
                    old_openai.api_key = OPENAI_API_KEY
                    cont_resp = old_openai.ChatCompletion.create(
                        model=os.environ.get("OPENAI_MODEL", OPENAI_MODEL),
                        messages=[
                            {"role": "system", "content": "You are concise and return strict JSON arrays only."},
                            {"role": "user", "content": cont_prompt}
                        ],
                        max_tokens=1200,
                        temperature=0.0
                    )
                cont_text, _ = _extract_text_and_meta(cont_resp)
                parsed = _parse_json_from_text((text or "") + (cont_text or ""))
                if isinstance(parsed, list):
                    out = []
                    for o in parsed:
                        if not isinstance(o, dict): continue
                        if "issue" in o and "recommended_actions" in o:
                            recs = o.get("recommended_actions") or []
                            if isinstance(recs, str):
                                recs = [s.strip() for s in re.split(r'[\n;]', recs) if s.strip()][:3]
                            recs = [str(x).strip() for x in recs][:3]
                            out.append({
                                "issue": str(o.get("issue")),
                                "percent_of_reviews": float(o.get("percent_of_reviews", 0)),
                                "percent_negative_of_neg_reviews": float(o.get("percent_negative_of_neg_reviews", 0)),
                                "priority": str(o.get("priority", "Low")),
                                "recommended_actions": recs,
                                "short_explanation": str(o.get("short_explanation", ""))[:300],

                            })
                    if out:
                        print("OpenAI continuation returned %d insights.", len(out))
                        return out
            except Exception as e:
                print("Continuation attempt failed: %s", e)

    # All attempts failed
    print("OpenAI aggregated insights failed after retries: falling back to deterministic.")
    return None

# ----------------------------
# MAIN view
# ----------------------------
@csrf_exempt
def predict_sentiment(request):
    """
    Batched predict_sentiment that accepts:
      - old format: list of plain review strings
      - new format: list of strings like "0~@~{...}" where the JSON contains latest_month_reviews array
    Outputs include bb_id for each review (or None).
    """
    try:
        # Ensure models/tools are loaded once per process
        load_review_model()
        try_load_spacy()
        try_load_langtool()

        if request.method != "POST":
            return JsonResponse({"error": "Use POST method"}, status=405)

        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"error": "Invalid JSON body"}, status=400)

        reviews_input = body.get("reviews")
        if not isinstance(reviews_input, list):
            return JsonResponse({"error": "Field 'reviews' (list) is required"}, status=400)

        # --------------------------
        # Parse input list into flattened reviews
        # Each flattened review: {"bb_id": str|None, "raw_review": str, "meta": dict}
        # Supports two forms:
        #  - plain string -> treat as raw_review
        #  - compound string "0~@~{...}" -> parse JSON, extract latest_month_reviews array
        # Also accepts if caller already passed a list of dicts (common alternative)
        # --------------------------
        flattened = []
        for item in reviews_input:
            # If user provided a dict directly (already parsed JSON)
            if isinstance(item, dict):
                # attempt to read either as a single review or container
                if "bb_id" in item and "msg" in item:
                    flattened.append({"bb_id": str(item.get("bb_id")), "raw_review": str(item.get("msg") or ""), "meta": item})
                    continue
                # if it's a container with latest_month_reviews
                if "latest_month_reviews" in item and isinstance(item["latest_month_reviews"], list):
                    for rev in item["latest_month_reviews"]:
                        bb = rev.get("bb_id")
                        msg = rev.get("msg") or ""
                        flattened.append({"bb_id": str(bb) if bb is not None else None, "raw_review": str(msg), "meta": rev})
                    continue
                # fallback: stringify
                flattened.append({"bb_id": None, "raw_review": json.dumps(item), "meta": item})
                continue

            # If the item is a string
            if isinstance(item, str):
                s = item.strip()
                # check for the "0~@~{...}" pattern (split on "~@~")
                if "~@~" in s:
                    try:
                        _, json_part = s.split("~@~", 1)
                        # robust: find first { and last } to extract JSON substring
                        start = json_part.find("{")
                        end = json_part.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            json_sub = json_part[start:end+1]
                        else:
                            json_sub = json_part
                        parsed = json.loads(json_sub)
                        # If parsed contains latest_month_reviews array
                        lm = parsed.get("latest_month_reviews") if isinstance(parsed, dict) else None
                        if isinstance(lm, list):
                            for rev in lm:
                                bb = rev.get("bb_id")
                                msg = rev.get("msg") or ""
                                flattened.append({"bb_id": str(bb) if bb is not None else None, "raw_review": str(msg), "meta": rev})
                            continue
                        # else maybe parsed is itself a list or single review
                        if isinstance(parsed, list):
                            for rev in parsed:
                                if isinstance(rev, dict):
                                    bb = rev.get("bb_id")
                                    msg = rev.get("msg") or ""
                                    flattened.append({"bb_id": str(bb) if bb is not None else None, "raw_review": str(msg), "meta": rev})
                                else:
                                    flattened.append({"bb_id": None, "raw_review": str(rev), "meta": rev})
                            continue
                        # fallback: use parsed as single dict
                        if isinstance(parsed, dict):
                            bb = parsed.get("bb_id")
                            msg = parsed.get("msg") or ""
                            flattened.append({"bb_id": str(bb) if bb is not None else None, "raw_review": str(msg), "meta": parsed})
                            continue
                    except Exception as e:
                        # parsing failed; fall through to treat the whole string as raw text
                        print("DEBUG: JSON parse from ~@~ string failed:", e)
                        flattened.append({"bb_id": None, "raw_review": s, "meta": None})
                        continue
                # If not the compound format, maybe the string itself is JSON (try parse)
                try:
                    maybe = json.loads(s)
                    if isinstance(maybe, dict) and "latest_month_reviews" in maybe and isinstance(maybe["latest_month_reviews"], list):
                        for rev in maybe["latest_month_reviews"]:
                            bb = rev.get("bb_id")
                            msg = rev.get("msg") or ""
                            flattened.append({"bb_id": str(bb) if bb is not None else None, "raw_review": str(msg), "meta": rev})
                        continue
                    # if it's a single review dict
                    if isinstance(maybe, dict) and "msg" in maybe:
                        bb = maybe.get("bb_id")
                        msg = maybe.get("msg") or ""
                        flattened.append({"bb_id": str(bb) if bb is not None else None, "raw_review": str(msg), "meta": maybe})
                        continue
                    # array of reviews
                    if isinstance(maybe, list):
                        for rev in maybe:
                            if isinstance(rev, dict):
                                bb = rev.get("bb_id")
                                msg = rev.get("msg") or ""
                                flattened.append({"bb_id": str(bb) if bb is not None else None, "raw_review": str(msg), "meta": rev})
                            else:
                                flattened.append({"bb_id": None, "raw_review": str(rev), "meta": rev})
                        continue
                except Exception:
                    # not JSON, keep as raw string
                    flattened.append({"bb_id": None, "raw_review": s, "meta": None})
                    continue
            else:
                # unknown type - stringify
                flattened.append({"bb_id": None, "raw_review": str(item), "meta": item})

        # After parsing, flattened is a list of review dicts
        if not flattened:
            return JsonResponse({"predictions": [], "insights": [], "ai_insights_text": ""}, status=200)

        # --------------------------
        # Build arrays for batching
        # --------------------------
        raw_texts = [r["raw_review"] for r in flattened]
        bb_ids = [r["bb_id"] for r in flattened]
        raw_metas = [r["meta"] for r in flattened]
        n_reviews = len(raw_texts)

        # --------------------------
        # Clean all reviews first (batch)
        # --------------------------
        cleaned_texts = [clean_review_text(t) for t in raw_texts]

        # --------------------------
        # Batch review-level sentiment inference
        # --------------------------
        try:
            enc_reviews = _review_tokenizer(cleaned_texts, return_tensors="tf", truncation=True, padding=True, max_length=128)
            out_reviews = _review_model(enc_reviews)
            probs_all = softmax_probs_from_logits(out_reviews.logits)  # shape (N, num_labels)
        except Exception as e:
            # If batching fails, try fallback to single queries (robustness)
            print("DEBUG: Batched review-level inference failed:", e)
            probs_all_list = []
            for t in cleaned_texts:
                try:
                    enc = _review_tokenizer(t, return_tensors="tf", truncation=True, padding="max_length", max_length=128)
                    o = _review_model(enc)
                    probs = softmax_probs_from_logits(o.logits)[0]
                except Exception as ee:
                    print("DEBUG: Single review inference failed:", ee)
                    probs = np.array([0.33, 0.34, 0.33])
                probs_all_list.append(probs)
            probs_all = np.vstack(probs_all_list)

        # Prepare basic review-level meta
        review_meta = []
        for i in range(n_reviews):
            probs = probs_all[i]
            pred_idx = int(np.argmax(probs))
            review_sentiment = REVIEW_LABEL_MAP.get(pred_idx, "Neutral")
            review_conf = float(probs[pred_idx])
            prob_map = {str(j): float(probs[j]) for j in range(probs.shape[0])}
            review_meta.append({
                "sentiment": review_sentiment,
                "confidence": review_conf,
                "prob_map": prob_map
            })

        # --------------------------
        # Collect phrase candidates (model-driven preferred)
        # --------------------------
        PHRASE_CANDIDATES_PER_REVIEW = 3
        PHRASE_SCORE_BATCH = 64

        candidates_per_review = [[] for _ in range(n_reviews)]
        all_candidates = []   # list of tuples (rev_idx, phrase)
        for i, cleaned in enumerate(cleaned_texts):
            raw_candidates = []
            if extract_aspect_adjs is not None:
                try:
                    raw_candidates = extract_aspect_adjs(cleaned, rel_threshold=0.45) or []
                except Exception as e:
                    print("DEBUG: extract_aspect_adjs failed:", e)
                    raw_candidates = []
            seen_local = set()
            for p in raw_candidates:
                if not p or not isinstance(p, str):
                    continue
                norm = re.sub(r"\s+", " ", p.strip()).strip(" .,:;!?")
                if len(re.findall(r"\w+", norm)) < 2:
                    continue
                key = norm.lower()
                if key in seen_local:
                    continue
                seen_local.add(key)
                candidates_per_review[i].append(norm)
                all_candidates.append((i, norm))
                if len(candidates_per_review[i]) >= PHRASE_CANDIDATES_PER_REVIEW:
                    break

        # fallback to first-sentence preview when no phrase candidates found for a review
        for i, arr in enumerate(candidates_per_review):
            if not arr:
                preview = cleaned_texts[i].strip().split(".")[0].strip().title() if cleaned_texts[i] else ""
                if preview:
                    candidates_per_review[i].append(preview)
                    all_candidates.append((i, preview))

        # --------------------------
        # Deduplicate unique phrases and batch-score them
        # --------------------------
        uniq_map = {}
        unique_phrases = []
        for _, phrase in all_candidates:
            key = phrase.lower()
            if key not in uniq_map:
                uniq_map[key] = len(unique_phrases)
                unique_phrases.append(phrase)

        phrase_results = [None] * len(unique_phrases)
        if unique_phrases:
            for start in range(0, len(unique_phrases), PHRASE_SCORE_BATCH):
                chunk = unique_phrases[start:start+PHRASE_SCORE_BATCH]
                try:
                    encp = _review_tokenizer(chunk, return_tensors="tf", truncation=True, padding=True, max_length=64)
                    outp = _review_model(encp)
                    probs_chunk = softmax_probs_from_logits(outp.logits)
                    for j, probs in enumerate(probs_chunk):
                        pred_idx = int(np.argmax(probs))
                        phrase_results[start + j] = {
                            "label": REVIEW_LABEL_MAP.get(pred_idx, "Neutral"),
                            "confidence": float(probs[pred_idx])
                        }
                except Exception as e:
                    print("DEBUG: Batched phrase scoring failed, falling back to per-phrase. Error:", e)
                    for j, phrase in enumerate(chunk):
                        try:
                            encp = _review_tokenizer(phrase, return_tensors="tf", truncation=True, padding="max_length", max_length=64)
                            outp = _review_model(encp)
                            probs = softmax_probs_from_logits(outp.logits)[0]
                            pred_idx = int(np.argmax(probs))
                            phrase_results[start + j] = {
                                "label": REVIEW_LABEL_MAP.get(pred_idx, "Neutral"),
                                "confidence": float(probs[pred_idx])
                            }
                        except Exception as ee:
                            print("DEBUG: Single phrase scoring failed:", ee)
                            phrase_results[start + j] = {"label": "Neutral", "confidence": 0.33}

        # Map scored phrases back to each review, apply thresholds and lexicon fallback
        results_keywords = [[] for _ in range(n_reviews)]
        for rev_idx, phrase in all_candidates:
            key = phrase.lower()
            if key not in uniq_map:
                continue
            idx = uniq_map[key]
            pr = phrase_results[idx]
            if pr is None:
                continue
            conf = pr.get("confidence", 0.0)
            lbl = pr.get("label", "Neutral")
            if conf >= PHRASE_CONF_THRESHOLD or (lbl in {"Negative", "Positive"} and conf >= 0.35):
                results_keywords[rev_idx].append({"phrase": phrase, "sentiment": lbl, "confidence": float(conf)})
            else:
                lex = lexicon_label(phrase)
                if lex:
                    results_keywords[rev_idx].append({"phrase": phrase, "sentiment": lex, "confidence": max(0.35, float(conf))})

        # De-duplicate per review and pick top K (max 8)
        final_keywords = []
        for kw_list in results_keywords:
            dedup = {}
            for k in kw_list:
                key = k["phrase"].lower()
                if key not in dedup or k["confidence"] > dedup[key]["confidence"]:
                    dedup[key] = k
            sorted_k = sorted(dedup.values(), key=lambda x: x["confidence"], reverse=True)[:8]
            final_keywords.append(sorted_k)

        # Assemble per-review results (include bb_id)
        results = []
        for i in range(n_reviews):
            rm = review_meta[i]
            keywords = final_keywords[i]
            if not keywords:
                preview = cleaned_texts[i].strip().split(".")[0].title() if cleaned_texts[i] else ""
                if preview:
                    keywords = [{"phrase": preview, "sentiment": rm["sentiment"], "confidence": rm["confidence"]}]
            results.append({
                "bb_id": bb_ids[i],
                "review": raw_texts[i],
                "cleaned_review": cleaned_texts[i],
                "sentiment": rm["sentiment"],
                "confidence": float(rm["confidence"]),
                "probabilities": rm["prob_map"],
                "keywords": keywords,
                "meta": raw_metas[i],
                "debug": {"raw": [p for p in (candidates_per_review[i] or [])], "kept": [k["phrase"] for k in keywords], "dropped": []}
            })

        # Build aggregated insights + formatted text (OpenAI then deterministic)
        insights, ai_insights_text, openai_used_flag = [], "", False
        raw_topics = _build_aggregate_topic_stats(results)
        ai_generated = None
        if USE_OPENAI:
            try:
                ai_generated = generate_aggregated_insights_via_openai(
                    {d["topic"]: d for d in sorted(raw_topics.values(), key=lambda x: (x.get("negative_mentions", 0), x.get("total_mentions", 0)), reverse=True)},
                    len(results),
                    max_topics=12
                )
            except Exception as e:
                print("DEBUG: generate_aggregated_insights_via_openai exception:", e)
                ai_generated = None

        if ai_generated:
            openai_used_flag = True
            insights = []
            for g in ai_generated:
                insights.append({
                    "issue": g.get("issue"),
                    "summary": f"{g.get('percent_of_reviews',0):.0f}% of reviews mention {g.get('issue')}.",
                    "priority": g.get("priority"),
                    "recommended_actions": g.get("recommended_actions"),
                    "short_explanation": g.get("short_explanation", ""),
                    "source": g.get("source", "openai")
                })
            lines = []
            for ins in insights:
                lines.append(f"{ins['issue'].title()}")
                if ins.get("summary"):
                    lines.append(ins["summary"])
                lines.append(f"{ins.get('priority','Low')} priority")
                lines.append("Recommended Actions:")
                for a in ins.get("recommended_actions", []):
                    lines.append(f"- {a}")
                if ins.get("source"):
                    lines.append(f"(source: {ins.get('source')})")
                lines.append("")
            ai_insights_text = "\n".join(lines).strip()
        else:
            ordered = sorted(raw_topics.values(), key=lambda x: (x.get("negative_mentions", 0), x.get("total_mentions", 0)), reverse=True)
            insights = []
            for d in ordered[:12]:
                entry = _deterministic_insight_for_topic(d["topic"], d.get("total_mentions", 0), d.get("negative_mentions", 0), len(results))
                insights.append({
                    "issue": entry["issue"],
                    "summary": f"{entry['percent_of_reviews']:.0f}% of reviews mention {entry['issue']}.",
                    "priority": entry["priority"],
                    "recommended_actions": entry["recommended_actions"],
                    "short_explanation": "",
                    "source": entry.get("source", "deterministic")
                })
            lines = []
            for ins in insights:
                lines.append(f"{ins['issue'].title()}")
                if ins.get("summary"): lines.append(ins['summary'])
                lines.append(f"{ins.get('priority','Low')} priority")
                lines.append("Recommended Actions:")
                for a in ins.get("recommended_actions", []):
                    lines.append(f"- {a}")
                if ins.get("source"): lines.append(f"(source: {ins.get('source')})")
                lines.append("")
            ai_insights_text = "\n".join(lines).strip()

        resp_payload = {"predictions": results, "insights": insights, "ai_insights_text": ai_insights_text, "openai_used": bool(openai_used_flag)}
        return JsonResponse(resp_payload, status=200)

    except Exception as e:
        print("DEBUG: Prediction failed:", e)
        return JsonResponse({"error": f"Prediction failed: {str(e)}"}, status=500)

