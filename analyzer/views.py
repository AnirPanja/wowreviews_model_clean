import os
import json
import numpy as np
import tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    TFAutoModelForTokenClassification,
)

# ----------------------------
# Load review-level sentiment model
# ----------------------------
review_model_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../wowreviews_model/reports_model/tripadvisor_bert_model_chunked/final",
    )
)
print(f"Loading review-level model from {review_model_dir}...")
review_tokenizer = AutoTokenizer.from_pretrained(review_model_dir)
review_model = TFAutoModelForSequenceClassification.from_pretrained(review_model_dir)

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ----------------------------
# Load phrase-level sentiment model
# ----------------------------
phrase_model_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../wowreviews_model/reports_model/phrase_sentiment_model",
    )
)
print(f"Loading phrase-level model from {phrase_model_dir}...")
phrase_tokenizer = AutoTokenizer.from_pretrained(phrase_model_dir)
phrase_model = TFAutoModelForTokenClassification.from_pretrained(
    phrase_model_dir, from_pt=False  # ensure TensorFlow weights are loaded
)

id2label = phrase_model.config.id2label


# ----------------------------
# Helper: Get phrase-level sentiments
# ----------------------------
def get_phrase_sentiments(review: str):
    """Run token classification model for phrase-level sentiments."""
    inputs = phrase_tokenizer(
        review,
        return_tensors="tf",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    outputs = phrase_model(inputs)
    predictions = tf.argmax(outputs.logits, axis=-1).numpy()[0]
    tokens = phrase_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    phrases_with_sentiments = []
    current_phrase, current_sent = [], None

    for token, label_id in zip(tokens, predictions):
        label = id2label[label_id]

        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if label == "O":
            if current_phrase:
                phrases_with_sentiments.append(
                    (" ".join(current_phrase), current_sent)
                )
                current_phrase, current_sent = [], None
            continue

        if label.startswith("B-"):
            if current_phrase:
                phrases_with_sentiments.append(
                    (" ".join(current_phrase), current_sent)
                )
            current_phrase = [token]
            current_sent = label.split("-")[1].capitalize()

        elif label.startswith("I-") and current_phrase:
            current_phrase.append(token)

    # Catch last phrase
    if current_phrase:
        phrases_with_sentiments.append((" ".join(current_phrase), current_sent))

    # Clean wordpieces
    clean_phrases = []
    for phrase, sentiment in phrases_with_sentiments:
        phrase = phrase.replace("##", "")
        clean_phrases.append((phrase.strip(), sentiment))

    return clean_phrases


# ----------------------------
# Django API endpoint
# ----------------------------
@csrf_exempt
def predict_sentiment(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            reviews = data.get("reviews", [])
            if not reviews or not isinstance(reviews, list):
                return JsonResponse(
                    {"error": "Reviews field is required and must be a list of strings"},
                    status=400,
                )

            results = []
            for review in reviews:
                if not isinstance(review, str):
                    return JsonResponse(
                        {"error": "All reviews must be strings"}, status=400
                    )

                # ---- Review-level sentiment ----
                inputs = review_tokenizer(
                    review,
                    return_tensors="tf",
                    padding="max_length",
                    truncation=True,
                    max_length=64,
                )
                outputs = review_model(inputs)
                probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()
                predicted_class = np.argmax(probabilities, axis=1)[0]
                confidence = probabilities[0][predicted_class]
                sentiment = sentiment_map[predicted_class]

                # ---- Phrase-level sentiment ----
                phrase_sentiments = get_phrase_sentiments(review)

                results.append(
                    {
                        "review": review,
                        "sentiment": sentiment,
                        "confidence": float(confidence),
                        "probabilities": {
                            k: float(v) for k, v in enumerate(probabilities[0])
                        },
                        "keywords": [
                            {"phrase": p, "sentiment": s}
                            for p, s in phrase_sentiments
                        ],
                    }
                )

            return JsonResponse({"predictions": results}, status=200)

        except Exception as e:
            return JsonResponse(
                {"error": f"Prediction failed: {str(e)}"}, status=500
            )

    return JsonResponse({"error": "Use POST method"}, status=405)
