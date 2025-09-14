from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"  # 6 emotion classes

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
_device = 0 if torch.cuda.is_available() else -1

_pipe = pipeline("text-classification", model=_model, tokenizer=_tokenizer, device=_device, return_all_scores=True)

def classify_emotion(text: str):
    """
    Returns: (top_emotion, confidence, scores_dict)
    """
    results = _pipe(text, truncation=True)
    scores = {r["label"]: r["score"] for r in results[0]}
    top = max(scores, key=scores.get)
    return top, scores[top], scores
