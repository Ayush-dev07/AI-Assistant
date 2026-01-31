INTENT_LABLES = [
    "open_app",
    "close_app",
    "set_volume",
    "set_brightness",
    "shutdown_system",
    "unknown"
]

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels = len(INTENT_LABLES)
)

def predict_intent_transformer(text):
    if text is None or not str(text).strip():
        return "Unknown", 0.0
                               
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    conf, idx = torch.max(probs, dim=1)

    return INTENT_LABLES[idx.item()], conf.item()