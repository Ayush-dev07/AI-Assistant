from App_registry import get_app_name, extract_number, detect_unit
from Memory import get_memory
from transformers import pipeline
from auto_app_detection import scan_desktop_files
import os
import torch
torch.set_num_threads(2)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import logging
logging.set_verbosity_error()


APPS = scan_desktop_files()

ner = None 
def get_ner():
    global ner
    if ner is None:
        ner = pipeline(
            "ner",
            model="elastic/distilbert-base-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
            )
        return ner

def extract_entities_transformer(text):
    if text is None or not str(text).strip():
        return {}
    
    ner = get_ner()
    results = ner(text)
    entities = {}

    for r in results:
        if r["entity_group"] == "MISC":
            entities.setdefault("app", r["word"].lower())
        if r["entity_group"] == "ORG":
            entities.setdefault("app", r["wprd"].lower())
        if r["entity_group"] == "QUANTITY":
            entities["value"] = int("".join(filter(str.isdigit, r["word"])))

    return entities

def normalize_app_name(raw):
    raw = raw.lower()
    for canonical, meta in APPS.items():
        if raw == canonical or raw in meta["aliases"]:
            return canonical
    return None

def extract_entities(intent, command):
    entities = {}

    if intent in ["open_app", "close_app"]:
        app = get_app_name(command)
        if app:
            entities["app"] = app
    if intent in ["set_volume", "set_brightness"]:
        value = extract_number(command)
        unit = detect_unit(command)

        if value is not None:
            entities["value"] = value
        if unit:
            entities["unit"] = unit

    if intent == "open_app" and "app" not in entities:
        pref = get_memory("preferred_browser")
        if pref:
            entities["app"] = pref

    if intent == "set_brightness" and "value" not in entities:
        pref = get_memory("preferred_brightness")
        if pref:
            entities["value"] = int(pref)

    return entities
