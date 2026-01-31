import joblib
import os
from Zero_two import listen_once, execute_command
from Speaker import speak
import sqlite3
from Intents.Intent_trainer import train_model
COMMAND_COUNTER = 0
PROMOTE_EVERY = 10

MODEL_PATH = "intent_model.pkl"
VECTORIZER_PATH = "intent_vectorizer.pkl"

def load_or_train_model():
    global model, vectorizer

    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model loaded successfully.")
    else:
        print("No trained model found. Training now...")
        model, vectorizer = train_model()

load_or_train_model()

def predict_intent(command: str):
    command = command.lower().strip()
    X = vectorizer.transform([command])
    probs = model.predict_proba(X)[0]

    intent = model.classes_[probs.argmax()]
    conf = probs.max()

    return intent, conf

def save_feedback(text, intent, confirmed):
    global COMMAND_COUNTER

    conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO feedback (text, intent, confirmed) VALUES (?, ?, ?)",
        (text.lower(), intent, confirmed)
    )
    conn.commit()
    conn.close()

    if confirmed == 1:
        COMMAND_COUNTER += 1
        if COMMAND_COUNTER % PROMOTE_EVERY ==0:
            promote_and_retrain()

def promote_and_retrain():
    print("Updating training data...")

    conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
    cur = conn.cursor()

    cur.execute("SELECT text, intent FROM feedback WHERE confirmed=1")
    rows = cur.fetchall()

    if rows:
        cur.executemany(
            "INSERT INTO intent_samples (text, intent) VALUES (?, ?)",
            rows
        )
        cur.execute("DELETE FROM feedback WHERE confirmed=1")

    conn.commit()
    conn.close()

    train_model()

def handled_command(text):
    intent, conf = predict_intent(text)

    conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
    cur = conn.cursor()
    cur.execute("SELECT threshold FROM intent_stats WHERE intent=?", (intent,))
    row = cur.fetchone()

    if row is None:
        ensure_intent_exists(intent)
        threshold = 0.20
    else:
        threshold = row[0]
    
    conn.commit()
    conn.close()
    
    if conf >= threshold:
        execute_command(intent)
        save_feedback(text, intent, 1)
        update_intent_stats(intent, True)
        return

    speak(f"Did you mean {intent.replace('_',' ')}?")
    answer = listen_once()

    if "yes" in answer.lower():
        execute_command(intent)
        save_feedback(text, intent, 1)
        update_intent_stats(intent, True)
    else:
        speak("What should i do?")
        correct = listen_once()
        save_feedback(text, correct, 1)
        update_intent_stats(correct, True)
        update_intent_stats(intent, False)

def update_intent_stats(intent, correct):
    conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
    cur = conn.cursor()

    cur.execute("""
                UPDATE intent_stats
                SET seen = seen + 1, correct = correct + ?
                WHERE intent = ?
                """, (1 if correct else 0, intent))
    conn.commit()
    conn.close()

    recalc_intent_threshold(intent)

def recalc_intent_threshold(intent):
    conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
    cur = conn.cursor()

    cur.execute("SELECT seen, correct FROM intent_stats WHERE intent=?", (intent,))
    row = cur.fetchone()

    if row is None:
        seen = 0
        correct = 0
    else:
        seen, correct = row

    if seen < 10:
        new_t = 0.25
    else:
        acc = correct/seen
        new_t = min(0.85, 0.25 + acc * 0.5)

    cur.execute("UPDATE intent_stats SET threshold=? WHERE intent=?", (new_t, intent))
    conn.commit()
    conn.close()

def should_execute(intent, conf):
    conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
    cur = conn.cursor()

    cur.execute("SELECT threshold FROM intent_stats WHERE intent=?", (intent,))
    t = cur.fetchone()[0]
    conn.close()
    
    return conf >= t

def ensure_intent_exists(intent):
    conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
    cur = conn.cursor()

    cur.execute(
        "INSERT OR IGNORE INTO intent_stats (intent, seen, correct, threshold) VALUES (?, 0, 0, 0.25)",
        (intent,)
    )
    conn.commit()
    conn.close()
