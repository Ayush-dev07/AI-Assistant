import sqlite3
from Preferred_memory_loop import increment_usage

PREFERENCE_THRESHOLD = 50
DB = "/home/ayush02/zero_two_intents.db"

def set_memory(key, value):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "REPLACE INTO user_memory (key, value) VALUES (?, ?)",
        (key, str(value))
    )
    conn.commit()
    conn.close()

def get_memory(key):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT value FROM user_memory WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def learn(intent, entities):
    if intent == "open_app" and "app" in entities:
        app = entities["app"]
        increment_usage("open_app", app)

    if intent == "set_brightness" and "value" in entities:
        increment_usage("set_brightness", entities["value"])

