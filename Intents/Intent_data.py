import sqlite3

conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
cursor = conn.cursor()

samples = [
    ("ok stop", "sleep_words"),
    ("sleep", "sleep_words"),
    ("that's enough for today", "sleep_words"),
    ("zero two goodnight", "sleep_words"),
    ("rest zero two", "sleep_words"),
    ("stop zero two", "sleep_words"),
    ("exit", "sleep_words"),
    
]

cursor.executemany(
    "INSERT INTO intent_samples (text, intent) VALUES(?,?)",
    samples
)

conn.commit()
conn.close()

print("Intent samples inserted.")