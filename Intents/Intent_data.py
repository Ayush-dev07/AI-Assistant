import sqlite3

conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
cursor = conn.cursor()

samples = [
    ("decrease volume", "set_volume"),
    ("reduce volume", "set_volume"),
    ("", "set_volume"),
    ("hand control zero two", "set_volume"),
    ("start gesture", "set_volume"),
    ("actions command zero two", "set_volume"),
    ("gesture", "set_volume"),
    
]

cursor.executemany(
    "INSERT INTO intent_samples (text, intent) VALUES(?,?)",
    samples
)

conn.commit()
conn.close()

print("Intent samples inserted.")