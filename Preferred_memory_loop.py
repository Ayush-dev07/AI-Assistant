import sqlite3
import math

DB = "/home/ayush02/zero_two_intents.db"

def increment_usage(intent, value):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
                INSERT INTO preference_usage(intent, value, count)
                VALUES (?, ?, 1)
                ON CONFLICT(intent, value)
                DO UPDATE SET count = count + 1
                """, (intent, str(value)))
    conn.commit()
    conn.close()

def get_usage(intent, value):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
                SELECT count FROM preference_usage
                WHERE intent=? AND value=?
                """, (intent, str(value)))
    row= cur.fetchone()
    conn.close()

    return row[0] if row else 0

def get_preference(intent):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
                SELECT value, count FROM preference_usage
                WHERE intent=?
                ORDER BY count DESC
                LIMIT 1
                """, (intent,))
    
    row = cur.fetchone()
    conn.close()

    if not row:
        return None, 0
    
    return row[0], row[1]

def update_intent_entropy(intent):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    
    cur.execute("""
                SELECT count FROM preference_usage
                WHERE intent=?
                """, (intent,))
    
    counts = [r[0] for r in cur.fetchall()]
    conn.close()
    if not counts:
        return
    total = sum(counts)
    entropy = 0

    for c in counts:
        p = c/total
        entropy -= p*math.log(p+1e-9)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
                INSERT INTO intent_preferred (intent, total, entropy)
                VALUES (?, ?, ?)
                ON CONFLICT(intent)
                DO UPDATE SET total=?, entropy=?
                """, (intent, total, entropy, total, entropy))
    conn.commit()
    conn.close()

def dominance_ratio_for_intent(intent):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("SELECT entropy FROM intent_preferred WHERE intent=?", (intent,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return 1.5 
    entropy = row[0]

    return 1.1 + entropy

def get_dominance_preference(intent):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
                SELECT value, count FROM preference_usage
                WHERE intent=?
                ORDER BY count DESC
                LIMIT 2 
                """, (intent,))
    rows = cur.fetchall()
    conn.close()

    if len(rows) == 0:
        return None
    
    if len(rows) == 1:
        return rows[0][0]
    (v1, c1), (_, c2) = rows
    ratio = dominance_ratio_for_intent(intent)
    
    if c1>= 50 and c1>= c2*ratio:
        return v1
    return None
    