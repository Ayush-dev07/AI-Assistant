import sqlite3
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = None
model = None

def train_model():
    global vectorizer, model
    conn = sqlite3.connect("/home/ayush02/zero_two_intents.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT text, intent FROM intent_samples")
    rows = cursor.fetchall()
    conn.close()

    texts = [row[0] for row in rows]
    lables = [row[1] for row in rows]

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        max_iter= 1000,
        class_weight= "balanced"
    )

    model.fit(X, lables)

    joblib.dump(model, "intent_model.pkl")
    joblib.dump(vectorizer, "intent_vectorizer.pkl")

    print("Intent model trained using SQlite.")
    return model, vectorizer
