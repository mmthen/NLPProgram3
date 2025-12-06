import os
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""
Dataset cleaning
Removes punctuation and alphanumeric characters
and strips/collapses white spaces
"""


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


"""
Load Data
"""


def load_data(path):
    sents = []
    labels = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            label, sent = line.split("\t", 1)
            labels.append(int(label))
            sents.append(sent)
    return sents, labels


"""
Train Model
"""


def train_model(word):
    print(f'\nTraining model for {word}')

    dataPath = f"data/{word}.txt"

    sents, labels = load_data(dataPath)

    clean = [clean_text(s) for s in sents]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words=None)

    X = vectorizer.fit_transform(clean)

    # Training split
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.20, random_state=42,
        stratify=labels)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Get some accuracy
    val_acc = clf.score(X_val, y_val)
    print(f'Validation accuracy: {val_acc:.4f}')

    clf.fit(X, labels)

    os.makedirs("saved_models", exist_ok=True)

    """
    Saving the trained models and learned vectorizer parameters 
    so we dont have retrain later
    """
    joblib.dump(vectorizer, f"saved_models/{word}_vectorizer.pkl")
    joblib.dump(clf, f"saved_models/{word}_model.pkl")

    print(f"Saved models and vectorizer for {word}")


if __name__ == "__main__":
    for w in ["director", "overtime", "rubbish"]:
        train_model(w)





