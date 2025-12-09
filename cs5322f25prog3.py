
import re
from pathlib import Path

import joblib

#File paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "saved_models"

#Model cached
VECTOR_CACHE = {}
MODEL_CACHE = {}


#Preprocessing matching the training
def preprocess(text, word=None):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_model(word):

    if word not in VECTOR_CACHE or word not in MODEL_CACHE:
        vec_path = MODELS_DIR / f"{word}_vectorizer.pkl"
        clf_path = MODELS_DIR / f"{word}_model.pkl"

        if not vec_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found for '{word}': {vec_path}")
        if not clf_path.exists():
            raise FileNotFoundError(f"Model file not found for '{word}': {clf_path}")

        VECTOR_CACHE[word] = joblib.load(vec_path)
        MODEL_CACHE[word] = joblib.load(clf_path)

    return VECTOR_CACHE[word], MODEL_CACHE[word]

def _predict_for_word(sent_list, word):
    
    vectorizer, clf = load_model(word)

    #Preprocess input sentences
    processed = [preprocess(s, word) for s in sent_list]

    #Vectorize and predict 
    X = vectorizer.transform(processed)
    preds = clf.predict(X)

    #Return list of python integers
    return [int(p) for p in preds]


def WSD_Test_director(list):
    print("Predicting for 'director'")
    return _predict_for_word(list, "director")

def WSD_Test_rubbish(list):
    print("Predicting for 'rubbish'")
    return _predict_for_word(list, "rubbish")

def WSD_Test_overtime(list):
    print("Predicting for 'overtime'") 
    return _predict_for_word(list, "overtime")


test_director = [
    "The managing director approved the changes.",     # sense 1
    "The film director started shooting the scene."     # sense 2
]

test_overtime = [
    "She worked three hours of overtime last night.",    # sense 1
    "The game went into overtime after a late goal."     # sense 2
]

test_rubbish = [
    "He threw the broken plate into the rubbish bin.",   # sense 1
    "That's absolute rubbish, and you know it."          # sense 2
]

print(WSD_Test_director(test_director))
print(WSD_Test_overtime(test_overtime))
print(WSD_Test_rubbish(test_rubbish))