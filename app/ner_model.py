import pickle

import spacy

MODEL_PATH = "ner_model.pkl"

def load_model():
    """Load the pre-trained SpaCy NER model."""
    return spacy.load("en_core_web_sm")

def save_model(nlp):
    """Save the trained NER model."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(nlp, f)

def load_saved_model():
    """Load the saved NER model."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
