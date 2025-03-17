import os

import spacy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "app/ner_model_best")


def load_model():
    """Load the trained NER model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Check your path!")

    print(f"✅ Loading NER model from {MODEL_PATH}...")
    return spacy.load(MODEL_PATH)


def predict_entities(nlp, text):
    """Predict named entities from the input text."""
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_,
        }
        for ent in doc.ents
    ]
