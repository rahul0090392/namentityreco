import spacy


def load_model():
    """Load the pre-trained SpaCy NER model."""
    return spacy.load("en_core_web_sm")

def predict_entities(nlp, text):
    """Predict named entities from text."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
