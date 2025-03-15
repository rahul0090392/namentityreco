from fastapi import FastAPI

from app.ner_model import load_model, predict_entities

app = FastAPI()
nlp = load_model()

@app.post("/predict")
def predict(text: str):
    entities = predict_entities(nlp, text)
    return {"entities": entities}
