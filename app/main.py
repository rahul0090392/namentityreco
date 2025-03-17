from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from app.ner_model import load_model, predict_entities

# Load environment variables
load_dotenv()

app = FastAPI()

# Load the trained NER model on startup
nlp = load_model()

# Define request model
class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "NER API is running!"}


@app.post("/predict")
def predict(request: TextRequest):
    """
    Predict named entities in the given text.

    Request:
        - text (str): The input text.

    Response:
        - entities (list): List of identified entities.
    """
    entities = predict_entities(nlp, request.text)
    return {"entities": entities}
