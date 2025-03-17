Named Entity Recognition (NER) API

Overview

This project provides a FastAPI-based Named Entity Recognition (NER) API using a custom-trained spaCy model. The model is trained on the CoNLL 2003 dataset, which includes entity types like:

PERSON (e.g., Elon Musk, Barack Obama)

ORG (e.g., Google, Tesla)

LOC (e.g., New York, California)

MISC (e.g., Olympics, UEFA)

The API is built to load a trained model and predict named entities in input text.

🚀 Features

Pre-trained spaCy NER model fine-tuned on CoNLL 2003

FastAPI-based API for real-time entity recognition

Docker support for easy deployment

Environment variable-based configuration

Lightweight & optimized for production

📥 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/rahul0090392/namentityreco.git
cd namentityreco

2️⃣ Install Dependencies

Ensure you have Python 3.11 installed.

pip install -r requirements.txt

3️⃣ Set Environment Variables

Create a .env file in the root directory and specify the model path:

MODEL_PATH=./app/ner_model

4️⃣ Run the FastAPI Server

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

🐳 Deploy with Docker

1️⃣ Build the Docker Image

docker build -t ner_api .

2️⃣ Run the Container

docker run -d -p 8000:8000 --env-file .env --name ner_api ner_api

3️⃣ Verify the API is Running

curl http://localhost:8000/

Response:

{"message": "NER API is running!"}

🔌 API Endpoints

1️⃣ Health Check

GET /

Response:

{"message": "NER API is running!"}

2️⃣ Named Entity Recognition (NER)

POST /predict

Request:

{
  "text": "Elon Musk founded Tesla and SpaceX."
}

Response:

{
  "entities": [
    {"text": "Elon Musk", "start": 0, "end": 9, "label": "PERSON"},
    {"text": "Tesla", "start": 18, "end": 23, "label": "ORG"},
    {"text": "SpaceX", "start": 28, "end": 34, "label": "ORG"}
  ]
}

🛠 Development & Contribution

1️⃣ Run in Development Mode

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

2️⃣ Code Formatting & Linting

black .
flake8

3️⃣ Run Tests

pytest tests/

📜 License

This project is MIT licensed. See the LICENSE file for details.

🔗 Author

Developed by Rahul Jain.

GitHub: rahul0090392

LinkedIn: rahul-jain