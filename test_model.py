import spacy

# Load the trained model
model_path = "ner_model_conll_improved_best"  # Update if needed
nlp = spacy.load(model_path)
from spacy.scorer import Scorer
from spacy.training import Example

# Check if the model is correctly loaded
print(f"✅ Model loaded from {model_path}")

test_sentences = [
    "Elon Musk founded Tesla and SpaceX.",
    "Microsoft is a tech company based in Redmond.",
    "Google is headquartered in California.",
    "Barack Obama was president of the United States.",
    "The European Union held meetings in Brussels.",
]

for text in test_sentences:
    doc = nlp(text)
    print(f"\n📜 Text: {text}")
    print("🔹 Recognized Entities:")

    if len(doc.ents) == 0:
        print("  - No entities recognized")
    else:
        for ent in doc.ents:
            print(f"  - '{ent.text}' → {ent.label_}")


def evaluate_model(nlp, examples):
    """
    Evaluate model performance on validation examples.

    Args:
        nlp (spacy.Language): Trained spaCy model
        examples (list): List of Example objects for evaluation.

    Returns:
        dict: Evaluation scores.
    """
    scorer = Scorer()
    pred_examples = [
        Example(nlp(example.text), example.reference) for example in examples
    ]
    scores = scorer.score(pred_examples)  # ✅ FIXED: Correct scoring method

    return {
        "ner_p": scores["ents_p"],
        "ner_r": scores["ents_r"],
        "ner_f": scores["ents_f"],
    }


# Run evaluation (if you have `dev_data`)
dev_data = [
    (
        "Microsoft was founded by Bill Gates.",
        {"entities": [(0, 9, "ORG"), (25, 35, "PERSON")]},
    ),
    (
        "Facebook and Twitter are social media platforms.",
        {"entities": [(0, 8, "ORG"), (13, 20, "ORG")]},
    ),
    ("Paris is the capital of France.", {"entities": [(0, 5, "LOC"), (24, 30, "LOC")]}),
]

# Convert validation data into Example objects
dev_examples = [
    Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in dev_data
]

# Get scores
scores = evaluate_model(nlp, dev_examples)

# Print scores
print(
    f"\n📊 Precision: {scores['ner_p']:.4f}, Recall: {scores['ner_r']:.4f}, F1-score: {scores['ner_f']:.4f}"
)
