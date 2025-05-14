from pdf2image import convert_from_path
import pytesseract
from transformers import pipeline
import re

# Étape 1 : OCR du PDF
images = convert_from_path("YB.pdf")
text = ""
for image in images:
    text += pytesseract.image_to_string(image, lang="eng+fra") + "\n"

# Étape 2 : Diviser en lignes (ou phrases)
lines = [line.strip() for line in text.split("\n") if line.strip()]

# Étape 3 : Définir les étiquettes que tu veux identifier
candidate_labels = ["name", "email", "phone number", "address", "skills", "experience", "education", "linkedin", "summary"]

# Étape 4 : Charger le modèle zero-shot
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

# Étape 5 : Appliquer le classifieur à chaque ligne
print("\n--- 🧠 Résultat Zero-shot ---\n")
for line in lines:
    result = classifier(line, candidate_labels, multi_label=True)
    for label, score in zip(result['labels'], result['scores']):
        if score > 0.85:  # Seuil de confiance
            print(f"[{label.upper():12}] {line}")
            break
