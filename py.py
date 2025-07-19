import os
from paddleocr import PaddleOCR
import spacypython
from transformers import pipeline

# --------------------------
# Étape 1 : Extraction OCR
# --------------------------
def extract_text(image_path):
    ocr = PaddleOCR(lang="en", use_angle_cls=True)
    result = ocr.ocr(image_path, rec=True)
    extracted_text = ""
    for line in result:
        for res in line:
            text_line = res[1][0]
            extracted_text += text_line + "\n"
    return extracted_text

# --------------------------
# Étape 2 : Segmentation du texte
# --------------------------
def segment_text(text):
    headers = {
        "INFORMATIONS PERSONNELLES": ["INFORMATIONS PERSONNELLES", "COORDONNÉES", "CONTACT"],
        "DIPLOMES": ["DIPLOMES", "FORMATION", "ÉDUCATION"],
        "CERTIFICATIONS": ["CERTIFICATIONS", "CERTIFICATS"],
        "EXPERIENCES": ["EXPERIENCES", "EXPERIENCE", "EXPERIENCES PROFESSIONNELLES", "PARCOURS PROFESSIONNEL"]
    }
    segments = {key: "" for key in headers}
    lines = text.splitlines()
    current_section = None
    for line in lines:
        line_upper = line.strip().upper()
        found_header = False
        for section, header_list in headers.items():
            if any(h in line_upper for h in header_list):
                current_section = section
                found_header = True
                break
        if not found_header and current_section:
            segments[current_section] += line + "\n"
    return segments

# --------------------------
# (Optionnel) Étape 3 : Extraction d'entités avec Transformers
# --------------------------
def extract_entities(text, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")
    return ner_pipeline(text)

# --------------------------
# Pipeline complet (sans sauvegarde)
# --------------------------
def process_cv(image_path):
    # Extraction OCR
    text = extract_text(image_path)
    print("========= TEXTE EXTRAIT =========")
    print(text)

    # Segmentation en sections
    segments = segment_text(text)
    print("\n========= SEGMENTS IDENTIFIÉS =========")
    for key, value in segments.items():
        print(f"\n--- {key} ---")
        print(value.strip())

    # Extraction d'entités
    entities = extract_entities(text)
    print("\n========= ENTITÉS EXTRAIRES VIA TRANSFORMERS =========")
    for ent in entities:
        print(f"{ent['entity_group']}: {ent['word']} (score: {ent['score']:.2f})")

# --------------------------
# Exemple d'utilisation
# --------------------------
if __name__ == "__main__":
    image_path = "YB.png"  # Remplace par le chemin de ton image
    process_cv(image_path)
