import easyocr
from PIL import Image
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pdf2image import convert_from_path
import torch

def pdf_to_image(pdf_path, page_number=0):
    # Convertir le PDF en images (une image par page)
    pages = convert_from_path(pdf_path)
    # Retourner l'image de la page souhaitée (ici la première page)
    return pages[page_number]

def extract_text_and_boxes(image):
    """
    Utilise EasyOCR pour extraire le texte et les bounding boxes d'une image.
    Retourne une liste de mots et une liste de coordonnées sous forme [x_min, y_min, x_max, y_max].
    """
    reader = easyocr.Reader(['fr', 'en'], gpu=False)
    # Convertir l'image PIL en tableau numpy
    image_np = np.array(image)
    results = reader.readtext(image_np, detail=1)
    words = []
    boxes = []
    for detection in results:
        bbox, text, _ = detection
        words.append(text)
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return words, boxes

def normalize_boxes(boxes, image_width, image_height):
    normalized_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        normalized_box = [
            int((x_min / image_width) * 1000),
            int((y_min / image_height) * 1000),
            int((x_max / image_width) * 1000),
            int((y_max / image_height) * 1000)
        ]
        normalized_boxes.append(normalized_box)
    return normalized_boxes

def segment_with_layoutlmv3(image):
    """
    Utilise LayoutLMv3 pour segmenter le contenu du document.
    Si l'entrée est un chemin de fichier, l'image est ouverte et convertie en RGB.
    """
    # Check if the input is a file path (string), then open the image.
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Extraire le texte et les positions
    words, boxes = extract_text_and_boxes(image)
    
    # Normaliser les bounding boxes dans la plage 0-1000
    width, height = image.size
    boxes = normalize_boxes(boxes, width, height)
    
    # Charger le processor et le modèle LayoutLMv3 avec OCR désactivé
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    
    # Préparer l'encodage
    encoding = processor(image, text=words, boxes=boxes, return_tensors="pt", truncation=True)
    
    # Passage par le modèle
    outputs = model(**encoding)
    
    # Récupérer les prédictions
    predictions = outputs.logits.argmax(-1)[0].tolist()
    
    # Décoder les tokens et leurs labels
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    predicted_labels = [model.config.id2label[p] for p in predictions]
    
    segmentation = list(zip(tokens, predicted_labels))
    return segmentation

if __name__ == "__main__":
    pdf_path = "./YB.png"
    segmentation = segment_with_layoutlmv3(pdf_path)
    for token, label in segmentation:
        print(f"{token}: {label}")
