from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
import torch
from PIL import Image
import os

# Step 1: Convert PDF to image
images = convert_from_path("YB.pdf")

# Use the first page only for demo (can loop for more)
image = images[0]
image.save("page.png")

# Step 2: OCR to get words and bounding boxes
ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
words = []
boxes = []
for i in range(len(ocr_data["text"])):
    if int(ocr_data["conf"][i]) > 60 and ocr_data["text"][i].strip() != "":
        words.append(ocr_data["text"][i])
        x, y, w, h = (ocr_data["left"][i], ocr_data["top"][i],
                      ocr_data["width"][i], ocr_data["height"][i])
        boxes.append([x, y, x + w, y + h])

# Normalize boxes to 0-1000 (as required by LayoutLMv2)
width, height = image.size
normalized_boxes = []
for box in boxes:
    x0, y0, x1, y1 = box
    normalized_box = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    normalized_boxes.append(normalized_box)

# Step 3: Load processor and model
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)

model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased")

# Step 4: Encode inputs
encoding = processor(image, words, boxes=normalized_boxes, return_tensors="pt", truncation=True, padding="max_length")
outputs = model(**encoding)

# Step 5: Get predictions
predictions = torch.argmax(outputs.logits, dim=-1)
labels = predictions[0].tolist()

# Step 6: Decode predicted labels
tokenizer = processor.tokenizer
tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

print("\n--- Parsed Tokens and Predicted Labels ---")
for token, label in zip(tokens, labels):
    if token not in ["[PAD]", "[CLS]", "[SEP]"]:
        print(f"{token:15s} -> Label ID: {label}")
