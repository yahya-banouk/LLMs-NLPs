import boto3
import json
import pdfplumber
import re

# Configuration AWS
REGION = "us-east-1"
INFERENCE_PROFILE_ARN = "arn:aws:bedrock:us-east-1:078143108430:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0"
bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=REGION)

# Nettoyage du texte extrait
def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'(?<=\w) (?=\w)', '', text)  # Supprimer les espaces inutiles entre les lettres
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return clean_text(text)

# Structure de sortie attendue
EXAMPLE_STRUCTURE = {
    "personal_info": {
        "full_name": "",
        "email": "",
        "phone": "",
        "address": "",
        "city": "",
        "postal_code": "",
        "country": "",
        "birth_date": "",
        "linkedin": "",
        "github": "",
        "languages": []
    },
    "experiences": [],
    "education": [],
    "certifications": [],
    "skills": [],
    "projects": [],
    "bio": ""
}

def create_payload(prompt: str, example_json: dict, instruction: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{instruction}\n\n"
                            f"Texte à analyser :\n{prompt}\n\n"
                            f"Structure JSON attendue :\n{json.dumps(example_json, indent=2)}"
                        )
                    }
                ]
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.2,
        "anthropic_version": "bedrock-2023-05-31"
    }

def invoke_model(prompt: str, example_json: dict, instruction: str):
    payload = create_payload(prompt, example_json, instruction)
    try:
        response = bedrock_runtime_client.invoke_model(
            modelId=INFERENCE_PROFILE_ARN,
            body=json.dumps(payload),
        )
        response_body = response['body'].read().decode('utf-8')
        result = json.loads(response_body)
        content_text = result.get("content", [{}])[0].get("text", "")
        return json.loads(content_text)
    except json.JSONDecodeError as e:
        print("Erreur JSON : ", e)
        print("Réponse brute :\n", content_text)
    except Exception as e:
        print(f"Erreur invocation modèle : {e}")
    return None


if __name__ == "__main__":
    instruction = (
        "Tu es un assistant intelligent. Ton rôle est d’extraire les informations structurées d’un CV brut. "
        "Tu dois générer un JSON contenant : les informations personnelles (nom, email, téléphone, adresse, ville, pays, date de naissance, LinkedIn, GitHub, langues), "
        "les expériences professionnelles (entreprise, titre, dates, description), les formations (diplôme, établissement, année), "
        "les certifications, les compétences, les projets, et une courte bio. "
        "Utilise le maximum de détails disponibles. Si une information est absente, laisse-la vide. "
        "Le format doit être un JSON strictement valide et bien formé."
    )

    pdf_path = "./YB.pdf"
    raw_text = extract_text_from_pdf(pdf_path)

    result = invoke_model(raw_text, EXAMPLE_STRUCTURE, instruction)

    print("\n===== JSON STRUCTURÉ =====\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
