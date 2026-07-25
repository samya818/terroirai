import os
import json
from typing import Optional

try:
    import google.generativeai as genai
except Exception as exc:
    genai = None
    _GENAI_IMPORT_ERROR = exc
else:
    _GENAI_IMPORT_ERROR = None

# Configurer l'API de Gemini avec la clé d'API
api_key = os.environ.get("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key or ""
if api_key and genai is not None:
    genai.configure(api_key=api_key)

def ask_agronomist(question: str, context: dict) -> str:
    """
    Pose une question agricole à Gemini en lui fournissant tout le contexte
    agronomique et géophysique de la parcelle (rendement estimé, maladies, sol, NDVI).
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return (
            "Désolé, la clé d'API GEMINI_API_KEY n'est pas configurée dans l'environnement. "
            "Veuillez configurer la clé pour recevoir des conseils personnalisés de l'agronome virtuel en darija."
        )

    prompt = f"""
    Tu es un conseiller agronome marocain expérimenté. Tu t'adresses à un agriculteur local.
    Ton rôle est d'analyser les données de sa parcelle agricole et de répondre à sa question de manière très claire, pragmatique et rassurante en utilisant la Darija marocaine (en alphabet latin/arabe selon le format usuel ou en lettres latines simples et lisibles : "arabizi").

    Voici les données physiques et les diagnostics de la parcelle :
    - Culture : {context.get('crop_type', 'Non spécifiée')}
    - Rendement estimé (WOFOST) : {context.get('yield_estimate')} kg/ha
    - Maladie détectée : {context.get('disease', 'aucune maladie détectée')}
    - Indice NDVI (Santé végétative par satellite) : {context.get('ndvi', 'Non disponible')}
    - Caractéristiques du sol :
      * pH : {context.get('soil_ph', 'Non disponible')}
      * Argile : {context.get('clay_pct', 'Non disponible')}%
      * Sable : {context.get('sand_pct', 'Non disponible')}%

    Question de l'agriculteur : "{question}"

    Instructions de réponse :
    1. Réponds en Darija marocaine (lettres latines simples, style SMS/conversationnel propre).
    2. Utilise les données fournies pour justifier ton conseil (ex: si le pH est bas ou s'il y a du mildiou, dis-lui quoi faire spécifiquement pour sa culture).
    3. Reste court, concret, avec des actions applicables sur le terrain.
    """
    
    if genai is None:
        return (
            "Désolé, l'API Gemini n'est pas disponible dans cet environnement. "
            "Le flux de conseil agronomique passe en mode dégradé."
        )

    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur lors de la génération de la réponse de l'agronome : {str(e)}"


def ask_agronomist_interactive(question: str, history: list, current_form: dict, context: Optional[dict] = None) -> dict:
    """
    Mode conversationnel interactif.
    Permet de guider l'agriculteur à remplir sa parcelle et d'extraire les paramètres (crop, sowing_date, location, etc.).
    Retourne un dictionnaire:
    {
       "response": "Le message à afficher à l'utilisateur",
       "extracted_data": {
           "crop_type": "wheat" | "barley" | "olive" | "potato" | "tomato" | null,
           "sowing_date": "YYYY-MM-DD" | null,
           "location": "Nom de ville ou région au Maroc" | null,
           "historical_yields": [{"year": int, "yield": float}] | null
       }
    }
    """
    def normalize_crop(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        text = text.lower()
        if any(keyword in text for keyword in ['blé', 'ble', 'qam7', 'qamh', 'wheat', 'kam7']):
            return 'wheat'
        if any(keyword in text for keyword in ['orge', 'barley', 'ch3ir']):
            return 'barley'
        if any(keyword in text for keyword in ['olive', 'zitoun', 'zitoune']):
            return 'olive'
        if any(keyword in text for keyword in ['pomme de terre', 'batata', 'potato', 'batat', 'patate']):
            return 'potato'
        if any(keyword in text for keyword in ['tomate', 'tomato', 'tamata']):
            return 'tomato'
        return None

    def parse_date(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        normalized = text.lower().replace('/', '-').replace('.', '-').replace(',', ' ')
        tokens = normalized.split()
        for token in tokens:
            if '-' in token:
                parts = token.split('-')
                if len(parts) == 3:
                    try:
                        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                        if 1 <= day <= 31 and 1 <= month <= 12 and year > 1900:
                            return f"{year:04d}-{month:02d}-{day:02d}"
                    except Exception:
                        continue
        months = {
            'janvier': '01', 'février': '02', 'fevrier': '02', 'mars': '03', 'avril': '04', 'mai': '05', 'juin': '06',
            'juillet': '07', 'août': '08', 'aout': '08', 'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
        }
        words = [w.strip('.,') for w in normalized.split()]
        for word in words:
            if word in months:
                day = None
                year = None
                for token in words:
                    if token.isdigit() and 1 <= int(token) <= 31 and day is None:
                        day = int(token)
                    if token.isdigit() and len(token) == 4:
                        year = int(token)
                if day and year:
                    return f"{year:04d}-{int(months[word]):02d}-{day:02d}"
        return None

    def parse_location(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        lower = text.lower()
        places = {
            'beni mellal': 'Beni Mellal',
            'بني ملال': 'Beni Mellal',
            'casablanca': 'Casablanca',
            'الدار البيضاء': 'Casablanca',
            'rabat': 'Rabat',
            'الرباط': 'Rabat',
            'marrakech': 'Marrakech',
            'مراكش': 'Marrakech',
            'fes': 'Fes',
            'fès': 'Fes',
            'فاس': 'Fes'
        }
        for key, value in places.items():
            if key in lower:
                return value
        return None

    def parse_historical(text: Optional[str]):
        if not text:
            return None
        tokens = [token.strip('.,') for token in text.replace('/', ' ').split()]
        results = []
        for idx, token in enumerate(tokens):
            if token.isdigit() and len(token) == 4:
                year = int(token)
                if idx + 1 < len(tokens) and tokens[idx + 1].replace('.', '', 1).isdigit():
                    results.append({'year': year, 'yield': float(tokens[idx + 1])})
        return results if results else None

    def build_response(message: str, extracted: dict) -> dict:
        return {
            'response': message,
            'extracted_data': extracted
        }

    if not os.environ.get("GEMINI_API_KEY"):
        return build_response(
            "Désolé, la clé d'API GEMINI_API_KEY n'est pas configurée dans l'environnement.",
            {}
        )

    crop_type = normalize_crop(current_form.get('crop_type'))
    sowing_date = current_form.get('sowing_date')
    location = current_form.get('location')
    historical_yields = current_form.get('historical_yields')
    photo_uploaded = bool(current_form.get('photo_uploaded'))

    extracted_data = {
        'crop_type': None,
        'sowing_date': None,
        'location': None,
        'historical_yields': None
    }

    if not crop_type:
        extracted_data['crop_type'] = normalize_crop(question)
    if not sowing_date:
        extracted_data['sowing_date'] = parse_date(question)
    if not location:
        extracted_data['location'] = parse_location(question)
    if not historical_yields:
        extracted_data['historical_yields'] = parse_historical(question)

    if any(extracted_data.values()):
        response_parts = []
        if extracted_data['crop_type']:
            response_parts.append(f"Très bien, je note {extracted_data['crop_type']} comme culture.")
        if extracted_data['sowing_date']:
            response_parts.append(f"Date de semis enregistrée : {extracted_data['sowing_date']}.")
        if extracted_data['location']:
            response_parts.append(f"Localisation capturée : {extracted_data['location']}.")
        if extracted_data['historical_yields']:
            response_parts.append("Historique de rendement ajouté.")
        response_parts.append("Je peux maintenant continuer avec la suite du formulaire.")
        return build_response(' '.join(response_parts), extracted_data)

    missing_fields = []
    if not crop_type:
        missing_fields.append("le type de culture")
    if not photo_uploaded:
        missing_fields.append("la photo de la feuille")
    if not location:
        missing_fields.append("la localisation")
    if not sowing_date:
        missing_fields.append("la date de semis")

    # Mettre à jour les variables avec ce qui a été extrait localement de la question courante
    if extracted_data['crop_type']:
        crop_type = extracted_data['crop_type']
        if "le type de culture" in missing_fields:
            missing_fields.remove("le type de culture")
    if extracted_data['sowing_date']:
        sowing_date = extracted_data['sowing_date']
        if "la date de semis" in missing_fields:
            missing_fields.remove("la date de semis")
    if extracted_data['location']:
        location = extracted_data['location']
        if "la localisation" in missing_fields:
            missing_fields.remove("la localisation")

    if genai is None:
        # Fallback si Gemini n'est pas disponible
        fallback_msg = "Désolé, Gemini n'est pas disponible actuellement. "
        if missing_fields:
            fallback_msg += f"Pour lancer l'analyse complète, veuillez renseigner : {', '.join(missing_fields)}."
        return build_response(fallback_msg, extracted_data)

    prompt = f"""
    Tu es un conseiller agronome marocain virtuel. Réponds de manière utile et professionnelle à la question de l'agriculteur (en darija marocaine écrite en lettres latines ou arabizi, ou en français si la question est en français).
    
    Ton objectif est d'aider l'agriculteur sur ses questions agricoles générales ou spécifiques, tout en l'accompagnant à compléter les informations de sa parcelle.

    Voici les informations de la parcelle actuellement connues :
    - Culture : {crop_type or 'Non spécifiée'}
    - Date de semis : {sowing_date or 'Non spécifiée'}
    - Localisation : {location or 'Non spécifiée'}
    - Historique de rendement : {historical_yields or 'Aucun historique renseigné'}
    - Photo de la feuille téléchargée : {'Oui' if photo_uploaded else 'Non'}

    Voici les informations manquantes prioritaires pour exécuter la simulation et la fusion de rendement :
    {", ".join(missing_fields) if missing_fields else "Aucune, toutes les informations clés sont prêtes !"}

    Message / Question de l'agriculteur : "{question}"

    Instructions de réponse :
    1. Réponds précisément à la question de l'agriculteur d'abord (que ce soit une question sur l'irrigation, les engrais, les maladies ou générale).
    2. À la fin de ta réponse, si des informations clés sont manquantes, rappelle-lui de manière amicale et non bloquante de les fournir (ex: en sélectionnant sa culture, en chargeant une photo de feuille ou en indiquant le lieu sur la carte) afin de pouvoir lancer l'analyse agronomique complète de sa parcelle.
    3. Si le message de l'agriculteur contenait de nouvelles informations pour le formulaire (ex: "j'ai semé du blé" ou "je suis à Béni Mellal"), extraits-les.
    4. Retourne TOUJOURS un JSON strict ayant exactement ce format :
    {{
      "response": "Ta réponse d'agronome suivie du rappel amical",
      "extracted_data": {{
         "crop_type": "wheat" ou "barley" ou "olive" ou "potato" ou "tomato" ou null (uniquement si détecté dans le dernier message),
         "sowing_date": "YYYY-MM-DD" ou null (uniquement si détecté dans le dernier message),
         "location": "Nom de la ville/région au Maroc" ou null (uniquement si détecté dans le dernier message),
         "historical_yields": [ {{"year": 2024, "yield": 4500.0}} ] ou null (uniquement si détecté dans le dernier message)
      }}
    }}
    """

    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        text = response.text.strip()
        res_data = json.loads(text)
        if not isinstance(res_data, dict):
            raise ValueError('Format JSON invalide')
        
        # Combiner les données extraites localement et via Gemini
        gemini_extracted = res_data.get('extracted_data', {}) or {}
        final_extracted = {
            'crop_type': extracted_data['crop_type'] or gemini_extracted.get('crop_type'),
            'sowing_date': extracted_data['sowing_date'] or gemini_extracted.get('sowing_date'),
            'location': extracted_data['location'] or gemini_extracted.get('location'),
            'historical_yields': extracted_data['historical_yields'] or gemini_extracted.get('historical_yields')
        }
        
        return {
            'response': res_data.get('response', "Je n'ai pas compris, peux-tu reformuler ?"),
            'extracted_data': final_extracted
        }
    except Exception as e:
        print(f"Erreur ask_agronomist_interactive: {e}")
        return build_response(
            "J'ai rencontré un problème technique avec Gemini. Je continue en mode dégradé.",
            extracted_data
        )
