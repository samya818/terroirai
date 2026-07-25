import os
import numpy as np
from PIL import Image
from typing import Tuple

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    _TF_IMPORT_ERROR = e
else:
    _TF_IMPORT_ERROR = None

# Cache pour les modèles chargés
_models = {}

CLASSES_PLANT_VILLAGE = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_YellowLeaf___Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

CLASSES_OLIVE = [
    "Olive___Bacterial_leaf_spot",
    "Olive___Peacock_spot",
    "Olive___healthy"
]

CLASSES_CITRUS = [
    "Citrus___Anthracnose",
    "Citrus___Bacterial_Blight",
    "Citrus___Citrus_Canker",
    "Citrus___Curl_Virus",
    "Citrus___Deficiency_Leaf",
    "Citrus___Dry_Leaf",
    "Citrus___healthy",
    "Citrus___Sooty_Mould",
    "Citrus___Spider_Mites"
]

CLASSES_WHEAT = [
    "Wheat___Aphid",
    "Wheat___Mite",
    "Wheat___Stem_Fly",
    "Wheat___Black_Rust",
    "Wheat___Brown_Rust",
    "Wheat___Yellow_Rust",
    "Wheat___Smut",
    "Wheat___Common_Root_Rot",
    "Wheat___Leaf_Blight",
    "Wheat___Wheat_Blast",
    "Wheat___Fusarium_Head_Blight",
    "Wheat___Septoria_Leaf_Blotch",
    "Wheat___Spot_Blotch",
    "Wheat___Tan_Spot",
    "Wheat___healthy"
]

# Mapping des cultures vers la clé de leur modèle respectif
CROP_MAPPING = {
    # Français
    'tomate': 'PlantVillage',
    'pomme_de_terre': 'PlantVillage',
    'pomme de terre': 'PlantVillage',
    'raisin': 'PlantVillage',
    'mais': 'PlantVillage',
    'maïs': 'PlantVillage',
    'pomme': 'PlantVillage',
    'peche': 'PlantVillage',
    'pêche': 'PlantVillage',
    'poivron': 'PlantVillage',
    'orange': 'PlantVillage',
    'fraise': 'PlantVillage',
    'courge': 'PlantVillage',
    'ble': 'weatdiseases',
    'blé': 'weatdiseases',
    'orge': 'weatdiseases',
    'olivier': 'OliveLeaf',
    'citron': 'citrus_leaf',
    'mandarine': 'citrus_leaf',
    # Anglais / Backend Standard
    'tomato': 'PlantVillage',
    'potato': 'PlantVillage',
    'grape': 'PlantVillage',
    'corn': 'PlantVillage',
    'maize': 'PlantVillage',
    'apple': 'PlantVillage',
    'peach': 'PlantVillage',
    'pepper': 'PlantVillage',
    'strawberry': 'PlantVillage',
    'squash': 'PlantVillage',
    'wheat': 'weatdiseases',
    'barley': 'weatdiseases',
    'olive': 'OliveLeaf',
    'citrus': 'citrus_leaf',
    'lemon': 'citrus_leaf',
    'mandarin': 'citrus_leaf',
}

# Modèles et chemins correspondants
MODEL_PATHS = {
    'PlantVillage': ('PlantVillage', 'model.h5'),
    'weatdiseases': ('weatdiseases', 'wheat_diseases.h5'),
    'OliveLeaf': ('OliveLeaf', 'olive_leaf.h5'),
    'citrus_leaf': ('citrus_leaf', 'citrus_leaf_p1.h5')
}

MODEL_CLASSES = {
    'PlantVillage': CLASSES_PLANT_VILLAGE,
    'weatdiseases': CLASSES_WHEAT,
    'OliveLeaf': CLASSES_OLIVE,
    'citrus_leaf': CLASSES_CITRUS
}

# Garder la compatibilité avec la variable globale de l'ancienne version
CLASSES = CLASSES_PLANT_VILLAGE

def load_model(model_key: str = 'PlantVillage'):
    """
    Charge de manière paresseuse un modèle spécifique par sa clé et le met en cache.
    """
    global _models
    if tf is None:
        raise RuntimeError(f"TensorFlow indisponible : {_TF_IMPORT_ERROR}")

    if model_key not in _models:
        if model_key not in MODEL_PATHS:
            raise ValueError(f"Modèle inconnu : {model_key}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder, filename = MODEL_PATHS[model_key]
        model_path = os.path.join(script_dir, 'models', folder, filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le modèle '{model_path}' n'existe pas.")

        _models[model_key] = tf.keras.models.load_model(model_path, compile=False)

    return _models[model_key]

def _normalize_rgb_for_segmentation(img: Image.Image) -> np.ndarray:
    """
    Normalise l'exposition et l'éclairage de l'image avant la segmentation couleur.
    L'objectif est de réduire l'impact des ombres et des reflets sur le calcul de sévérité.
    """
    rgb = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0

    # Réduction des effets de contraste extrême sur les pixels très sobres ou très lumineux
    lower, upper = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - lower) / (upper - lower + 1e-8), 0.0, 1.0)

    return rgb


def estimate_severity(img: Image.Image) -> float:
    """
    Estime la sévérité d'une infection foliaire avec une segmentation couleur plus robuste.

    Approche retenue :
    1. normalisation de l'éclairage de l'image,
    2. segmentation HSV de la feuille visible,
    3. séparation des zones vertes saines vs. tissus lésés/jaunes/bruns,
    4. ratio de pixels malades sur la surface foliaire totale.
    """
    rgb = _normalize_rgb_for_segmentation(img)
    hsv = np.array(np.round(rgb * 255.0), dtype=np.uint8)
    hsv = np.array(Image.fromarray(hsv, mode='RGB').convert('HSV'))

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 1. Masque de feuille plus robuste : on exclut les tons trop sombres, trop clairs
    #    et les zones très neutres (saturation faible), qui correspondent souvent au sol ou au fond.
    leaf_mask = (s > 25) & (v > 35) & (v < 240)

    # 2. Vert sain : plage plus stricte pour éviter de tenir compte du bruit de fond.
    healthy_mask = leaf_mask & (h >= 36) & (h <= 105) & (s >= 40)

    # 3. Tissus malades : toute partie verte visible mais non saine, plus les zones
    #    jaune/brun/rouge qui restent sur la feuille.
    #    On garde ainsi une segmentation plus réaliste sur les images de terrain.
    diseased_mask = leaf_mask & (~healthy_mask)

    healthy_pixels = np.sum(healthy_mask)
    diseased_pixels = np.sum(diseased_mask)
    total_leaf_pixels = healthy_pixels + diseased_pixels

    if total_leaf_pixels <= 0:
        return 0.15

    severity = diseased_pixels / total_leaf_pixels
    severity = float(np.clip(severity, 0.05, 0.95))
    return severity

def predict_disease(img: Image.Image, crop_type: str = 'tomato') -> Tuple[str, float, float]:
    """
    Reçoit une instance PIL Image et le type de culture (crop_type),
    charge le modèle approprié, effectue la prédiction, estime la sévérité
    et renvoie un tuple (nom_classe, confiance_float, severite_float).
    """
    severity_pct = estimate_severity(img)
    severity_pct = max(0.05, min(0.95, severity_pct))

    if tf is None:
        return "Unknown", 0.0, 0.0

    try:
        # Résolution et normalisation du modèle pour le type de culture
        crop_normalized = str(crop_type).lower().strip()
        model_key = CROP_MAPPING.get(crop_normalized, 'PlantVillage')

        model = load_model(model_key)
        class_list = MODEL_CLASSES[model_key]

        # Prétraitement de l'image (224, 224, 3)
        img_resized = img.convert('RGB').resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_idx])
        predicted_class = class_list[predicted_idx] if predicted_idx < len(class_list) else "Unknown"

        # Estimation de la sévérité : désactivée si le modèle ne détecte pas de maladie (sain ou inconnu)
        if "healthy" in predicted_class.lower() or predicted_class == "Unknown":
            severity_pct = 0.0
        else:
            severity_pct = max(0.05, min(0.95, severity_pct))

        return predicted_class, confidence, severity_pct
    except Exception as exc:
        print(f"[predict_disease] Fallback simple activé : {exc}")
        return "Unknown", 0.0, 0.0
