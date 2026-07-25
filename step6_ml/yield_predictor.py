import os
import pickle
import numpy as np
from typing import Tuple, Optional

try:
    import pandas as pd
except Exception as exc:
    pd = None
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception as exc:
    RandomForestRegressor = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "yield_rf_model.pkl")
EXPECTED_FEATURES = [
    "wofost_yield", "ndvi", "clay_pct", "sand_pct", "soil_ph",
    "accumulated_temp", "disease_severity", "soil_moisture_pct",
    "season_rainfall_mm", "growth_window_avg_temp",
    "vegetative_stage_idx", "irrigation_idx", "fertilization_idx", "variety_idx"
]

def train_and_save_model(num_samples: int = 5000):
    """
    Génère un jeu de données synthétique pour entraîner le modèle Random Forest,
    basé sur des relations agronomiques réalistes.
    """
    if pd is None:
        return None

    np.random.seed(42)
    
    # 1. Variables d'entrée (Features)
    wofost_yield = np.random.uniform(1500, 7000, num_samples)  # kg/ha
    ndvi = np.random.uniform(0.15, 0.9, num_samples)
    clay_pct = np.random.uniform(10, 50, num_samples)
    sand_pct = np.random.uniform(10, 70, num_samples)
    soil_ph = np.random.uniform(5.5, 8.5, num_samples)
    accumulated_temp = np.random.uniform(1500, 4500, num_samples)
    disease_severity = np.random.uniform(0.0, 0.9, num_samples)
    soil_moisture_pct = np.random.uniform(12.0, 45.0, num_samples)
    season_rainfall_mm = np.random.uniform(100.0, 600.0, num_samples)
    growth_window_avg_temp = np.random.uniform(11.0, 31.0, num_samples)
    vegetative_stage_idx = np.random.randint(0, 4, num_samples)
    irrigation_idx = np.random.randint(0, 4, num_samples)
    fertilization_idx = np.random.randint(0, 4, num_samples)
    variety_idx = np.random.randint(0, 3, num_samples)
    
    # Rendre les feuilles saines cohérentes avec une sévérité nulle
    healthy_indices = np.random.choice(num_samples, int(num_samples * 0.3), replace=False)
    disease_severity[healthy_indices] = 0.0

    # 2. Relation agronomique cible (Rendement Réel) avec bruit
    ndvi_factor = np.clip(ndvi / 0.75, 0.2, 1.2)
    disease_factor = 1.0 - (disease_severity * 0.75)
    ph_factor = 1.0 - 0.1 * np.abs(soil_ph - 6.5)
    soil_water_factor = 1.0 - 0.002 * np.abs(clay_pct - 30) - 0.001 * np.abs(sand_pct - 35)
    soil_water_factor = np.clip(soil_water_factor, 0.7, 1.0)
    water_supply_factor = 1.0 - 0.003 * np.abs(soil_moisture_pct - 28) + 0.001 * season_rainfall_mm / 100
    water_supply_factor = np.clip(water_supply_factor, 0.65, 1.15)
    temp_factor = 1.0 - 0.0001 * np.abs(accumulated_temp - 3000)
    temp_factor = np.clip(temp_factor, 0.6, 1.1)
    stage_factor = 1.0 + (vegetative_stage_idx - 1.5) * 0.04
    irrigation_factor = 1.0 + (irrigation_idx - 1.5) * 0.05
    fertilization_factor = 1.0 + (fertilization_idx - 1.5) * 0.06
    variety_factor = 1.0 + (variety_idx - 1.0) * 0.03

    final_yield = (
        wofost_yield * ndvi_factor * disease_factor * ph_factor
        * soil_water_factor * water_supply_factor * temp_factor
        * stage_factor * irrigation_factor * fertilization_factor * variety_factor
    )

    noise = np.random.normal(0, 0.08, num_samples)
    final_yield = final_yield * (1 + noise)
    final_yield = np.clip(final_yield, 200, 8500)

    df = pd.DataFrame({
        "wofost_yield": wofost_yield,
        "ndvi": ndvi,
        "clay_pct": clay_pct,
        "sand_pct": sand_pct,
        "soil_ph": soil_ph,
        "accumulated_temp": accumulated_temp,
        "disease_severity": disease_severity,
        "soil_moisture_pct": soil_moisture_pct,
        "season_rainfall_mm": season_rainfall_mm,
        "growth_window_avg_temp": growth_window_avg_temp,
        "vegetative_stage_idx": vegetative_stage_idx,
        "irrigation_idx": irrigation_idx,
        "fertilization_idx": fertilization_idx,
        "variety_idx": variety_idx,
        "final_yield": final_yield
    })

    X = df[[
        "wofost_yield", "ndvi", "clay_pct",
        "sand_pct", "soil_ph", "accumulated_temp",
        "disease_severity", "soil_moisture_pct",
        "season_rainfall_mm", "growth_window_avg_temp",
        "vegetative_stage_idx", "irrigation_idx",
        "fertilization_idx", "variety_idx"
    ]]
    y = df["final_yield"]

    print("Entraînement du modèle Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Créer le répertoire si inexistant
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé avec succès dans {MODEL_PATH}")
    return model


def generate_synthetic_data(num_samples: int = 5000):
    return train_and_save_model(num_samples=num_samples)


def get_trained_model():
    """
    Charge et renvoie le modèle entraîné. L'entraîne s'il n'existe pas encore.
    """
    if RandomForestRegressor is None:
        return None

    if not os.path.exists(MODEL_PATH):
        train_and_save_model()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    fitted_features = getattr(model, "feature_names_in_", None)
    if fitted_features is None or list(fitted_features) != EXPECTED_FEATURES:
        print("Le modèle enregistré est obsolète : entraînement du modèle sur le schéma courant.")
        train_and_save_model()
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    return model

def _normalize_stage_index(stage: Optional[str]) -> int:
    if stage is None:
        return 1
    stage_map = {
        'early': 0,
        'vegetative': 1,
        'flowering': 2,
        'maturity': 3,
    }
    return stage_map.get(str(stage).lower().strip(), 1)


def _normalize_level_index(level: Optional[str]) -> int:
    if level is None:
        return 1
    level_map = {
        'faible': 0,
        'moyen': 1,
        'fort': 2,
        'elevé': 3,
        'low': 0,
        'medium': 1,
        'high': 2,
        'very_high': 3,
    }
    return level_map.get(str(level).lower().strip(), 1)


def _normalize_variety_index(variety: Optional[str]) -> int:
    if variety is None:
        return 1
    variety_map = {
        'picholine': 0,
        'marocaine': 1,
        'autre': 2,
        'local': 1,
        'improved': 2,
        'hybrid': 2,
    }
    return variety_map.get(str(variety).lower().strip(), 1)


def predict_final_yield(
    wofost_yield: float,
    ndvi: float,
    clay_pct: float,
    sand_pct: float,
    soil_ph: float,
    accumulated_temp: float,
    disease_severity: float,
    soil_moisture_pct: Optional[float] = None,
    season_rainfall_mm: Optional[float] = None,
    growth_window_avg_temp: Optional[float] = None,
    vegetative_stage: Optional[str] = None,
    irrigation_level: Optional[str] = None,
    fertilization_level: Optional[str] = None,
    variety: Optional[str] = None
) -> Tuple[float, float, float, float]:
    """
    Utilise le modèle Random Forest pour prédire le rendement final et renvoie
    (rendement_estime, intervalle_bas, intervalle_haut, confiance_finale).
    """
    model = get_trained_model()

    if model is None:
        ndvi_value = ndvi if ndvi is not None else 0.5
        disease_value = disease_severity if disease_severity is not None else 0.0
        soil_ph_value = soil_ph if soil_ph is not None else 6.8
        clay_value = clay_pct if clay_pct is not None else 30.0
        sand_value = sand_pct if sand_pct is not None else 40.0
        temp_value = accumulated_temp if accumulated_temp is not None else 3000.0
        moisture_value = soil_moisture_pct if soil_moisture_pct is not None else 28.0
        rainfall_value = season_rainfall_mm if season_rainfall_mm is not None else 250.0
        avg_temp_value = growth_window_avg_temp if growth_window_avg_temp is not None else 20.0
        stage_idx = _normalize_stage_index(vegetative_stage)
        irrigation_idx = _normalize_level_index(irrigation_level)
        fertilization_idx = _normalize_level_index(fertilization_level)
        variety_idx = _normalize_variety_index(variety)

        ndvi_factor = min(1.0, max(0.2, ndvi_value / 0.75))
        disease_factor = max(0.1, 1.0 - disease_value * 0.75)
        ph_factor = max(0.7, 1.0 - 0.1 * abs(soil_ph_value - 6.5))
        soil_factor = max(0.7, 1.0 - 0.002 * abs(clay_value - 30) - 0.001 * abs(sand_value - 35))
        water_factor = max(0.65, 1.0 - 0.003 * abs(moisture_value - 28) + 0.001 * rainfall_value / 100)
        temp_factor = max(0.6, 1.0 - 0.0001 * abs(temp_value - 3000))
        stage_factor = 1.0 + (stage_idx - 1.5) * 0.04
        irrigation_factor = 1.0 + (irrigation_idx - 1.5) * 0.05
        fertilization_factor = 1.0 + (fertilization_idx - 1.5) * 0.06
        variety_factor = 1.0 + (variety_idx - 1.0) * 0.03

        prediction = float(
            (wofost_yield if wofost_yield is not None else 3500.0)
            * ndvi_factor * disease_factor * ph_factor * soil_factor
            * water_factor * temp_factor * stage_factor
            * irrigation_factor * fertilization_factor * variety_factor
        )
        std_error = prediction * 0.10
        confidence_interval_low = max(0.0, prediction - 1.96 * std_error)
        confidence_interval_high = prediction + 1.96 * std_error
        confidence_score = float(np.clip(0.55 + 0.15 * (avg_temp_value / 30.0) + 0.1 * (min(1.0, rainfall_value / 350.0)) + 0.1 * (1.0 - disease_value), 0.2, 0.95))
        return prediction, confidence_interval_low, confidence_interval_high, confidence_score

    if pd is None:
        return 0.0, 0.0, 0.0, 0.0

    # Inputs
    features = pd.DataFrame([{
        "wofost_yield": wofost_yield if wofost_yield is not None else 3500.0,
        "ndvi": ndvi if ndvi is not None else 0.5,
        "clay_pct": clay_pct if clay_pct is not None else 30.0,
        "sand_pct": sand_pct if sand_pct is not None else 40.0,
        "soil_ph": soil_ph if soil_ph is not None else 6.8,
        "accumulated_temp": accumulated_temp if accumulated_temp is not None else 3000.0,
        "disease_severity": disease_severity if disease_severity is not None else 0.0,
        "soil_moisture_pct": soil_moisture_pct if soil_moisture_pct is not None else 28.0,
        "season_rainfall_mm": season_rainfall_mm if season_rainfall_mm is not None else 250.0,
        "growth_window_avg_temp": growth_window_avg_temp if growth_window_avg_temp is not None else 20.0,
        "vegetative_stage_idx": _normalize_stage_index(vegetative_stage),
        "irrigation_idx": _normalize_level_index(irrigation_level),
        "fertilization_idx": _normalize_level_index(fertilization_level),
        "variety_idx": _normalize_variety_index(variety)
    }])

    prediction = float(model.predict(features)[0])

    std_error = prediction * 0.10
    confidence_interval_low = max(0.0, prediction - 1.96 * std_error)
    confidence_interval_high = prediction + 1.96 * std_error
    confidence_score = float(np.clip(0.55 + 0.15 * ((growth_window_avg_temp if growth_window_avg_temp is not None else 20.0) / 30.0) + 0.1 * (min(1.0, (season_rainfall_mm if season_rainfall_mm is not None else 250.0) / 350.0)) + 0.1 * (1.0 - (disease_severity if disease_severity is not None else 0.0)), 0.2, 0.95))

    return prediction, confidence_interval_low, confidence_interval_high, confidence_score
