import io
import os
from datetime import date
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Importer les modules et le schéma
import json
from schema import ParcelRecord
from step1_disease.predict import predict_disease
from step2_soil.soil_api import get_soil_data
from step3_satellite.satellite_ndvi import get_ndvi
from step4_simulation.simulation_runner import run_wofost_simulation
from step5_llm.llm_service import ask_agronomist
from step6_ml.yield_predictor import predict_final_yield


app = FastAPI(
    title="TerroirAI API Gateway",
    description="API Gateway d'orchestration pour l'estimation de rendements agricoles et le diagnostic des cultures.",
    version="1.0.0"
)

# Nous utilisons Form pour recevoir à la fois le fichier image et les données textuelles/numériques.
# FastAPI gère facilement la réception multipart/form-data.

@app.post("/predict")
async def predict(
    farmer_id: str = Form("farmer_1"),
    lat: float = Form(...),
    lon: float = Form(...),
    crop_type: str = Form(...),
    photo: UploadFile = File(...),
    sowing_date: Optional[str] = Form(None),
    historical_yields: Optional[str] = Form(None),
    soil_moisture_pct: Optional[float] = Form(None),
    season_rainfall_mm: Optional[float] = Form(None),
    growth_window_avg_temp: Optional[float] = Form(None),
    vegetative_stage: Optional[str] = Form(None),
    irrigation_level: Optional[str] = Form(None),
    fertilization_level: Optional[str] = Form(None),
    variety: Optional[str] = Form(None),
    olive_num_trees: Optional[int] = Form(None),
    olive_age: Optional[str] = Form(None),
    olive_last_year: Optional[str] = Form(None),
    olive_size: Optional[str] = Form(None),
    olive_variety: Optional[str] = Form(None)
):
    """
    Point d'entrée unique de TerroirAI.
    Reçoit une photo de la culture + coordonnées, et orchestre le diagnostic, la simulation et le ML.
    """
    parcel_id = f"{farmer_id}_{lat}_{lon}"
    today = date.today()

    # 1. Lecture et validation de la photo
    try:
        contents = await photo.read()
        img = Image.open(io.BytesIO(contents))
        img = img.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fichier image invalide : {str(e)}")

    photo_array = np.asarray(img)
    brightness = float(np.mean(photo_array))
    contrast = float(np.std(photo_array))
    photo_quality_score = float(np.clip(0.4 + (contrast / 80.0) * 0.3 + (brightness / 255.0) * 0.3, 0.1, 0.99))
    
    # 2. Modèle Maladies & Sévérité (Étape 1)
    disease_class = "Unknown"
    disease_confidence = 0.0
    disease_severity = 0.0
    try:
        disease_class, disease_confidence, disease_severity = predict_disease(img, crop_type)
    except Exception as e:
        print(f"Avertissement : Échec de la détection de maladie ({e})")
        disease_class = "Erreur modèle"
        disease_confidence = 0.0
        disease_severity = 0.0

    # 3. Données Sol via SoilGrids API (Étape 2)
    soil_ph = 6.5  # default
    soil_clay_pct = 30.0
    soil_sand_pct = 40.0
    soil_source = 'default'
    soil_degraded = False
    try:
        soil_res = get_soil_data(lat, lon)
        if soil_res is not None:
            if 'phh2o' in soil_res and soil_res['phh2o'] is not None:
                soil_ph = float(soil_res['phh2o']) / 10.0
            if 'clay' in soil_res and soil_res['clay'] is not None:
                soil_clay_pct = float(soil_res['clay']) / 10.0
            if 'sand' in soil_res and soil_res['sand'] is not None:
                soil_sand_pct = float(soil_res['sand']) / 10.0
            soil_source = soil_res.get('source', 'SoilGrids')
            soil_degraded = soil_source != 'SoilGrids'
    except Exception as e:
        print(f"Avertissement : Échec de la récupération des données de sol ({e})")
        soil_degraded = True
        soil_source = 'error'

    # 4. Satellite NDVI via Google Earth Engine (Étape 3)
    ndvi_val = 0.45  # default
    ndvi_date_val = today
    ndvi_signal_quality = 0.4
    ndvi_source = 'default'
    ndvi_degraded = False
    try:
        # Période par défaut : le mois écoulé
        date_start = date(today.year, today.month - 1 if today.month > 1 else 12, today.day).isoformat()
        date_end = today.isoformat()
        ndvi_val = get_ndvi(lat, lon, date_start, date_end)
        ndvi_source = 'GEE'
        if ndvi_val is not None:
            ndvi_date_val = today
            ndvi_signal_quality = float(np.clip(0.55 + min(0.35, max(0.0, ndvi_val)) * 0.45, 0.2, 0.95))
        else:
            ndvi_val = 0.45
            ndvi_degraded = True
            ndvi_source = 'default'
    except Exception as e:
        print(f"Avertissement : Échec de l'obtention du NDVI Sentinel-2 ({e}). Utilisation d'un NDVI par défaut.")
        ndvi_val = 0.45
        ndvi_degraded = True
        ndvi_source = 'error'

    # Parse historical yields list
    parsed_historical = []
    if historical_yields:
        try:
            hist_list = json.loads(historical_yields)
            for item in hist_list:
                y = int(item.get("year"))
                val = float(item.get("yield"))
                parsed_historical.append((y, val))
        except Exception as e:
            print(f"Avertissement : Échec du parsing des rendements historiques ({e})")

    oil_yield_estimate = None
    olive_trees_count = None
    calibration_factor = None
    calibration_confidence = None
    final_prediction_confidence = 0.5
    wofost_degraded = False
    ml_degraded = False

    # Si c'est de l'olive, on utilise notre modèle agronomique spécifique
    if crop_type == 'olive':
        # Paramètres par défaut
        num_trees = olive_num_trees if olive_num_trees is not None else 50
        age_cat = olive_age if olive_age in ['jeune', 'mature', 'vieux'] else 'mature'
        last_year_good = True if olive_last_year == 'oui' else False
        size_cat = olive_size if olive_size in ['petit', 'moyen', 'grand'] else 'moyen'
        variety = olive_variety if olive_variety in ['picholine', 'marocaine', 'autre'] else 'picholine'

        # Rendement de base par arbre selon l'âge (kg d'olives par arbre)
        base_yield_per_tree = {
            'jeune': 8.0,
            'mature': 40.0,
            'vieux': 25.0
        }[age_cat]

        # Correction taille
        size_factor = {
            'petit': 0.7,
            'moyen': 1.0,
            'grand': 1.3
        }[size_cat]

        # Alternance biennale
        cycle_factor = 0.6 if last_year_good else 1.4

        # Variété
        variety_factor = 1.1 if variety == 'picholine' else 1.0

        # Télédétection (NDVI) correction : stress hydrique/canopée
        ndvi_factor = min(1.0, ndvi_val / 0.7) if ndvi_val is not None else 1.0
        
        # Correction maladie détectée sur feuilles (ex: Peacock Spot)
        disease_factor = 1.0 - (disease_severity * 0.75)

        # Rendement final par arbre et total
        yield_per_tree = base_yield_per_tree * size_factor * cycle_factor * variety_factor * ndvi_factor * disease_factor
        final_yield = num_trees * yield_per_tree
        oil_yield_estimate = final_yield * 0.20
        olive_trees_count = num_trees

        wofost_yield = 0.0
        wofost_lai = 0.0
        conf_low = final_yield * 0.85
        conf_high = final_yield * 1.15
        final_prediction_confidence = 0.7
    else:
        # 5. Simulation de croissance WOFOST avec climat de la saison & Calibration (Étape 4)
        wofost_yield = 0.0
        wofost_lai = 0.0
        accumulated_temp = 0.0
        calibration_factor = None
        calibration_confidence = None
        wofost_degraded = False
        try:
            wofost_yield, wofost_lai, accumulated_temp, calibration_factor, calibration_confidence = run_wofost_simulation(
                lat=lat, 
                lon=lon, 
                crop_type=crop_type, 
                soil_ph=soil_ph, 
                clay_pct=soil_clay_pct, 
                sand_pct=soil_sand_pct,
                sowing_date_str=sowing_date,
                historical_yields=parsed_historical if parsed_historical else None
            )
            if wofost_yield == 0.0:
                wofost_degraded = True
        except Exception as e:
            print(f"Avertissement : Échec de la simulation WOFOST ({e})")
            wofost_degraded = True

        # 6. Fusion des données par apprentissage supervisé (Random Forest) (Étape 6)
        final_yield = wofost_yield
        conf_low = wofost_yield * 0.8
        conf_high = wofost_yield * 1.2
        final_prediction_confidence = 0.5
        ml_degraded = False
        try:
            final_yield, conf_low, conf_high, final_prediction_confidence = predict_final_yield(
                wofost_yield=wofost_yield,
                ndvi=ndvi_val,
                clay_pct=soil_clay_pct,
                sand_pct=soil_sand_pct,
                soil_ph=soil_ph,
                accumulated_temp=accumulated_temp,
                disease_severity=disease_severity,
                soil_moisture_pct=soil_moisture_pct,
                season_rainfall_mm=season_rainfall_mm,
                growth_window_avg_temp=growth_window_avg_temp,
                vegetative_stage=vegetative_stage,
                irrigation_level=irrigation_level,
                fertilization_level=fertilization_level,
                variety=variety or olive_variety
            )
        except Exception as e:
            print(f"Avertissement : Échec de la prédiction Random Forest ({e})")
            ml_degraded = True
            final_yield = wofost_yield
            conf_low = wofost_yield * 0.8
            conf_high = wofost_yield * 1.2
            final_prediction_confidence = 0.4

    # 7. Enregistrement selon le schéma de données
    record = ParcelRecord(
        parcel_id=parcel_id,
        farmer_id=farmer_id,
        lat=lat,
        lon=lon,
        timestamp=today,
        crop_type=crop_type,
        disease_class=disease_class,
        disease_confidence=disease_confidence,
        disease_severity=disease_severity,
        photo_quality_score=photo_quality_score,
        soil_ph=soil_ph,
        soil_clay_pct=soil_clay_pct,
        soil_sand_pct=soil_sand_pct,
        ndvi=ndvi_val,
        ndvi_date=ndvi_date_val,
        wofost_yield_kg_ha=wofost_yield,
        wofost_lai=wofost_lai,
        wofost_calibration_factor=calibration_factor,
        wofost_calibration_confidence=calibration_confidence,
        ndvi_signal_quality=ndvi_signal_quality,
        farmer_historical_yields=parsed_historical if parsed_historical else None,
        final_yield_estimate=final_yield,
        confidence_interval_low=conf_low,
        confidence_interval_high=conf_high,
        final_prediction_confidence=final_prediction_confidence,
        oil_yield_estimate=oil_yield_estimate,
        olive_trees_count=olive_trees_count
    )

    result = record.to_dict()
    result.update({
        'services': {
            'soil': {
                'source': soil_source,
                'degraded': soil_degraded
            },
            'ndvi': {
                'source': ndvi_source,
                'degraded': ndvi_degraded
            },
            'wofost': {
                'degraded': wofost_degraded
            },
            'ml': {
                'degraded': ml_degraded
            }
        },
        'process_stages': [
            {'name': 'Analyse de l\'image', 'status': 'ok'},
            {'name': 'Récupération données sol', 'status': 'ok' if not soil_degraded else 'degraded'},
            {'name': 'Lecture NDVI', 'status': 'ok' if not ndvi_degraded else 'degraded'},
            {'name': 'Simulation WOFOST', 'status': 'ok' if not wofost_degraded else 'degraded'},
            {'name': 'Fusion ML', 'status': 'ok' if not ml_degraded else 'degraded'},
            {'name': 'Génération du conseil', 'status': 'ok'}
        ],
        'fallback_notice': []
    })

    if soil_degraded:
        result['fallback_notice'].append('mode dégradé activé pour la donnée sol')
    if ndvi_degraded:
        result['fallback_notice'].append('mode dégradé activé pour le NDVI')
    if wofost_degraded:
        result['fallback_notice'].append('mode dégradé activé pour la simulation WOFOST')
    if ml_degraded:
        result['fallback_notice'].append('mode dégradé activé pour la fusion ML')

    if result['fallback_notice']:
        result['fallback_notice'].append('prédiction basée sur valeur par défaut / confiance plus faible')

    return result

from step5_llm.llm_service import ask_agronomist_interactive

class ChatRequest(BaseModel):
    question: str
    history: Optional[list] = []
    current_form: Optional[dict] = {}
    context: Optional[dict] = None

@app.post("/chat")
async def chat(req: ChatRequest):
    # Appelle l'agronome virtuel interactif (Gemini)
    res = ask_agronomist_interactive(req.question, req.history or [], req.current_form or {}, req.context)
    return res

@app.get("/", response_class=HTMLResponse)
async def read_item():
    html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=device-width, initial-scale=1.0">
    <title>TerroirAI - Gestion et Diagnostic de Parcelle</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Cairo:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f1f15 0%, #070b19 100%);
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --primary: #10b981;
            --primary-glow: rgba(16, 185, 129, 0.15);
            --accent: #f59e0b;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            font-family: 'Outfit', sans-serif;
        }

        body.rtl {
            font-family: 'Cairo', sans-serif;
            direction: rtl;
        }

        header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--glass-border);
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logo {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(to right, #10b981, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .logo span {
            color: var(--text-main);
            -webkit-text-fill-color: var(--text-main);
            font-size: 0.9rem;
            font-weight: 300;
            border: 1px solid var(--glass-border);
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            background: var(--glass-bg);
        }

        .lang-switch {
            display: flex;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.25rem;
            border-radius: 8px;
            border: 1px solid var(--glass-border);
        }

        .lang-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .lang-btn.active {
            background: var(--primary);
            color: #fff;
        }

        .container {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 2rem;
            padding: 2rem;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
        }

        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        h2 {
            font-size: 1.4rem;
            font-weight: 600;
            border-left: 4px solid var(--primary);
            padding-left: 0.75rem;
            color: #fff;
        }

        body.rtl h2 {
            border-left: none;
            border-right: 4px solid var(--primary);
            padding-left: 0;
            padding-right: 0.75rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        input[type="text"], input[type="number"], select {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            padding: 0.75rem;
            color: #fff;
            outline: none;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        select option {
            background-color: #0f1f15;
            color: #f3f4f6;
        }

        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .wizard-step {
            display: none;
            gap: 1rem;
            flex-direction: column;
        }

        .wizard-step.active {
            display: flex;
        }

        .step-indicator {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .step-pill {
            padding: 0.85rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            text-align: center;
            font-size: 0.85rem;
            font-weight: 700;
            cursor: pointer;
        }

        .step-pill.active {
            background: rgba(16, 185, 129, 0.18);
            border-color: rgba(16, 185, 129, 0.4);
            color: #fff;
        }

        .form-footer {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .stage-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 0.9rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.95rem;
        }

        .stage-status-ok { color: #10b981; font-weight: 700; }
        .stage-status-degraded { color: #f59e0b; font-weight: 700; }
        .fallback-banner {
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.25);
            border-radius: 12px;
            padding: 1rem;
            color: #f8b400;
        }

        .help-text {
            font-size: 0.9rem;
            color: var(--text-muted);
            line-height: 1.4;
        }

        .big-action {
            font-size: 1rem;
            padding: 1rem 1.1rem;
        }

        .wizard-progress {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .wizard-progress span {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .hidden {
            display: none !important;
        }

        .status-banner {
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: #f3f4f6;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .status-banner.degraded {
            background: rgba(245, 158, 11, 0.12);
            border-color: rgba(245, 158, 11, 0.25);
            color: #f8b400;
        }

        .status-banner strong {
            color: inherit;
        }

        .pill-tag {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: #fff;
            font-size: 0.8rem;
            gap: 0.3rem;
        }

        .pill-tag.degraded { background: rgba(245, 158, 11, 0.18); }
        .pill-tag.ok { background: rgba(16, 185, 129, 0.18); }

        .big-button {
            width: 100%;
            padding: 1rem 1.1rem;
            font-size: 1rem;
            border-radius: 12px;
        }

        .step-actions {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .step-actions .btn {
            flex: 1;
        }
        .step-actions .btn.secondary {
            background: rgba(255,255,255,0.05);
            color: #fff;
            border: 1px solid var(--glass-border);
        }

        .wizard-hint {
            padding: 0.85rem 1rem;
            border-radius: 12px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            color: var(--text-muted);
            font-size: 0.94rem;
            line-height: 1.5;
        }

        .wizard-hint strong { color: #fff; }

        .upload-preview {
            width: 100%;
            max-height: 180px;
            object-fit: contain;
            border-radius: 12px;
            margin-top: 1rem;
            display: none;
        }

        .button-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .button-row .btn {
            flex: 1;
            min-width: 140px;
        }
        .button-row .btn.primary {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }
        .button-row .btn.secondary {
            background: rgba(255,255,255,0.05);
            color: #fff;
            border: 1px solid var(--glass-border);
        }

        .batch-field {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .small-text {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .field-group {
            display:flex;
            flex-direction:column;
            gap:0.5rem;
        }

        .field-group label {
            font-size:0.92rem;
        }

        .field-group input,
        .field-group select {
            min-height: 48px;
        }

        .small-badge {
            display:inline-flex;
            align-items:center;
            padding:0.25rem 0.5rem;
            border-radius:999px;
            font-size:0.75rem;
            background: rgba(255,255,255,0.08);
        }

        .small-badge.ok { color:#10b981; }
        .small-badge.degraded { color:#f59e0b; }

        .step-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .step-subtitle {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }
        .file-upload {
            border: 2px dashed var(--glass-border);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            background: rgba(255, 255, 255, 0.01);
        }

        .file-upload:hover {
            border-color: var(--primary);
            background: rgba(16, 185, 129, 0.02);
        }

        .file-upload input {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }

        .file-upload-preview {
            max-height: 150px;
            margin-top: 1rem;
            border-radius: 8px;
            display: none;
        }

        .btn {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 1rem;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
            text-align: center;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }

        .btn:disabled {
            background: var(--text-muted);
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        /* Results / Right side */
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            height: 100%;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 1rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .metric-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #fff;
            margin-top: 0.25rem;
        }

        .metric-label {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .yield-banner {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .yield-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .yield-val {
            font-size: 2.2rem;
            font-weight: 700;
            color: #10b981;
        }

        .chat-section {
            display: flex;
            flex-direction: column;
            border-top: 1px solid var(--glass-border);
            padding-top: 1.5rem;
            flex: 1;
            min-height: 350px;
        }

        .chat-box {
            flex: 1;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 1rem;
            overflow-y: auto;
            max-height: 350px;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .chat-message {
            max-width: 80%;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            font-size: 0.95rem;
            line-height: 1.4;
        }

        .message-bot {
            background: rgba(255, 255, 255, 0.06);
            align-self: flex-start;
            border-bottom-left-radius: 2px;
            color: #fff;
        }

        .message-user {
            background: var(--primary);
            align-self: flex-end;
            border-bottom-right-radius: 2px;
            color: #fff;
        }

        body.rtl .message-bot {
            align-self: flex-start;
            border-bottom-left-radius: 2px;
        }

        body.rtl .message-user {
            align-self: flex-end;
            border-bottom-right-radius: 2px;
        }

        .chat-input-container {
            display: flex;
            gap: 0.5rem;
        }

        .chat-input {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            padding: 0.75rem;
            color: #fff;
            outline: none;
        }

        .send-btn {
            background: var(--primary);
            border: none;
            border-radius: 8px;
            color: #fff;
            padding: 0 1.25rem;
            cursor: pointer;
            font-weight: 600;
        }

        /* Demo badge helper */
        .helper-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--glass-border);
            color: var(--text-muted);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }

        .helper-btn:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.1);
        }

        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .status-healthy { background: rgba(16, 185, 129, 0.2); color: #10b981; }

        /* Map styling */
        #map {
            width: 100%;
            height: 250px;
            border-radius: 12px;
            border: 1px solid var(--glass-border);
            z-index: 1;
        }

        /* Custom leaflet styles to blend with dark mode */
        .leaflet-container {
            background: #090f0b !important;
        }
        .leaflet-tile {
            filter: invert(100%) hue-rotate(180deg) brightness(95%) contrast(90%);
        }
        .leaflet-bar a {
            background-color: rgba(20, 20, 20, 0.8) !important;
            color: #fff !important;
            border-bottom: 1px solid var(--glass-border) !important;
        }
        .leaflet-bar a:hover {
            background-color: var(--primary) !important;
        }
        .leaflet-popup-content-wrapper {
            background: #0f1f15 !important;
            color: #fff !important;
            border: 1px solid var(--glass-border);
        }
        .leaflet-popup-tip {
            background: #0f1f15 !important;
        }
    </style>
</head>
<body class="rtl">
    <header>
        <div class="logo">TerroirAI <span data-i18n="logoSub">Diagnostic & Simulation</span></div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div class="lang-switch">
                <button onclick="setLanguage('fr')" id="btn-fr" class="lang-btn">FR</button>
                <button onclick="setLanguage('ar')" id="btn-ar" class="lang-btn active">عربي</button>
            </div>
            <span class="status-badge status-healthy">API Live</span>
        </div>
    </header>

    <div class="container">
        <!-- Configuration Form -->
        <div class="card">
            <h2 data-i18n="parcelData">Données de la Parcelle</h2>
            <div class="step-indicator">
                <div class="step-pill active" id="step-pill-1" onclick="goToStep(1)" data-i18n="stepPill1">1. Culture</div>
                <div class="step-pill" id="step-pill-2" onclick="goToStep(2)" data-i18n="stepPill2">2. Photo</div>
                <div class="step-pill" id="step-pill-3" onclick="goToStep(3)" data-i18n="stepPill3">3. Localisation</div>
                <div class="step-pill" id="step-pill-4" onclick="goToStep(4)" data-i18n="stepPill4">4. Semis</div>
                <div class="step-pill" id="step-pill-5" onclick="goToStep(5)" data-i18n="stepPill5">5. Prédiction</div>
            </div>
            <form id="predictForm" enctype="multipart/form-data">
                <div class="wizard-step active" id="wizard-step-1">
                    <div class="step-title" data-i18n="step1Title">Choisissez votre culture</div>
                    <div class="step-subtitle" data-i18n="step1Subtitle">Sélectionnez une culture simple pour démarrer.</div>
                    <div class="field-group">
                        <label for="crop_type" data-i18n="cropType">Type de culture</label>
                        <select id="crop_type" name="crop_type" required>
                            <option value="wheat" data-i18n="cropWheat">Blé</option>
                            <option value="barley" data-i18n="cropBarley">Orge</option>
                            <option value="olive" data-i18n="cropOlive">Olivier</option>
                            <option value="potato" data-i18n="cropPotato">Pomme de terre</option>
                            <option value="tomato" data-i18n="cropTomato">Tomate</option>
                        </select>
                    </div>
                    <div class="wizard-hint" data-i18n="cropHint">Choisissez la culture que vous avez semée dans ce champ. Cela permet au système de s'adapter aux bons modèles.</div>
                    <div class="button-row">
                        <button type="button" class="btn primary" onclick="goToStep(2)" data-i18n="nextBtn">Suivant →</button>
                    </div>
                </div>

                <div class="wizard-step" id="wizard-step-2">
                    <div class="step-title" data-i18n="step2Title">Ajoutez une photo</div>
                    <div class="step-subtitle" data-i18n="step2Subtitle">Une seule feuille nette suffit.</div>
                    <div class="wizard-hint" data-i18n="photoHint">Plus la photo est proche et nette, meilleur sera le diagnostic de maladie.</div>
                    <div class="file-upload">
                        <p style="color: var(--text-muted);" data-i18n="uploadPrompt">Cliquez ou déposez une image ici</p>
                        <input type="file" id="photo" name="photo" accept="image/*" required onchange="previewFile()">
                        <img id="filePreview" class="upload-preview" src="" alt="Aperçu">
                    </div>
                    <div class="button-row">
                        <button type="button" class="btn secondary" onclick="goToStep(1)" data-i18n="prevBtn">← Précédent</button>
                        <button type="button" class="btn primary" onclick="goToStep(3)" data-i18n="nextBtn">Suivant →</button>
                    </div>
                </div>

                <div class="wizard-step" id="wizard-step-3">
                    <div class="step-title" data-i18n="step3Title">Indiquez la localisation</div>
                    <div class="step-subtitle" data-i18n="step3Subtitle">Choisissez votre champ sur la carte ou entrez un lieu.</div>
                    <div class="field-group">
                        <label data-i18n="locationSearch">Rechercher une localisation (Commune, Ville...) :</label>
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                            <input type="text" id="search-input" data-i18n-placeholder="locationSearchPlaceholder" placeholder="مثال: بني ملال" style="flex: 1; min-width: 140px;">
                            <button type="button" class="btn secondary" onclick="geocodeAddress()" style="padding: 0 1.25rem; min-width: 120px;"><span data-i18n="searchBtn">🔍 Rechercher</span></button>
                        </div>
                    </div>
                    <div class="field-group">
                        <label data-i18n="mapSelect">Sélectionnez votre parcelle sur la carte :</label>
                        <div id="map" style="height: 230px; border-radius: 14px; overflow: hidden;"></div>
                    </div>
                    <div class="grid-2">
                        <div class="field-group">
                            <label for="lat" data-i18n="latitude">Latitude</label>
                            <input type="number" id="lat" name="lat" step="any" value="32.32" required>
                        </div>
                        <div class="field-group">
                            <label for="lon" data-i18n="longitude">Longitude</label>
                            <input type="number" id="lon" name="lon" step="any" value="-6.38" required>
                        </div>
                    </div>
                    <button type="button" class="btn secondary" onclick="setMoroccoFarm()" data-i18n="loadCoords">Charger coordonnées test (Béni Mellal)</button>
                    <div class="button-row">
                        <button type="button" class="btn secondary" onclick="goToStep(2)" data-i18n="prevBtn">← Précédent</button>
                        <button type="button" class="btn primary" onclick="goToStep(4)" data-i18n="nextBtn">Suivant →</button>
                    </div>
                </div>

                <div class="wizard-step" id="wizard-step-4">
                    <div class="step-title" data-i18n="step4Title">Renseignez la date de semis</div>
                    <div class="step-subtitle" data-i18n="step4Subtitle">La date de semis aide à calculer la croissance et la récolte.</div>
                    <div class="field-group">
                        <label for="sowing_date" data-i18n="sowingDate">Date de semis</label>
                        <input type="date" id="sowing_date" name="sowing_date" required>
                    </div>
                    <div class="wizard-hint" data-i18n="sowingHint">Si vous ne la connaissez pas, choisissez la date la plus proche du jour où vous avez planté.</div>
                    <div class="button-row">
                        <button type="button" class="btn secondary" onclick="goToStep(3)" data-i18n="prevBtn">← Précédent</button>
                        <button type="button" class="btn primary" onclick="goToStep(5)" data-i18n="nextBtn">Suivant →</button>
                    </div>
                </div>

                <div class="wizard-step" id="wizard-step-5">
                    <div class="step-title" data-i18n="step5Title">Lancer la prédiction</div>
                    <div class="step-subtitle" data-i18n="step5Subtitle">Vérifiez vos informations et exécutez l'analyse.</div>
                    <div class="wizard-hint" data-i18n="readyHint">Le système va analyser l'image, récupérer les données sol, lire le NDVI, lancer la simulation et fusionner le modèle.</div>
                    <div class="field-group" style="margin-top: 1rem;">
                        <label data-i18n="historicalYieldsTitle" style="font-weight: bold;">Rendements Historiques (optionnel)</label>
                        <div class="grid-2">
                            <input type="number" id="hist_year" data-i18n-placeholder="yearPlaceholder" placeholder="Année (ex: 2024)" min="2000" max="2030">
                            <input type="number" id="hist_yield" data-i18n-placeholder="yieldPlaceholder" placeholder="Rendement (kg/ha)" min="0">
                        </div>
                        <button type="button" class="btn secondary" onclick="addHistoricalRecord()" style="margin-top: 0.5rem;" data-i18n="addHistBtn">Ajouter</button>
                        <div id="histRecordsList" style="margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.25rem;"></div>
                        <input type="hidden" id="historical_yields" name="historical_yields">
                    </div>
                    <div class="wizard-progress" id="progressPreview"></div>
                    <div class="button-row">
                        <button type="button" class="btn secondary" onclick="goToStep(4)" data-i18n="prevBtn">← Précédent</button>
                        <button type="submit" class="btn primary big-button" id="submitBtn" data-i18n="analyzeBtn">Lancer l'Analyse TerroirAI</button>
                    </div>
                </div>

                <div class="form-footer">
                    <div class="wizard-hint" id="stepHelp">Suivez les étapes une par une. Vous pouvez aussi poser vos questions au chat pour remplir automatiquement le formulaire.</div>
                </div>
            </form>
        </div>

        <!-- Right Column / Dashboards -->
        <div style="display: flex; flex-direction: column; gap: 2rem; width: 100%;">
            <!-- Chat Card (Always Visible) -->
            <div class="card" style="justify-content: flex-start; min-height: 400px; display: flex; flex-direction: column;">
                <h2 data-i18n="chatTitle">Agronome Virtuel Gemini Chat</h2>
                <div class="chat-box" id="chatBox" style="flex: 1; min-height: 250px; max-height: 350px;">
                    <div class="chat-message message-bot" id="welcomeMsg" data-i18n="chatWelcome">
                        Salam! Je suis votre agronome virtuel. Pour commencer, dites-moi quelle culture vous faites (blé, orge, olive, pomme de terre, tomate) ou posez-moi vos questions !
                    </div>
                </div>
                <div class="chat-input-container" style="margin-top: 1rem;">
                    <input type="text" class="chat-input" id="chatInput" placeholder="Posez une question..." onkeydown="if(event.key === 'Enter') sendChatMessage()">
                    <button class="send-btn" onclick="sendChatMessage()" data-i18n="chatSendBtn">Envoyer</button>
                </div>
            </div>

            <!-- Results Dashboard Card -->
            <div class="card" style="justify-content: flex-start;">
                <h2 data-i18n="diagnosticsTitle">Diagnostic & Recommandations Agronomiques</h2>
                
                <div id="noResults" style="text-align: center; color: var(--text-muted); padding: 3rem 0;">
                    <p data-i18n="noResults">Aucune analyse en cours. Veuillez remplir le formulaire à gauche ou discuter avec l'agronome pour lancer l'analyse.</p>
                </div>

                <div id="resultsDashboard" style="display: none;" class="results-container">
                    <!-- Yield result -->
                    <div class="yield-banner">
                        <div>
                            <div class="yield-title" data-i18n="yieldTitle">Rendement Final Estimé</div>
                            <div class="yield-val" id="resYield">- kg/ha</div>
                        </div>
                        <div style="text-align: right;">
                            <div class="yield-title" data-i18n="confidenceInterval">Intervalle de confiance</div>
                            <div style="font-weight: 600; color: #fff;" id="resInterval">-</div>
                        </div>
                    </div>

                    <!-- Metrics Grid -->
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label" data-i18n="diseaseTitle">Diagnostic Maladie</div>
                            <div class="metric-value" id="resDisease" style="font-size: 1.1rem; color: #f59e0b;">-</div>
                            <div class="metric-label" id="resDiseaseConf">-</div>
                            <div class="metric-label" id="resDiseaseSeverity" style="color: #ef4444; font-weight: bold; margin-top: 0.25rem;">-</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label" data-i18n="ndviTitle">Indice NDVI (Végétation)</div>
                            <div class="metric-value" id="resNDVI">-</div>
                            <div class="metric-label" id="resNDVIDate">-</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label" data-i18n="soilTitle">Carac. du Sol</div>
                            <div class="metric-value" id="resSoil" style="font-size: 0.95rem; text-align: left;">-</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label" data-i18n="laiTitle">WOFOST (LAI)</div>
                            <div class="metric-value" id="resLAI">-</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script>
        let currentParcelContext = null;
        let map, marker;
        let currentLang = 'ar';
        let historicalRecords = [];
        let chatHistory = [];

        const translations = {
            fr: {
                logoSub: "Diagnostic & Simulation",
                parcelData: "Données de la Parcelle",
                farmerId: "Identifiant de l'agriculteur",
                locationSearch: "Rechercher une localisation (Commune, Ville...) :",
                locationSearchPlaceholder: "Ex: Beni Mellal",
                mapSelect: "Sélectionnez votre parcelle sur la carte :",
                latitude: "Latitude",
                longitude: "Longitude",
                loadCoords: "Charger coordonnées test (Béni Mellal)",
                cropType: "Type de culture",
                cropWheat: "Blé",
                cropBarley: "Orge",
                cropOlive: "Olivier",
                cropPotato: "Pomme de terre",
                cropTomato: "Tomate",
                sowingDate: "Date de semis",
                historicalYieldsTitle: "Rendements Historiques (Calibration)",
                addHistBtn: "Ajouter",
                photoLabel: "Photo de la culture",
                cropWarningText: "💡 <strong>Important :</strong> Pour toutes les cultures (blé, orge, olive, pomme de terre, tomate), veuillez prendre une photo nette de très près (gros plan) d'une seule feuille malade pour obtenir un diagnostic de maladie précis.",
                uploadPrompt: "Cliquez ou déposez une image ici",
                analyzeBtn: "Lancer l'Analyse TerroirAI",
                analyzing: "Traitement et simulation en cours...",
                diagnosticsTitle: "Diagnostic & Recommandations Agronomiques",
                noResults: "Aucune analyse en cours. Veuillez remplir le formulaire à gauche et cliquer sur 'Lancer l'Analyse'.",
                yieldTitle: "Rendement Final Estimé",
                confidenceInterval: "Intervalle de confiance",
                diseaseTitle: "Diagnostic Maladie",
                ndviTitle: "Indice NDVI (Végétation)",
                soilTitle: "Carac. du Sol",
                laiTitle: "WOFOST (LAI)",
                chatTitle: "Agronome Virtuel Gemini Chat",
                chatInputPlaceholder: "Posez une question...",
                chatSendBtn: "Envoyer",
                chatWelcome: "Salam! Je suis votre agronome virtuel. J'ai analysé les données de votre parcelle. Posez-moi vos questions sur l'irrigation, la fertilisation ou la santé de votre culture ! (Darija/Français)",
                // Step titles & subtitles
                stepPill1: "1. Culture",
                stepPill2: "2. Photo",
                stepPill3: "3. Localisation",
                stepPill4: "4. Semis",
                stepPill5: "5. Prédiction",
                step1Title: "Choisissez votre culture",
                step1Subtitle: "Sélectionnez une culture simple pour démarrer.",
                step2Title: "Ajoutez une photo",
                step2Subtitle: "Une seule feuille nette suffit.",
                step3Title: "Indiquez la localisation",
                step3Subtitle: "Choisissez votre champ sur la carte ou entrez un lieu.",
                step4Title: "Renseignez la date de semis",
                step4Subtitle: "La date de semis aide à calculer la croissance et la récolte.",
                step5Title: "Lancer la prédiction",
                step5Subtitle: "Vérifiez vos informations et exécutez l'analyse.",
                // Navigation buttons
                nextBtn: "Suivant →",
                prevBtn: "← Précédent",
                backBtn: "Retour",
                // Hints
                cropHint: "Choisissez la culture que vous avez semée dans ce champ. Cela permet au système de s'adapter aux bons modèles.",
                photoHint: "Plus la photo est proche et nette, meilleur sera le diagnostic de maladie.",
                sowingHint: "Si vous ne la connaissez pas, choisissez la date la plus proche du jour où vous avez planté.",
                readyHint: "Le système va analyser l'image, récupérer les données sol, lire le NDVI, lancer la simulation et fusionner le modèle.",
                // Placeholders
                yearPlaceholder: "Année (ex: 2024)",
                yieldPlaceholder: "Rendement (kg/ha)",
                searchBtn: "🔍 Rechercher",
                // Help steps
                helpStep1: "Choisissez la culture d’abord, puis ajoutez la photo.",
                helpStep2: "Prenez ou choisissez une photo nette d’une seule feuille.",
                helpStep3: "Sélectionnez le champ ou entrez des coordonnées précises.",
                helpStep4: "Saisissez la date de semis pour guider le modèle.",
                helpStep5: "Révisez vos informations et validez la prédiction."
            },
            ar: {
                logoSub: "التشخيص والمحاكاة",
                parcelData: "بيانات الحقل",
                farmerId: "معرّف الفلاح",
                locationSearch: "البحث عن موقع (جماعة، مدينة...) :",
                locationSearchPlaceholder: "مثال: بني ملال",
                mapSelect: "حدد موقع حقلك على الخريطة :",
                latitude: "خط العرض (Latitude)",
                longitude: "خط الطول (Longitude)",
                loadCoords: "تحميل إحداثيات تجريبية (بني ملال)",
                cropType: "نوع المحصول",
                cropWheat: "القمح",
                cropBarley: "الشعير",
                cropOlive: "الزيتون",
                cropPotato: "البطاطس",
                cropTomato: "الطماطم",
                sowingDate: "تاريخ البذر",
                historicalYieldsTitle: "المردودية التاريخية (المعايرة)",
                addHistBtn: "إضافة",
                photoLabel: "صورة المحصول",
                cropWarningText: "💡 <strong>هام جداً:</strong> لجميع المحاصيل (القمح، الشعير، الزيتون، البطاطس، الطماطم)، يرجى التقاط صورة واضحة وعن قرب (زوم) لورقة نبات واحدة مصابة لتشخيص دقيق.",
                uploadPrompt: "اضغط هنا أو اسحب الصورة هنا",
                analyzeBtn: "تشغيل تحليل TerroirAI",
                analyzing: "جاري التحليل والمحاكاة...",
                diagnosticsTitle: "التشخيص والتوصيات الزراعية",
                noResults: "لا توجد نتائج حاليا. يرجى تعبئة الاستمارة على اليسار والضغط على 'تشغيل تحليل'.",
                yieldTitle: "المردود النهائي المتوقع",
                confidenceInterval: "مجال الثقة",
                diseaseTitle: "تشخيص المرض",
                ndviTitle: "مؤشر الغطاء النباتي (NDVI)",
                soilTitle: "خصائص التربة",
                laiTitle: "مؤشر المساحة الورقية (LAI)",
                chatTitle: "مستشارك الفلاحي Gemini Chat",
                chatInputPlaceholder: "اطرح سؤالك هنا...",
                chatSendBtn: "إرسال",
                chatWelcome: "السلام عليكم! أنا مستشارك الفلاحي الافتراضي. تفضل بطرح أسئلتك حول السقي، التسميد أو صحة المحصول بالدارجة أو العربية.",
                // Step titles & subtitles
                stepPill1: "1. المحصول",
                stepPill2: "2. الصورة",
                stepPill3: "3. الموقع",
                stepPill4: "4. البذر",
                stepPill5: "5. التوقع",
                step1Title: "اختر محصولك",
                step1Subtitle: "اختر محصولاً للبدء.",
                step2Title: "أضف صورة",
                step2Subtitle: "ورقة واحدة واضحة تكفي.",
                step3Title: "حدد الموقع",
                step3Subtitle: "اختر حقلک على الخريطة أو أدخل موقعاً.",
                step4Title: "أدخل تاريخ البذر",
                step4Subtitle: "تاريخ البذر يساعد في حساب النمو والحصاد.",
                step5Title: "شغّل التوقع",
                step5Subtitle: "تحقق من معلوماتك ثم شغّل التحليل.",
                // Navigation buttons
                nextBtn: "التالي ←",
                prevBtn: "→ السابق",
                backBtn: "العودة",
                // Hints
                cropHint: "اختر المحصول الذي زرعته في هذا الحقل. هذا يساعد النظام على استخدام النماذج المناسبة.",
                photoHint: "كلما كانت الصورة قريبة وواضحة، كان تشخيص المرض أفضل.",
                sowingHint: "إذا كنت لا تعرفه، اختر التاريخ الأقرب لزراعتك.",
                readyHint: "سيقوم النظام بتحليل الصورة وجمع بيانات التربة وقراءة NDVI وتشغيل المحاكاة ودمج النماذج.",
                // Placeholders
                yearPlaceholder: "السنة (مثال: 2024)",
                yieldPlaceholder: "المردود (كجم/هكتار)",
                searchBtn: "🔍 بحث",
                // Help steps
                helpStep1: "اختر المحصول أولاً، ثم تابع تحميل الصورة.",
                helpStep2: "التقط أو اختر صورة واضحة لورقة واحدة.",
                helpStep3: "حدد موقع الحقل أو أضف إحداثيات دقيقة.",
                helpStep4: "أدخل تاريخ البذر لمساعدة النموذج.",
                helpStep5: "قم بمراجعة البيانات وتأكيد التوقعات."
            }
        };

        let activeStep = 1;

        function updateStepHelp() {
            const helpText = document.getElementById('stepHelp');
            if (helpText) {
                const key = 'helpStep' + activeStep;
                if (translations[currentLang] && translations[currentLang][key]) {
                    helpText.innerText = translations[currentLang][key];
                }
            }
        }

        function setLanguage(lang) {
            currentLang = lang;
            document.querySelectorAll('.lang-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('btn-' + lang).classList.add('active');

            if (lang === 'ar') {
                document.body.classList.add('rtl');
            } else {
                document.body.classList.remove('rtl');
            }

            // Mettre à jour tous les textes i18n
            document.querySelectorAll('[data-i18n]').forEach(el => {
                const key = el.getAttribute('data-i18n');
                if (translations[lang][key]) {
                    el.innerHTML = translations[lang][key];
                }
            });

            // Mettre à jour les placeholders i18n
            document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
                const key = el.getAttribute('data-i18n-placeholder');
                if (translations[lang][key]) {
                    el.placeholder = translations[lang][key];
                }
            });

            // Input placeholders spécifiques
            const chatInput = document.getElementById('chatInput');
            if (chatInput) chatInput.placeholder = translations[lang]['chatInputPlaceholder'];

            const searchInput = document.getElementById('search-input');
            if (searchInput) searchInput.placeholder = translations[lang]['locationSearchPlaceholder'];

            // Mettre à jour le texte d'aide du step courant
            updateStepHelp();
        }

        // Gestion des rendements historiques
        function addHistoricalRecord() {
            const yearInput = document.getElementById('hist_year');
            const yieldInput = document.getElementById('hist_yield');
            const year = parseInt(yearInput.value);
            const yieldVal = parseFloat(yieldInput.value);
            
            if (isNaN(year) || isNaN(yieldVal)) {
                alert(currentLang === 'ar' ? "يرجى إدخال سنة ومردودية صالحة" : "Veuillez entrer une année et un rendement valides");
                return;
            }
            
            if (historicalRecords.some(r => r.year === year)) {
                alert(currentLang === 'ar' ? "السنة مضافة بالفعل" : "Cette année est déjà ajoutée");
                return;
            }
            
            historicalRecords.push({ year, yield: yieldVal });
            yearInput.value = "";
            yieldInput.value = "";
            
            renderHistoricalRecords();
        }
        
        function removeHistoricalRecord(index) {
            historicalRecords.splice(index, 1);
            renderHistoricalRecords();
        }
        
        function renderHistoricalRecords() {
            const listDiv = document.getElementById('histRecordsList');
            listDiv.innerHTML = "";
            
            historicalRecords.forEach((rec, idx) => {
                const item = document.createElement('div');
                item.style.display = "flex";
                item.style.justifyContent = "space-between";
                item.style.alignItems = "center";
                item.style.background = "rgba(255, 255, 255, 0.05)";
                item.style.padding = "0.4rem 0.6rem";
                item.style.borderRadius = "6px";
                item.style.fontSize = "0.9rem";
                
                item.innerHTML = `
                    <span>${rec.year} : <b>${rec.yield} kg/ha</b></span>
                    <button type="button" onclick="removeHistoricalRecord(${idx})" style="background: transparent; border: none; color: #ef4444; cursor: pointer; font-weight: bold; padding: 0 5px;">✕</button>
                `;
                listDiv.appendChild(item);
            });
            
            document.getElementById('historical_yields').value = JSON.stringify(historicalRecords);
        }

        // Geocode Nomimatin OSM
        async function geocodeAddress() {
            const query = document.getElementById('search-input').value.trim();
            if (!query) return;
            try {
                const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`);
                const data = await res.json();
                if (data && data.length > 0) {
                    const lat = parseFloat(data[0].lat);
                    const lon = parseFloat(data[0].lon);
                    updateCoordsInputs(lat, lon);
                    marker.setLatLng([lat, lon]);
                    map.setView([lat, lon], 12);
                } else {
                    alert(currentLang === 'ar' ? "لم يتم العثور على الموقع" : "Lieu non trouvé");
                }
            } catch (err) {
                alert("Error / خطأ: " + err.message);
            }
        }

        // Initialize Map
        window.addEventListener('DOMContentLoaded', () => {
            // Appliquer la langue arabe par défaut
            setLanguage('ar');

            // Fix Leaflet marker icon asset 404 path issues when loaded from CDN
            delete L.Icon.Default.prototype._getIconUrl;
            L.Icon.Default.mergeOptions({
                iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
                iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
                shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
            });

            // Centré sur le Maroc (Béni Mellal / Tadla par défaut)
            map = L.map('map').setView([32.32, -6.38], 10);
            
            // Fond de carte style satellite ou propre
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19,
                attribution: '© OpenStreetMap contributors'
            }).addTo(map);

            // Ajouter le marqueur initial
            marker = L.marker([32.32, -6.38], { draggable: true }).addTo(map);

            // Événement déplacement du marqueur
            marker.on('dragend', function (e) {
                const latlng = marker.getLatLng();
                updateCoordsInputs(latlng.lat, latlng.lng);
            });

            // Événement clic sur la carte
            map.on('click', function (e) {
                marker.setLatLng(e.latlng);
                updateCoordsInputs(e.latlng.lat, e.latlng.lng);
            });

            // Gérer le changement de culture
            const cropSelect = document.getElementById('crop_type');
            cropSelect.addEventListener('change', handleCropChange);
            handleCropChange();
            updateWizardProgress();
        });

        function goToStep(step) {
            activeStep = step;
            for (let i = 1; i <= 5; i++) {
                document.getElementById('wizard-step-' + i).classList.toggle('active', i === step);
                document.getElementById('step-pill-' + i).classList.toggle('active', i === step);
            }
            updateStepHelp();

            // Fix Leaflet tile loading bug when container transitions from display: none to flex
            if (step === 3 && map) {
                setTimeout(() => {
                    map.invalidateSize();
                }, 100);
            }
        }

        function updateWizardProgress() {
            const cropSelected = document.getElementById('crop_type').value;
            const hasPhoto = document.getElementById('photo').files.length > 0;
            const sowingDate = document.getElementById('sowing_date').value;
            const lat = document.getElementById('lat').value;
            const lon = document.getElementById('lon').value;
            const progress = document.getElementById('progressPreview');
            if (!progress) return;

            const steps = [
                { label: currentLang === 'ar' ? 'المحصول المختار' : 'Culture sélectionnée', done: !!cropSelected },
                { label: currentLang === 'ar' ? 'الصورة المرفقة' : 'Photo de la feuille', done: hasPhoto },
                { label: currentLang === 'ar' ? 'الموقع المحدد' : 'Localisation définie', done: !!lat && !!lon },
                { label: currentLang === 'ar' ? 'تاريخ البذر' : 'Date de semis', done: !!sowingDate },
                { label: currentLang === 'ar' ? 'جاهز للتحليل' : 'Prêt à analyser', done: !!cropSelected && hasPhoto && !!sowingDate && !!lat && !!lon }
            ];
            progress.innerHTML = steps.map(step => `
                <div class="stage-card">
                    <span>${step.label}</span>
                    <span class="small-badge ${step.done ? 'ok' : 'degraded'}">${step.done ? (currentLang === 'ar' ? 'تم' : 'OK') : (currentLang === 'ar' ? 'مطلوب' : 'Manquant')}</span>
                </div>
            `).join('');
        }

        function handleCropChange() {
            updateWizardProgress();
        }

        function updateCoordsInputs(lat, lon) {
            document.getElementById('lat').value = lat.toFixed(6);
            document.getElementById('lon').value = lon.toFixed(6);
        }

        function setMoroccoFarm() {
            const lat = 32.32;
            const lon = -6.38;
            updateCoordsInputs(lat, lon);
            marker.setLatLng([lat, lon]);
            map.setView([lat, lon], 12);
        }

        function previewFile() {
            const preview = document.getElementById('filePreview');
            const file = document.getElementById('photo').files[0];
            const reader = new FileReader();

            reader.addEventListener("load", function () {
                preview.src = reader.result;
                preview.style.display = "block";
                updateWizardProgress();
            }, false);

            if (file) {
                reader.readAsDataURL(file);
            }
        }

        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const submitBtn = document.getElementById('submitBtn');
            submitBtn.disabled = true;
            submitBtn.innerText = translations[currentLang]['analyzing'];

            const formData = new FormData(this);
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Erreur HTTP ' + response.status);
                }

                const data = await response.json();
                displayResults(data);
                renderProcessStages(data.process_stages || []);
                renderFallbackNotice(data.fallback_notice || []);
            } catch (err) {
                alert(currentLang === 'ar' ? 'حدث خطأ: ' + err.message : 'Erreur : ' + err.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = translations[currentLang]['analyzeBtn'];
            }
        });

        function renderProcessStages(stages) {
            const progress = document.getElementById('progressPreview');
            if (!progress) return;
            if (!Array.isArray(stages) || stages.length === 0) return;
            progress.innerHTML = stages.map(stage => `
                <div class="stage-card">
                    <span>${stage.name}</span>
                    <span class="small-badge ${stage.status === 'ok' ? 'ok' : 'degraded'}">${stage.status === 'ok' ? (currentLang === 'ar' ? 'تم' : 'OK') : (currentLang === 'ar' ? 'متدهور' : 'Dégradé')}</span>
                </div>
            `).join('');
        }

        function renderFallbackNotice(noticeLines) {
            let noticeArea = document.getElementById('fallbackArea');
            if (!noticeArea) {
                const resultsCard = document.querySelector('.results-container');
                if (!resultsCard) return;
                noticeArea = document.createElement('div');
                noticeArea.id = 'fallbackArea';
                noticeArea.className = 'fallback-banner hidden';
                resultsCard.parentNode.insertBefore(noticeArea, resultsCard);
            }
            if (!Array.isArray(noticeLines) || noticeLines.length === 0) {
                noticeArea.classList.add('hidden');
                return;
            }
            noticeArea.innerHTML = noticeLines.map(line => `<div>• ${line}</div>`).join('');
            noticeArea.classList.remove('hidden');
        }

        function displayResults(data) {
            document.getElementById('noResults').style.display = 'none';
            document.getElementById('resultsDashboard').style.display = 'flex';

            // Rendement
            if (data.final_yield_estimate) {
                if (data.crop_type === 'olive') {
                    const olivesVal = Math.round(data.final_yield_estimate);
                    const oilVal = Math.round(data.oil_yield_estimate || 0);
                    document.getElementById('resYield').innerText = olivesVal + " kg (Olives)";
                    document.getElementById('resInterval').innerHTML = `
                        ${Math.round(data.confidence_interval_low)} - ${Math.round(data.confidence_interval_high)} kg<br>
                        <span style="color: #10b981; font-weight: bold;">💧 Huile / زيت : ${oilVal} kg</span>
                    `;
                } else {
                    document.getElementById('resYield').innerText = Math.round(data.final_yield_estimate) + " kg/ha";
                    document.getElementById('resInterval').innerText = Math.round(data.confidence_interval_low) + " - " + Math.round(data.confidence_interval_high) + " kg/ha";
                }
            } else {
                document.getElementById('resYield').innerText = "N/A";
                document.getElementById('resInterval').innerText = "N/A";
            }

            // Maladie
            document.getElementById('resDisease').innerText = data.disease_class || (currentLang === 'ar' ? 'غير معروف' : 'Inconnue');
            document.getElementById('resDiseaseConf').innerText = (currentLang === 'ar' ? 'الثقة: ' : 'Confiance : ') + ((data.disease_confidence || 0) * 100).toFixed(1) + "%";

            if (data.disease_severity && data.disease_severity > 0) {
                const sevPct = (data.disease_severity * 100).toFixed(0);
                document.getElementById('resDiseaseSeverity').innerText = currentLang === 'ar'
                    ? `شدة الإصابة: ${sevPct}%`
                    : `Sévérité : ${sevPct}%`;
                document.getElementById('resDiseaseSeverity').style.display = "block";
            } else {
                document.getElementById('resDiseaseSeverity').style.display = "none";
            }

            // NDVI
            document.getElementById('resNDVI').innerText = data.ndvi ? data.ndvi.toFixed(2) : "N/A";
            document.getElementById('resNDVIDate').innerText = (currentLang === 'ar' ? 'التاريخ: ' : 'Date : ') + (data.ndvi_date || "N/A");

            // Sol
            let soilHtml = "";
            if (data.soil_ph) soilHtml += "pH: " + data.soil_ph.toFixed(1) + "<br>";
            if (data.soil_clay_pct) soilHtml += "Argile/طين: " + data.soil_clay_pct.toFixed(1) + "%<br>";
            if (data.soil_sand_pct) soilHtml += "Sable/رمل: " + data.soil_sand_pct.toFixed(1) + "%";
            document.getElementById('resSoil').innerHTML = soilHtml || "N/A";

            // LAI
            document.getElementById('resLAI').innerText = (data.crop_type === 'olive') ? "N/A (Arbres)" : (data.wofost_lai ? data.wofost_lai.toFixed(2) : "N/A");

            // Context pour Gemini
            currentParcelContext = {
                crop_type: data.crop_type,
                yield_estimate: data.final_yield_estimate ? Math.round(data.final_yield_estimate) : 0,
                oil_yield_estimate: data.oil_yield_estimate ? Math.round(data.oil_yield_estimate) : 0,
                disease: data.disease_class,
                disease_severity: data.disease_severity,
                ndvi: data.ndvi,
                soil_ph: data.soil_ph,
                clay_pct: data.soil_clay_pct,
                sand_pct: data.soil_sand_pct,
                services: data.services || {}
            };

            // Notifier dans le chat
            let summaryText = "";
            if (data.crop_type === 'olive') {
                const olivesVal = Math.round(data.final_yield_estimate || 0);
                const oilVal = Math.round(data.oil_yield_estimate || 0);
                summaryText = currentLang === 'ar'
                    ? `✅ <b>تم الانتهاء من التحليل!</b> المردود المتوقع هو <b>${olivesVal} كجم من الزيتون</b> (وحوالي <b>${oilVal} كجم من زيت الزيتون</b>).`
                    : `✅ <b>Analyse complétée !</b> Le rendement estimé est de <b>${olivesVal} kg de olives</b> (environ <b>${oilVal} kg d'huile d'olive</b>).`;
            } else {
                summaryText = currentLang === 'ar' 
                    ? `✅ <b>تم الانتهاء من التحليل!</b> المردود المتوقع هو <b>${Math.round(data.final_yield_estimate || 0)} كجم/هكتار</b>.`
                    : `✅ <b>Analyse complétée !</b> Le rendement estimé est de <b>${Math.round(data.final_yield_estimate || 0)} kg/ha</b>.`;
            }
            appendMessage(summaryText, 'bot');
            chatHistory.push({ role: 'model', parts: [summaryText] });
        }

        function translateCrop(crop) {
            const trans = { 
                fr: { 'wheat': 'Blé', 'barley': 'Orge', 'olive': 'Olivier', 'potato': 'Pomme de terre', 'tomato': 'Tomate' },
                ar: { 'wheat': 'القمح', 'barley': 'الشعير', 'olive': 'الزيتون', 'potato': 'البطاطس', 'tomato': 'الطماطم' }
            };
            return trans[currentLang][crop] || crop;
        }

        async function geocodeLocationName(name) {
            try {
                const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(name)}`);
                const data = await res.json();
                if (data && data.length > 0) {
                    const lat = parseFloat(data[0].lat);
                    const lon = parseFloat(data[0].lon);
                    updateCoordsInputs(lat, lon);
                    marker.setLatLng([lat, lon]);
                    map.setView([lat, lon], 12);
                    appendMessage(currentLang === 'ar' ? `📍 تم تحديد الموقع: ${name}` : `📍 Localisation mise à jour : ${name}`, 'bot');
                }
            } catch (err) {
                console.error("Geocoding failed for:", name, err);
            }
        }

        async function sendChatMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;

            // Ajouter le message utilisateur
            appendMessage(message, 'user');
            input.value = "";
            chatHistory.push({ role: 'user', parts: [message] });

            // Préparer les données actuelles du formulaire
            const currentForm = {
                crop_type: document.getElementById('crop_type').value,
                sowing_date: document.getElementById('sowing_date').value,
                location: `${document.getElementById('lat').value}, ${document.getElementById('lon').value}`,
                historical_yields: document.getElementById('historical_yields').value,
                photo_uploaded: document.getElementById('photo').files.length > 0
            };

            // Message bot "En cours..."
            const loadingText = currentLang === 'ar' ? "المستشار يفكر..." : "L'agronome réfléchit...";
            const errorText = currentLang === 'ar' ? "خطأ في الاتصال بـ Gemini." : "Désolé, une erreur s'est produite avec Gemini Chat.";
            const loadingMsg = appendMessage(loadingText, 'bot');

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: message,
                        history: chatHistory,
                        current_form: currentForm,
                        context: currentParcelContext
                    })
                });

                if (!response.ok) throw new Error("Erreur");

                const result = await response.json();
                loadingMsg.remove();
                
                // Afficher le message
                appendMessage(result.response, 'bot');
                chatHistory.push({ role: 'model', parts: [result.response] });

                // Traiter les données extraites
                if (result.extracted_data) {
                    const ext = result.extracted_data;
                    let hasNewExtraction = false;
                    
                    // 1. Crop type
                    if (ext.crop_type) {
                        document.getElementById('crop_type').value = ext.crop_type;
                        hasNewExtraction = true;
                    }
                    
                    // 2. Sowing date
                    if (ext.sowing_date) {
                        document.getElementById('sowing_date').value = ext.sowing_date;
                        hasNewExtraction = true;
                    }
                    
                    // 3. Location geocoding
                    if (ext.location) {
                        await geocodeLocationName(ext.location);
                        hasNewExtraction = true;
                    }
                    
                    // 4. Historical yields
                    if (ext.historical_yields && Array.isArray(ext.historical_yields)) {
                        historicalRecords = ext.historical_yields.map(y => ({ year: parseInt(y.year), yield: parseFloat(y.yield) }));
                        renderHistoricalRecords();
                        hasNewExtraction = true;
                    }

                    if (hasNewExtraction) {
                        // Mettre en évidence les champs modifiés avec un flash vert d'animation micro-UI
                        document.querySelectorAll('.form-group input, .form-group select').forEach(el => {
                            if (el.value) {
                                el.style.transition = 'box-shadow 0.3s ease';
                                el.style.boxShadow = '0 0 10px rgba(16, 185, 129, 0.5)';
                                setTimeout(() => { el.style.boxShadow = 'none'; }, 1500);
                            }
                        });
                    }

                    // Déclenchement automatique de la prédiction si complet
                    const hasPhoto = document.getElementById('photo').files.length > 0;
                    const readyCrop = document.getElementById('crop_type').value;
                    const readySowing = document.getElementById('sowing_date').value;
                    
                    if (readyCrop && readySowing && !hasPhoto) {
                        appendMessage(currentLang === 'ar' 
                            ? "⚠️ يرجى تحميل صورة لأوراق المحصول في النموذج على اليسار لتشغيل التحليل." 
                            : "⚠️ Veuillez charger une photo des feuilles dans le formulaire à gauche pour lancer l'analyse.", 'bot');
                    } else if (hasPhoto && readyCrop && readySowing) {
                        appendMessage(currentLang === 'ar'
                            ? "🚀 كل المعلومات متوفرة! جاري تشغيل التحليل..."
                            : "🚀 Toutes les informations sont prêtes ! Lancement de l'analyse...", 'bot');
                        
                        // Submit form programmatically
                        document.getElementById('predictForm').requestSubmit();
                    }
                }

            } catch (err) {
                loadingMsg.remove();
                appendMessage(errorText, 'bot');
            }
        }

        function appendMessage(text, sender) {
            const chatBox = document.getElementById('chatBox');
            const msgEl = document.createElement('div');
            msgEl.className = 'chat-message ' + (sender === 'user' ? 'message-user' : 'message-bot');
            msgEl.innerText = text;
            chatBox.appendChild(msgEl);
            chatBox.scrollTop = chatBox.scrollHeight;
            return msgEl;
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


