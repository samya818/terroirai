from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from datetime import date

@dataclass
class ParcelRecord:
    parcel_id: str
    farmer_id: str
    lat: float
    lon: float
    timestamp: date
    crop_type: str  # "wheat", "barley", "olive"
    
    # Résultat du modèle maladies (étape 1)
    disease_class: Optional[str] = None
    disease_confidence: Optional[float] = None
    disease_severity: Optional[float] = None
    photo_quality_score: Optional[float] = None
    
    # Résultat du sol (étape 2)
    soil_ph: Optional[float] = None
    soil_clay_pct: Optional[float] = None
    soil_sand_pct: Optional[float] = None
    
    # Résultat satellite (étape 3)
    ndvi: Optional[float] = None
    ndvi_date: Optional[date] = None  # DATE de l'image satellite, pas juste la valeur !
    
    # Résultat WOFOST (étape 4)
    wofost_yield_kg_ha: Optional[float] = None
    wofost_lai: Optional[float] = None
    wofost_calibration_factor: Optional[float] = None
    wofost_calibration_confidence: Optional[float] = None
    ndvi_signal_quality: Optional[float] = None
    
    # Historique déclaré par l'agriculteur
    farmer_historical_yields: Optional[List[Tuple[int, float]]] = None  # [(2022, 2500), (2023, 1800), ...]
    
    # Résultat final (à construire à l'étape 6)
    final_yield_estimate: Optional[float] = None
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None
    final_prediction_confidence: Optional[float] = None
    
    # Paramètres et résultats optionnels pour l'olivier
    oil_yield_estimate: Optional[float] = None
    olive_trees_count: Optional[int] = None

    def to_dict(self) -> dict:
        """Sérilialise l'objet en dictionnaire JSON-friendly (conversion des dates en chaînes ISO)."""
        d = asdict(self)
        if isinstance(d['timestamp'], date):
            d['timestamp'] = d['timestamp'].isoformat()
        if isinstance(d['ndvi_date'], date):
            d['ndvi_date'] = d['ndvi_date'].isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ParcelRecord":
        """Désérialise un dictionnaire JSON-friendly en instance de ParcelRecord."""
        timestamp_val = data.get('timestamp')
        if isinstance(timestamp_val, str):
            timestamp_val = date.fromisoformat(timestamp_val)
        
        ndvi_date_val = data.get('ndvi_date')
        if isinstance(ndvi_date_val, str):
            ndvi_date_val = date.fromisoformat(ndvi_date_val)
        
        historical = data.get('farmer_historical_yields')
        if historical:
            historical = [tuple(x) if isinstance(x, (list, tuple)) else x for x in historical]
        
        return cls(
            parcel_id=data['parcel_id'],
            farmer_id=data['farmer_id'],
            lat=data['lat'],
            lon=data['lon'],
            timestamp=timestamp_val,
            crop_type=data['crop_type'],
            disease_class=data.get('disease_class'),
            disease_confidence=data.get('disease_confidence'),
            disease_severity=data.get('disease_severity'),
            photo_quality_score=data.get('photo_quality_score'),
            soil_ph=data.get('soil_ph'),
            soil_clay_pct=data.get('soil_clay_pct'),
            soil_sand_pct=data.get('soil_sand_pct'),
            ndvi=data.get('ndvi'),
            ndvi_date=ndvi_date_val,
            wofost_yield_kg_ha=data.get('wofost_yield_kg_ha'),
            wofost_lai=data.get('wofost_lai'),
            wofost_calibration_factor=data.get('wofost_calibration_factor'),
            wofost_calibration_confidence=data.get('wofost_calibration_confidence'),
            ndvi_signal_quality=data.get('ndvi_signal_quality'),
            farmer_historical_yields=historical,
            final_yield_estimate=data.get('final_yield_estimate'),
            confidence_interval_low=data.get('confidence_interval_low'),
            confidence_interval_high=data.get('confidence_interval_high'),
            final_prediction_confidence=data.get('final_prediction_confidence'),
            oil_yield_estimate=data.get('oil_yield_estimate'),
            olive_trees_count=data.get('olive_trees_count')
        )
