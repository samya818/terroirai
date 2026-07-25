import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import os
import json
import math

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soil_cache.json")

# Valeurs régionales par défaut en fonction de la proximité géographique (ex: Maroc/Beni Mellal et global)
REGIONAL_DEFAULTS = [
    {
        "name": "Beni Mellal / Tadla (Morocco)",
        "lat_min": 31.5, "lat_max": 33.5,
        "lon_min": -7.5, "lon_max": -5.5,
        "phh2o": 78,      # phh2o = pH * 10
        "clay": 280,     # clay = % * 10
        "sand": 350,     # sand = % * 10
        "ocd": 120       # ocd = % * 10 (ou similaire)
    },
    {
        "name": "Global Default",
        "lat_min": -90.0, "lat_max": 90.0,
        "lon_min": -180.0, "lon_max": 180.0,
        "phh2o": 75,
        "clay": 300,
        "sand": 400,
        "ocd": 150
    }
]

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_data):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except Exception:
        pass

def get_regional_default(lat, lon):
    for region in REGIONAL_DEFAULTS:
        if region["lat_min"] <= lat <= region["lat_max"] and region["lon_min"] <= lon <= region["lon_max"]:
            return {
                "phh2o": region["phh2o"],
                "clay": region["clay"],
                "sand": region["sand"],
                "ocd": region["ocd"]
            }
    return {
        "phh2o": 75,
        "clay": 300,
        "sand": 400,
        "ocd": 150
    }

def fetch_from_api(lat, lon, timeout=30):
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "depth": "0-5cm",
        "properties": "phh2o,clay,sand,ocd",
        "value": "mean"
    }
    
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    
    result = {}
    layers = data.get('properties', {}).get('layers', [])
    if not layers:
        return None  # Pas de données à ces coordonnées
        
    for prop in layers:
        name = prop['name']
        depths = prop.get('depths', [])
        if depths:
            values = depths[0].get('values', {})
            value = values.get('mean', None)
            result[name] = value
            
    # S'assurer que les propriétés requises sont bien présentes et non nulles
    if not any(value is not None for value in result.values()):
        return None
        
    result['source'] = 'SoilGrids'
    return result

def get_soil_data(lat, lon):
    """
    Récupère le pH, l'argile et le sable à une coordonnée GPS via SoilGrids v2.0.
    Met en œuvre plusieurs stratégies de robustesse :
      1. Cache local pour éviter les requêtes API redondantes et lentes.
      2. Timeout configuré et tentatives automatiques.
      3. Recherche spatiale : si la coordonnée échoue ou renvoie null (ex: zone urbaine ou masque littoral),
         on balaie une grille de 4 points adjacents à +/- 0.05 degré (~ 5km).
      4. Fallback régional précis si l'API externe est complètement inaccessible ou inefficace.
    """
    # 1. Vérification du cache (arrondi à 4 décimales pour stabiliser la clé)
    cache_key = f"{round(lat, 4)},{round(lon, 4)}"
    cache = load_cache()
    if cache_key in cache:
        return cache[cache_key]

    # 2. Tentative sur le point d'origine
    try:
        res = fetch_from_api(lat, lon, timeout=30)
        if res is not None:
            cache[cache_key] = res
            save_cache(cache)
            return res
    except Exception as e:
        print(f"[SoilGrids API] Échec sur le point initial ({lat}, {lon}) : {e}")

    # 3. Balayage spatial (recherche des points voisins)
    # SoilGrids peut échouer ou renvoyer des couches vides à cause de masques (villes, rivières, rochers)
    # On essaye sur un rayon de ~ 5 km (+/- 0.05 degré de latitude/longitude)
    offsets = [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05)]
    for dlat, dlon in offsets:
        neighbor_lat = round(lat + dlat, 4)
        neighbor_lon = round(lon + dlon, 4)
        print(f"[SoilGrids API] Recherche sur le point voisin ({neighbor_lat}, {neighbor_lon})...")
        try:
            res = fetch_from_api(neighbor_lat, neighbor_lon, timeout=15)
            if res is not None:
                print(f"[SoilGrids API] Succès sur le point voisin ({neighbor_lat}, {neighbor_lon}) !")
                cache[cache_key] = res
                save_cache(cache)
                return res
        except Exception as e:
            print(f"[SoilGrids API] Échec sur le voisin ({neighbor_lat}, {neighbor_lon}) : {e}")

    # 4. Fallback de secours rigoureux (Régional / Global)
    print(f"[SoilGrids API] API SoilGrids indisponible ou coordonnées hors-limite. Utilisation du fallback régional.")
    fallback = get_regional_default(lat, lon)
    fallback['source'] = 'fallback-regional'
    # On n'enregistre pas forcément dans le cache persistant les fallbacks afin de retenter l'API plus tard
    return fallback

if __name__ == "__main__":
    # Test avec des coordonnées de Beni Mellal
    soil = get_soil_data(32.32, -6.38)
    print("Résultats obtenus :", soil)


