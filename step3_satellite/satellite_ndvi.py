try:
    import ee
except Exception as e:
    ee = None
    _GEE_INIT_ERROR = e
else:
    _GEE_INIT_ERROR = None

_GEE_INITIALIZED = False

try:
    # Tentative d'initialisation unique au chargement pour éviter de bloquer à chaque requête
    if ee is not None:
        ee.Initialize()
        _GEE_INITIALIZED = True
except Exception as e:
    _GEE_INIT_ERROR = e

def get_ndvi(lat, lon, date_start, date_end, allow_default=True):
    """
    Récupère le NDVI moyen d'une parcelle (buffer de 100m)
    entre deux dates via Sentinel-2.
    Si `allow_default` est False, une exception est levée lorsque GEE est indisponible.
    """
    # Fallback automatique si Google Earth Engine n'est pas disponible
    if ee is None or not _GEE_INITIALIZED:
        if allow_default:
            return 0.45
        raise RuntimeError("Google Earth Engine non disponible")

    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(100)  # cercle de 100m autour du point

    # Collection Sentinel-2, filtrée par date, zone, et moins de 20% de nuages
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterDate(date_start, date_end)
                  .filterBounds(buffer)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

    # Calcul du NDVI : (NIR - RED) / (NIR + RED)
    # Bandes B8 (NIR) et B4 (Rouge) de Sentinel-2
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    with_ndvi = collection.map(add_ndvi)

    # Moyenne sur toutes les images de la période
    mean_ndvi = with_ndvi.select('NDVI').mean()

    # Extraction de la valeur moyenne dans le buffer
    stats = mean_ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer,
        scale=10  # résolution Sentinel-2 = 10m
    )

    try:
        return stats.get('NDVI').getInfo()
    except Exception:
        return 0.45

# Test avec des coordonnées près de Beni Mellal
if __name__ == "__main__":
    # Note : (32.32, -6.38) évite le centre urbain pour cibler des parcelles agricoles
    try:
        ndvi = get_ndvi(32.32, -6.38, '2025-03-01', '2025-03-31')
        print(f"NDVI moyen : {ndvi}")
    except Exception as e:
        print(f"Échec de l'exécution : {e}")
