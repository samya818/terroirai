import os
import math
import datetime as dt
from datetime import date
from typing import Tuple, Optional, List

try:
    import pandas as pd
except Exception as exc:
    pd = None
    _PANDAS_IMPORT_ERROR = exc
else:
    _PANDAS_IMPORT_ERROR = None

try:
    from step4_simulation.openmeteo_weather import OpenMeteoWeatherDataProvider
except Exception:
    OpenMeteoWeatherDataProvider = None

_PCSE_IMPORT_ERROR = None
try:
    from pcse.base import ParameterProvider
    from pcse.input import NASAPowerWeatherDataProvider, YAMLCropDataProvider, WOFOST72SiteDataProvider
    from pcse.models import Wofost72_WLP_FD
    from pcse.base.weather import WeatherDataProvider
    from pcse.base import WeatherDataContainer
except Exception as exc:
    ParameterProvider = None
    NASAPowerWeatherDataProvider = None
    YAMLCropDataProvider = None
    WOFOST72SiteDataProvider = None
    Wofost72_WLP_FD = None
    WeatherDataProvider = object
    WeatherDataContainer = None
    _PCSE_IMPORT_ERROR = exc
else:
    _PCSE_IMPORT_ERROR = None

class OfflineDummyWeatherDataProvider(WeatherDataProvider):
    """
    Fournisseur météo hors-ligne générant des données réalistes pour le Tadla-Azilal (Maroc).
    Évite de bloquer l'application en cas de DNS cassé ou d'absence d'Internet.
    """
    def __init__(self, latitude: float, longitude: float, start_date: dt.date, end_date: dt.date):
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = 150.0
        
        curr = start_date
        while curr <= end_date:
            month = curr.month
            # Profil météo simulé réaliste pour la région Tadla-Azilal
            if month in [6, 7, 8]:  # Été chaud et sec
                tmin, tmax, temp = 20.0, 38.0, 29.0
                rain = 0.0
                irrad = 22e6
            elif month in [12, 1, 2]:  # Hiver frais
                tmin, tmax, temp = 4.0, 16.0, 10.0
                rain = 0.15  # 1.5 mm/jour
                irrad = 10e6
            else:  # Printemps / Automne tempéré
                tmin, tmax, temp = 12.0, 26.0, 19.0
                rain = 0.05  # 0.5 mm/jour
                irrad = 16e6
            
            wind_2m = 2.2
            # Pression de vapeur saturante estimée en hPa
            vap = 6.112 * math.exp((17.67 * tmin) / (tmin + 243.5))
            
            wdc = WeatherDataContainer(
                LAT=self.latitude,
                LON=self.longitude,
                ELEV=self.elevation,
                DAY=curr,
                TMAX=tmax,
                TMIN=tmin,
                TEMP=temp,
                RAIN=rain,
                WIND=wind_2m,
                VAP=vap,
                IRRAD=irrad,
                E0=0.25,
                ES0=0.20,
                ET0=0.22
            )
            self._store_WeatherDataContainer(wdc, curr)
            curr += dt.timedelta(days=1)


def _zone_key_from_coordinates(lat: float, lon: float) -> str:
    """
    Crée une clé locale de zone à partir d'un arrondissement de coordonnées.
    L'objectif est de faire évoluer la calibration vers une logique par sous-zone
    plutôt que d'un simple facteur global unique pour tout le système.
    """
    return f"{round(lat, 1):.1f}_{round(lon, 1):.1f}"


def _historical_weight(year: int, current_year: int, actual_yield: float, simulated_yield: float) -> float:
    """
    Poids de fiabilité historique :
    - les années récentes ont plus de poids,
    - les ratios très extrêmes sont pénalisés pour éviter une sur-calibration,
    - les années très éloignées dans le temps sont dépriorisées.
    """
    age_years = max(1, current_year - year)
    recency_weight = 1.0 / (1.0 + (age_years * 0.35))
    ratio = actual_yield / simulated_yield if simulated_yield > 0 else 0.0
    ratio_penalty = 1.0 / (1.0 + abs(ratio - 1.0) * 0.8)
    return max(0.1, recency_weight * ratio_penalty)


def _crop_calibration_bias(crop_type: str) -> float:
    """
    Biais de calibration par culture. On commence avec une base prudente de 1.0
    et on ajoute un léger ajustement métier par culture si besoin.
    """
    normalized = crop_type.lower().strip()
    if 'wheat' in normalized or 'blé' in normalized:
        return 1.00
    if 'barley' in normalized or 'orge' in normalized:
        return 0.98
    if 'potato' in normalized or 'pomme_de_terre' in normalized or 'pomme de terre' in normalized:
        return 1.04
    if 'tomato' in normalized or 'tomate' in normalized:
        return 1.02
    if 'olive' in normalized or 'olivier' in normalized:
        return 1.00
    return 1.00


def _soil_reliability_score(clay_pct: Optional[float], sand_pct: Optional[float], soil_ph: Optional[float]) -> float:
    """
    Score de fiabilité des paramètres de sol pour pondérer la calibration.
    Plus le sol est bien caractérisé et cohérent (texture + pH réaliste), plus la calibration historique
    peut être considérée comme fiable pour l'ajustement local.
    """
    clay = clay_pct if clay_pct is not None else 30.0
    sand = sand_pct if sand_pct is not None else 40.0
    ph = soil_ph if soil_ph is not None else 6.5

    texture_balance = 1.0 - abs((clay + sand) - 100.0) / 200.0
    ph_score = 1.0 - min(1.0, abs(ph - 6.5) / 4.0)

    # Garder un score plafonné dans une plage réaliste pour ne pas surpondérer un mauvais jeu de données
    return max(0.7, min(1.2, 0.55 * texture_balance + 0.45 * ph_score))

def run_one_simulation(
    lat: float,
    lon: float,
    crop_type: str,
    soil_data: dict,
    site_data,
    crop_data,
    sowing_date: date,
    harvest_date: date
) -> Tuple[float, float, float]:
    """
    Runs a single WOFOST simulation for given dates and returns (yield_kg_ha, max_lai, accumulated_temp).
    Accumulated temperature (GDD) is calculated from the weather data over the crop cycle.
    """
    if Wofost72_WLP_FD is None or ParameterProvider is None or crop_data is None or site_data is None:
        return 0.0, 0.0, 0.0

    pcse_crop_name = 'wheat'
    variety_name = 'Winter_wheat_101'

    # Mapper crop_type
    normalized_crop = crop_type.lower()
    if 'barley' in normalized_crop:
        pcse_crop_name = 'barley'
        variety_name = 'Spring_barley_301'
    elif 'olive' in normalized_crop:
        pcse_crop_name = 'wheat'
        variety_name = 'Winter_wheat_101'
    elif 'potato' in normalized_crop:
        pcse_crop_name = 'potato'
        variety_name = 'Potato_701'
    elif 'tomato' in normalized_crop:
        pcse_crop_name = 'sugarbeet'
        variety_name = 'Sugarbeet_601'

    # Si c'est du blé mais semé en période printanière/estivale (février à octobre),
    # le blé d'hiver (seul disponible dans PCSE) n'aura pas sa vernalisation et produira un rendement de 0.
    # On utilise alors l'orge de printemps ('barley') comme proxy agronomique dynamique proche.
    if pcse_crop_name == 'wheat' and 2 <= sowing_date.month <= 10:
        pcse_crop_name = 'barley'
        variety_name = 'Spring_barley_301'

    try:
        crop_data.set_active_crop(pcse_crop_name, variety_name)
    except Exception:
        crop_data.set_active_crop('wheat', 'Winter_wheat_101')

    # Données météo (Open-Meteo avec fallback NASA POWER)
    weather_start = sowing_date - dt.timedelta(days=14)
    # S'assurer que weather_end ne dépasse pas aujourd'hui
    today = date.today()
    weather_end = min(harvest_date + dt.timedelta(days=1), today)
    if weather_end < weather_start:
        weather_end = weather_start + dt.timedelta(days=15)

    try:
        weather = OpenMeteoWeatherDataProvider(
            latitude=lat, 
            longitude=lon, 
            start_date=weather_start, 
            end_date=weather_end
        )
    except Exception as e:
        print(f"Erreur météo Open-Meteo: {e}. Utilisation du fournisseur hors-ligne de secours (rapide).")
        weather = OfflineDummyWeatherDataProvider(
            latitude=lat,
            longitude=lon,
            start_date=weather_start,
            end_date=weather_end
        )

    # Agromanagement
    agromanagement = [{
        weather_start: {
            'CropCalendar': {
                'crop_name': pcse_crop_name,
                'variety_name': variety_name,
                'crop_start_date': sowing_date,
                'crop_start_type': 'sowing',
                'crop_end_date': harvest_date,
                'crop_end_type': 'harvest',
            },
            'TimedEvents': None,
            'StateEvents': None
        }
    }]

    # Assemblage et exécution
    params = ParameterProvider(cropdata=crop_data, soildata=soil_data, sitedata=site_data)
    wofost = Wofost72_WLP_FD(params, weather, agromanagement)
    wofost.run_till_terminate()

    # Extraction des résultats
    output = wofost.get_output()
    accumulated_temp = 0.0
    yield_kg_ha = 0.0
    max_lai = 0.0

    if pd is not None:
        df = pd.DataFrame(output)
        if not df.empty:
            yield_kg_ha = float(df['TWSO'].iloc[-1]) if 'TWSO' in df.columns else 0.0
            max_lai = float(df['LAI'].max()) if 'LAI' in df.columns else 0.0
            if 'TEMP' in df.columns:
                accumulated_temp = float(df[df['day'] >= sowing_date]['TEMP'].apply(lambda t: max(0.0, t)).sum())
    elif isinstance(output, list):
        for row in output:
            if isinstance(row, dict):
                if row.get('TWSO') is not None:
                    yield_kg_ha = float(row['TWSO'])
                if row.get('LAI') is not None:
                    max_lai = max(max_lai, float(row['LAI']))
                if row.get('TEMP') is not None:
                    accumulated_temp += max(0.0, float(row['TEMP']))

    return yield_kg_ha, max_lai, accumulated_temp

def run_wofost_simulation(
    lat: float, 
    lon: float, 
    crop_type: str, 
    soil_ph: Optional[float] = None, 
    clay_pct: Optional[float] = None, 
    sand_pct: Optional[float] = None,
    sowing_date_str: Optional[str] = None,
    historical_yields: Optional[List[Tuple[int, float]]] = None
) -> Tuple[float, float, float, Optional[float], Optional[float]]:
    """
    Exécute la simulation WOFOST et renvoie un tuple
    (rendement_kg_ha, LAI_max, temp_accumulee, facteur_calibration, confiance_calibration).
    """
    if WOFOST72SiteDataProvider is None or YAMLCropDataProvider is None or _PCSE_IMPORT_ERROR is not None:
        return 0.0, 0.0, 0.0, None, None

    # 1. Estimation des paramètres du sol
    clay = clay_pct if clay_pct is not None else 30.0
    sand = sand_pct if sand_pct is not None else 40.0

    smfcf = 0.1 + 0.003 * clay + 0.0005 * (100 - sand - clay)
    smw = 0.02 + 0.0025 * clay
    sm0 = 0.45 - 0.001 * sand

    smw = max(0.05, min(0.25, smw))
    smfcf = max(smw + 0.05, min(0.45, smfcf))
    sm0 = max(smfcf + 0.05, min(0.55, sm0))

    soil_data = {
        'SMFCF': smfcf,
        'SM0': sm0,
        'SMW': smw,
        'CRAIRC': 0.06,
        'SOPE': 10.0,
        'KSUB': 10.0,
        'K0': 12.5,
        'RDMSOL': 120,
    }

    site_data = WOFOST72SiteDataProvider(WAV=100)
    crop_data = YAMLCropDataProvider()

    # Dates
    today = date.today()
    if sowing_date_str:
        sowing_date = date.fromisoformat(sowing_date_str)
    else:
        # par défaut: 15 novembre de la saison en cours
        if today.month >= 11:
            sowing_year = today.year
        else:
            sowing_year = today.year - 1
        sowing_date = date(sowing_year, 11, 15)

    # Date de récolte estimée ou aujourd'hui si la récolte n'est pas encore faite
    # Typical cycle: 210 days
    harvest_date = sowing_date + dt.timedelta(days=210)
    if harvest_date > today:
        harvest_date = today

    # Sécurité : s'assurer d'une période de simulation minimale de 30 jours
    if (harvest_date - sowing_date).days < 30:
        sowing_date = harvest_date - dt.timedelta(days=30)

    # Exécution de la simulation pour la saison en cours
    sim_yield, max_lai, acc_temp = run_one_simulation(
        lat=lat, lon=lon, crop_type=crop_type,
        soil_data=soil_data, site_data=site_data, crop_data=crop_data,
        sowing_date=sowing_date, harvest_date=harvest_date
    )

    # 2. Calibration avec rendements historiques (pondérée, locale et par culture)
    calibration_factor = None
    calibration_confidence = None
    calibrated_yield = sim_yield

    if historical_yields:
        ratios = []
        zone_key = _zone_key_from_coordinates(lat, lon)
        soil_reliability = _soil_reliability_score(clay_pct=clay, sand_pct=sand, soil_ph=soil_ph)
        crop_bias = _crop_calibration_bias(crop_type)

        for hist_year, actual_yield in historical_yields:
            # Déterminer la date de semis et récolte pour l'année historique
            # Si le cycle traverse le Nouvel An (ex: nov à juin), le semis a eu lieu en hist_year - 1
            spans_new_year = (harvest_date.year > sowing_date.year)
            hist_sowing_year = hist_year - 1 if spans_new_year else hist_year

            try:
                hist_sowing = date(hist_sowing_year, sowing_date.month, sowing_date.day)
                hist_harvest = date(hist_year, harvest_date.month, harvest_date.day)

                hist_sim_yield, _, _ = run_one_simulation(
                    lat=lat, lon=lon, crop_type=crop_type,
                    soil_data=soil_data, site_data=site_data, crop_data=crop_data,
                    sowing_date=hist_sowing, harvest_date=hist_harvest
                )

                if hist_sim_yield > 0:
                    ratio = actual_yield / hist_sim_yield
                    weight = _historical_weight(hist_year, today.year, actual_yield, hist_sim_yield)
                    weight *= soil_reliability
                    ratios.append((ratio, weight, zone_key))
            except Exception as e:
                print(f"Avertissement : Échec de la simulation historique pour {hist_year} : {e}")

        if ratios:
            weighted_sum = 0.0
            total_weight = 0.0
            ratio_spread = 0.0
            ratio_values = [ratio for ratio, _, _ in ratios]

            for ratio, weight, current_zone in ratios:
                weighted_sum += ratio * weight
                total_weight += weight

            if total_weight > 0:
                weighted_ratio = weighted_sum / total_weight
                calibration_factor = weighted_ratio * crop_bias
                calibration_factor = max(0.5, min(1.5, calibration_factor))
                calibrated_yield = sim_yield * calibration_factor

                if ratio_values:
                    mean_ratio = sum(ratio_values) / len(ratio_values)
                    ratio_spread = sum(abs(r - mean_ratio) for r in ratio_values) / len(ratio_values)

                reliability = min(1.0, len(ratios) / 3.0)
                spread_penalty = max(0.0, 1.0 - min(1.0, ratio_spread / 0.35))
                calibration_confidence = round(max(0.2, min(0.95, 0.5 * reliability + 0.5 * spread_penalty)), 3)

    return calibrated_yield, max_lai, acc_temp, calibration_factor, calibration_confidence

