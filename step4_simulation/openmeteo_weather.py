import datetime as dt
import requests
import math
import os
import json

_PCSE_IMPORT_ERROR = None
try:
    from pcse.base.weather import WeatherDataProvider
    from pcse.base import WeatherDataContainer
    from pcse.util import reference_ET
    from pcse.exceptions import PCSEError
except Exception as exc:
    WeatherDataProvider = object
    WeatherDataContainer = None
    reference_ET = None
    PCSEError = Exception
    _PCSE_IMPORT_ERROR = exc
else:
    _PCSE_IMPORT_ERROR = None

class OpenMeteoWeatherDataProvider(WeatherDataProvider):
    """
    Weather data provider that fetches daily weather variables from Open-Meteo API.
    Supports historical data (archive) and current season data.
    """
    
    def __init__(self, latitude: float, longitude: float, start_date: dt.date, end_date: dt.date):
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.angstA = 0.25
        self.angstB = 0.50

        if WeatherDataContainer is None or reference_ET is None:
            raise RuntimeError(f"PCSE n'est pas disponible : {_PCSE_IMPORT_ERROR}")

        # 1. Fetch data from Open-Meteo API
        self._fetch_openmeteo_data(start_date, end_date)
        
    def _fetch_openmeteo_data(self, start_date: dt.date, end_date: dt.date):
        cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_cache.json")
        cache_key = f"{round(self.latitude, 3)},{round(self.longitude, 3)},{start_date.isoformat()},{end_date.isoformat()}"
        
        # Charger le cache
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                pass
                
        # Si présent dans le cache
        if cache_key in cache:
            data = cache[cache_key]
        else:
            # API endpoint for Open-Meteo Historical Archive
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "temperature_2m_mean",
                    "precipitation_sum",
                    "wind_speed_10m_mean",
                    "dewpoint_2m_mean",
                    "shortwave_radiation_sum"
                ],
                "timezone": "auto"
            }
            
            try:
                r = requests.get(url, params=params, timeout=3)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                # Fallback to forecast API if archive API fails or is not yet complete for recent dates
                url_forecast = "https://api.open-meteo.com/v1/forecast"
                try:
                    r = requests.get(url_forecast, params=params, timeout=3)
                    r.raise_for_status()
                    data = r.json()
                except Exception as ex:
                    raise PCSEError(f"Failed to fetch weather data from Open-Meteo: {ex}")
            
            # Sauvegarder dans le cache
            try:
                cache[cache_key] = data
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            except Exception:
                pass
        
        self.elevation = data.get("elevation", 150.0)
        daily_data = data.get("daily", {})
        
        dates_str = daily_data.get("time", [])
        tmax_list = daily_data.get("temperature_2m_max", [])
        tmin_list = daily_data.get("temperature_2m_min", [])
        tmean_list = daily_data.get("temperature_2m_mean", [])
        precip_list = daily_data.get("precipitation_sum", [])
        wind_list = daily_data.get("wind_speed_10m_mean", [])
        dew_list = daily_data.get("dewpoint_2m_mean", [])
        rad_list = daily_data.get("shortwave_radiation_sum", [])
        
        for i, d_str in enumerate(dates_str):
            day = dt.date.fromisoformat(d_str)
            tmax = tmax_list[i]
            tmin = tmin_list[i]
            tmean = tmean_list[i]
            precip = precip_list[i]
            wind_10m = wind_list[i]
            tdew = dew_list[i]
            rad_mj = rad_list[i]
            
            # Check for NaNs/None
            if None in [tmax, tmin, tmean, precip, wind_10m, tdew, rad_mj]:
                continue
                
            # Convert units
            # Wind speed at 2m from 10m wind speed
            wind_2m = wind_10m * 0.748
            
            # Vapor pressure from dew point temperature (Tetens formula)
            vap = 6.112 * math.exp((17.67 * tdew) / (tdew + 243.5))
            
            # Global Radiation in J/m2/day
            irrad = rad_mj * 1e6
            
            # Rain in cm/day
            rain = precip / 10.0
            
            # Calculate Reference Evapotranspiration
            try:
                E0, ES0, ET0 = reference_ET(
                    day, self.latitude, self.elevation, tmin, tmax, irrad,
                    vap, wind_2m, self.angstA, self.angstB, self.ETmodel
                )
            except Exception as e:
                # Fallback default ET calculation
                E0, ES0, ET0 = 0.2, 0.2, 0.2
            
            wdc = WeatherDataContainer(
                LAT=self.latitude,
                LON=self.longitude,
                ELEV=self.elevation,
                DAY=day,
                TMAX=tmax,
                TMIN=tmin,
                TEMP=tmean,
                RAIN=rain,
                WIND=wind_2m,
                VAP=vap,
                IRRAD=irrad,
                E0=E0/10.0,
                ES0=ES0/10.0,
                ET0=ET0/10.0
            )
            
            self._store_WeatherDataContainer(wdc, day)
