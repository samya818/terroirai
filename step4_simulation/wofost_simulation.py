from pcse.base import ParameterProvider
from pcse.input import NASAPowerWeatherDataProvider
from pcse.input import YAMLCropDataProvider
from pcse.models import Wofost72_WLP_FD
from pcse.input import WOFOST72SiteDataProvider
import pandas as pd
from datetime import date

# 1. DONNÉES DU SOL (depuis l'étape 2)
# Exemple pour Beni Mellal
soil_data = {
    'SMFCF': 0.2,   # capacité au champ
    'SM0': 0.4,     # saturation
    'SMW': 0.1,     # point de flétrissement permanent
    'CRAIRC': 0.06, # air critique
    'SOPE': 10.0,   # percolation max zone racinaire (cm/jour)
    'KSUB': 10.0,   # percolation max sous-sol (cm/jour)
    'K0': 12.5,     # conductivité hydraulique (cm/jour)
    'RDMSOL': 120,  # profondeur max du sol (cm)
}

# 2. DONNÉES DE CONFIGURATION DU SITE
site_data = WOFOST72SiteDataProvider(WAV=100)

# 3. DONNÉES DE LA CULTURE (blé)
# Utilisation de YAMLCropDataProvider (inclus dans PCSE)
crop_data = YAMLCropDataProvider()
crop_data.set_active_crop('wheat', 'Winter_wheat_101')

# 4. DONNÉES MÉTÉO (NASA POWER)
# Coordonnées pour Beni Mellal
print("Récupération des données météo NASA POWER...")
weather = NASAPowerWeatherDataProvider(latitude=32.33, longitude=-6.36)
print("Données météo récupérées avec succès.")

# 5. AGROMANAGEMENT (Calendrier de culture)
agromanagement = [{
    date(2024, 11, 1): {
        'CropCalendar': {
            'crop_name': 'wheat',
            'variety_name': 'Winter_wheat_101',
            'crop_start_date': date(2024, 11, 15),
            'crop_start_type': 'sowing',
            'crop_end_date': date(2025, 6, 15),
            'crop_end_type': 'harvest',
        },
        'TimedEvents': None,
        'StateEvents': None
    }
}]

# 6. ASSEMBLAGE DES PARAMÈTRES
params = ParameterProvider(cropdata=crop_data, soildata=soil_data, sitedata=site_data)

# 7. EXÉCUTION DE LA SIMULATION
print("Démarrage de la simulation WOFOST...")
wofost = Wofost72_WLP_FD(params, weather, agromanagement)
wofost.run_till_terminate()
print("Simulation terminée.")

# 8. RÉSULTATS
output = wofost.get_output()
df = pd.DataFrame(output)
print("\n--- Résultats de la simulation (5 derniers jours) ---")
print(df[['day', 'LAI', 'TAGP', 'TWSO']].tail())
