# TerroirAI : Documentation de Référence, Manuel de Conception et Spécification Technique

TerroirAI est une plateforme d'aide à la décision agronomique de précision pour les parcelles agricoles du Tadla-Azilal (Maroc). Ce document détaille de manière exhaustive l'architecture logicielle, la construction étape par étape, les algorithmes de traitement, les équations physiques et mathématiques, les choix de conception, les difficultés rencontrées, leurs solutions, ainsi que les limitations et choix d'approximations techniques faits en toute transparence (honnêteté technique).

---

## 1. Organisation du Code et Fichiers Source

```text
terroirai/
├── main.py                     # API Gateway FastAPI & Interface web bilingue HTML/JS
├── schema.py                   # Structure et sérialisation des enregistrements parcelles (ParcelRecord)
├── step1_disease/              # Module 1 : Vision par ordinateur & Sévérité
│   ├── predict.py              # CNN Multi-Modèles paresseux + Algorithme HSV d'estimation de sévérité
│   └── models/                 # Dossier des modèles pré-entraînés
│       ├── PlantVillage/       # Modèle Général (38 classes - Tomates, pommes de terre, etc.)
│       ├── OliveLeaf/          # Modèle Olivier (3 classes)
│       ├── citrus_leaf/        # Modèle Agrumes (9 classes - Citron, Mandarine)
│       └── weatdiseases/       # Modèle Blé et Orge (15 classes)
├── step2_soil/                 # Module 2 : Propriétés et hydraulique des sols
│   ├── soil_api.py             # Client ISRIC SoilGrids v2.0 + Cache local + Fallback
│   └── soil_cache.json         # Cache persistant des requêtes de sol
├── step3_satellite/            # Module 3 : Télédétection et vigueur végétative
│   └── satellite_ndvi.py       # Intégration Google Earth Engine (Sentinel-2 SR)
├── step4_simulation/           # Module 4 : Croissance de culture physique
│   ├── openmeteo_weather.py    # Provider météo Open-Meteo pour PCSE/WOFOST
│   ├── simulation_runner.py    # Orchestration WOFOST, climat de la saison et Calibration
│   └── wofost_simulation.py    # Script de test unitaire pour WOFOST
├── step6_ml/                   # Module 6 : Apprentissage supervisé et Fusion
│   ├── yield_predictor.py      # Entraînement RandomForestRegressor + Inference
│   └── yield_rf_model.pkl      # Modèle Random Forest sérialisé
└── README.md                   # Ce fichier de documentation
```

---

## 2. Diagramme Conceptuel du Flux de Données

Le flux ci-dessous montre comment l'application combine la télédétection, la vision par ordinateur, les bases de données géophysiques mondiales, la thermodynamique de croissance et le Machine Learning :

```mermaid
graph TD
    A[Agriculteur : GPS + Date de Semis + Rendements Historiques + Photo] --> B[API Gateway FastAPI /predict]
    
    subgraph Diagnostic Sanitaire (Step 1)
        B -->|Photo de la culture| C[step1_disease/predict.py]
        C -->|1. CNN Classifier| C1[Classe de Maladie & Confiance]
        C -->|2. Masquage de Couleur HSV| C2[Taux de Sévérité de l'infection %]
    end
    
    subgraph Caractérisation du Sol (Step 2)
        B -->|Coordonnées GPS| D[step2_soil/soil_api.py]
        D -->|SoilGrids API + Balayage spatial| D1[Texture : Argile, Sable, pH]
        D1 -->|Formules de Pédotransfert| D2[Hydraulique du sol : SMFCF, SM0, SMW]
    end
    
    subgraph Vigueur Satellite (Step 3)
        B -->|GPS + Fenêtre temporelle| E[step3_satellite/satellite_ndvi.py]
        E -->|Sentinel-2 GEE Cloud Masking| E1[Indice NDVI Moyen]
    end
    
    subgraph Moteur Biophysique WOFOST (Step 4)
        D2 & B --> F[step4_simulation/simulation_runner.py]
        F -->|Météo dynamique Open-Meteo| F1[Climat réel quotidien de la saison]
        F -->|Historique météo + Rendements réels| F2[Ajustement de calibration du terroir]
        F1 & F2 --> F3[Rendement Biophysique + GDD accumulé + LAI Max]
    end
    
    subgraph Fusion par Machine Learning (Step 6)
        C1 & C2 & D1 & E1 & F3 --> G[step6_ml/yield_predictor.py]
        G -->|Random Forest Regressor| G1[Rendement Estimé Final + Intervalles de confiance]
    end
    
    subgraph Assistant Agronomique (Step 5)
        G1 & C1 & C2 & D1 & E1 --> H[step5_llm/llm_service.py]
        H -->|Gemini API Prompt Engineering| I[Conseiller Virtuel interactif en Darija]
    end
    
    I --> J[Tableau de Bord Cartographique UI]
```

---

## 3. Guide de Construction Étape par Étape

Le projet s'est construit de manière incrémentale en résolvant à chaque étape des problématiques d'intégration de données géospatiales et de biologie végétale :

### Étape 1 : Diagnostic de Maladie et Sévérité
*   **Objectif** : Identifier la pathologie présente sur la feuille et quantifier le niveau d'infection.
*   **Formule mathématique de Sévérité** :
    $$\text{Sévérité} = \frac{\text{Pixels Infectés}}{\text{Pixels Sains} + \text{Pixels Infectés}}$$
*   **Implémentation Code (HSV Masking Intelligent + Normalisation d'Exposition)** :
    ```python
    def _normalize_rgb_for_segmentation(img: Image.Image) -> np.ndarray:
        rgb = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
        lower, upper = np.percentile(rgb, (2, 98))
        rgb = np.clip((rgb - lower) / (upper - lower + 1e-8), 0.0, 1.0)
        return rgb

    def estimate_severity(img: Image.Image) -> float:
        rgb = _normalize_rgb_for_segmentation(img)
        hsv = np.array(np.round(rgb * 255.0), dtype=np.uint8)
        hsv = np.array(Image.fromarray(hsv, mode='RGB').convert('HSV'))

        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 1. Masque de feuille robuste : on exclut les zones trop sombres, trop claires
        #    et les zones neutres peu saturées (sol, fond, reflets parasites)
        leaf_mask = (s > 25) & (v > 35) & (v < 240)

        # 2. Vert sain : plage de teinte stricte pour limiter le bruit de fond
        healthy_mask = leaf_mask & (h >= 36) & (h <= 105) & (s >= 40)

        # 3. Tissus malades : image de terrain = zones foliaires visibles hors vert sain
        #    incluant teintes jaunâtres, brunes, rougâtres et taches de nécrose.
        diseased_mask = leaf_mask & (~healthy_mask)

        healthy_pixels = np.sum(healthy_mask)
        diseased_pixels = np.sum(diseased_mask)
        total = healthy_pixels + diseased_pixels

        if total <= 0:
            return 0.15

        severity = float(diseased_pixels / total)
        return float(np.clip(severity, 0.05, 0.95))
    ```
*   **Pourquoi cette amélioration est importante** :
    *   La normalisation RGB réduit l'impact des ombres, des reflets et de la sous-exposition sur l'image, ce qui améliore la stabilité de la segmentation.
    *   Le masque `leaf_mask` isole davantage la feuille réelle sur fond complexe (terre, pierres, zones non foliaires) sans dépendre uniquement de la sortie de la CNN.
    *   La sévérité est bornée entre $5\%$ et $95\%$ pour éviter les valeurs extrêmes de bruit sur les images de terrain.
    *   En cas d'image très dégradée ou mal focalisée, le système applique une valeur de repli rigoureuse de $15\%$ afin d'éviter un crash ou une estimation absurde.

### Étape 2 : Données du Sol et Propriétés Physiques (SoilGrids)
*   **Objectif** : Estimer le pH et les taux d'argile/sable.
*   **Formules hydrauliques de Pédotransfert (PTF)** :
    *   *Saturation* : $SM0 = 0.45 - 0.001 \times \text{Sable}$
    *   *Humidité au point de flétrissement* : $SMW = 0.02 + 0.0025 \times \text{Argile}$
    *   *Humidité à la capacité au champ* : $SMFCF = 0.1 + 0.003 \times \text{Argile} + 0.0005 \times (100 - \text{Sable} - \text{Argile})$
*   **Gestion des frontières (Balayage spatial)** :
    ```python
    # Recherche géospatiale si le point GPS tombe dans une ville ou sur le littoral (masque nul)
    offsets = [(-0.05, 0), (0.05, 0), (0, -0.05), (0, 0.05)] # ~5 km autour
    for dlat, dlon in offsets:
        res = fetch_from_api(lat + dlat, lon + dlon)
        if res is not None:
            return res
    ```

### Étape 3 : Télédétection de la Vigueur (NDVI Sentinel-2)
*   **Objectif** : Mesurer l'indice de végétation par différence normalisée de la parcelle.
*   **Formule mathématique** :
    $$\text{NDVI} = \frac{\text{B8 (NIR)} - \text{B4 (Rouge)}}{\text{B8} + \text{B4}}$$
*   **Implémentation GEE** :
    ```python
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterDate(date_start, date_end)
                  .filterBounds(buffer)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    ```

### Étape 4 : Simulation Biophysique et Météo de Saison (WOFOST)
*   **Objectif** : Alimenter WOFOST avec les données réelles du jour de semis jusqu'à aujourd'hui.
*   **Pipeline Météo** : Récupération quotidienne de la température, pluie, vitesse du vent et rayonnement solaire. Calcul de la pression de vapeur saturante (`VAP`) par la formule thermodynamique de Tetens :
    $$\text{VAP} = 6.112 \times \exp\left(\frac{17.67 \times T_{dew}}{T_{dew} + 243.5}\right)$$
*   **Calibration Rétroactive** :
    $$\text{Facteur de Calibration} = \frac{1}{N}\sum_{i=1}^N \frac{\text{Rendement Réel}_i}{\text{Rendement Simulé}_i}$$
    Ce coefficient (borné à $[0.5, 1.5]$) ajuste le rendement potentiel de la saison en cours.

### Étape 5 : Agronome Virtuel LLM (Gemini)
*   **Objectif** : Conseiller l'agriculteur en Darija ou en Français à partir du rapport de diagnostic.
*   **Prompt Engineering** : Le prompt configure Gemini pour agir en conseiller agricole local en traduisant les indicateurs (pH bas, présence de rouille foliaire) en solutions de traitement et d'irrigation adaptées, présentées en caractères latins (Arabizi) pour un style de chat accessible.

### Étape 6 : Fusion par Apprentissage Supervisé (Random Forest)
*   **Objectif** : Fusionner les features biophysiques et les observations réelles.
*   **Variables du modèle (Features)** :
    1.  `wofost_yield` : capture le rendement physique potentiel.
    2.  `ndvi` : capture la vigueur foliaire réelle sur pied.
    3.  `clay_pct`, `sand_pct`, `soil_ph` : caractérisent la texture et la composition chimique du sol.
    4.  `accumulated_temp` : capture le stress thermique et la phénologie (GDD).
    5.  `disease_severity` : capture le taux de destruction foliaire.

---

## 4. Choix de Conception et Débats Scientifiques

### Pourquoi coupler WOFOST (Biophysique) et Random Forest (Machine Learning) ?

Une question logique se pose : *Si le modèle de ML est entraîné sur des données générées par des équations physiques, pourquoi ne pas appliquer ces équations directement ?*

1.  **Limitation des Modèles Physiques (WOFOST)** : WOFOST est déterministe et thermodynamique. Il ne dispose d'aucune variable permettant d'injecter des signaux empiriques externes comme "80% de feuilles malades détectées par photo" ou "NDVI satellite de 0.45". Le Random Forest sert de **moteur de fusion empirique** capable de lier ces variables non physiques avec le potentiel physique simulé.
2.  **La Stratégie de Bootstrap (Démarrage)** : L'entraînement initial sur 5 000 exemples synthétiques sert à **pré-calibrer** le modèle de ML. Il apprend les lois physiques sous-jacentes (ex: la corrélation positive entre NDVI et rendement, la pénalité d'un pH acide, l'impact négatif d'une infection sévère).
3.  **L'Apprentissage Continu (Monde Réel)** : Dès que l'application collecte des **données réelles de rendement à la récolte** auprès des agriculteurs, le Random Forest est réentraîné sur ces vraies données. Le modèle ML s'émancipe des simplifications théoriques pour modéliser les **réalités complexes de terrain** (pratiques agricoles réelles, efficacité des engrais, micro-climats) non représentables par des équations physiques pures.

---

## 5. Limitations et Transparence (Honnêteté Technique)

Pour garantir la pérennité et la crédibilité scientifique du projet, voici les limites actuelles du système :

1.  **Entraînement sur Données Synthétiques** : Actuellement, le Random Forest est entraîné sur un jeu de données généré artificiellement. Bien que ce jeu suive des lois agronomiques réalistes (pénalités de pH, impact de la rouille sur la photosynthèse, influence de l'argile sur l'eau), il ne remplace pas un historique de mesures réelles. Il sert uniquement d'initialisation (bootstrap) en attendant de récolter de véritables données utilisateurs.
2.  **Limites du Modèle CNN (PlantVillage)** : Le CNN pour les maladies foliaires a été entraîné sur le dataset *PlantVillage*. Ce dataset contient des photos prises en laboratoire sur fond neutre uniforme. Dans les conditions réelles au champ (arrière-plan complexe contenant de la terre, des cailloux, des variations d'ombre et de lumière), la confiance du modèle de classification peut chuter. C'est pourquoi le masque de couleur HSV d'analyse de la sévérité est crucial : il isole la structure colorée de la feuille indépendamment de l'interprétation sémantique brute du CNN.
3.  **Absence native de la tomate dans WOFOST** : La librairie PCSE/WOFOST ne propose pas de variété de tomate par défaut dans son dictionnaire d'usine (`YAMLCropDataProvider`). Pour contourner cette limite, la culture de tomate est simulée avec les paramètres phénologiques de la betterave à sucre (`sugarbeet`) ou de la pomme de terre (`potato`), ce qui constitue une approximation biologique.
4.  **Dépendance à l'Authentification Google Earth Engine** : En l'état, l'application utilise l'authentification locale de la machine hôte. Dans un déploiement cloud réel (AWS, GCP), il est impératif de configurer un compte de service GCP doté de permissions spécifiques pour GEE afin de ne pas bloquer les requêtes des utilisateurs.
5.  **Découplage Frontend/Backend** : Actuellement, le fichier `main.py` fait office de serveur d'API et sert directement la page HTML/JS. Dans une architecture de production moderne, l'interface utilisateur devrait être séparée (hébergée sur Vercel/S3) et l'API hébergée séparément sur un service de conteneur (Docker, Cloud Run).

---

## 6. Difficultés Rencontrées et Solutions Implémentées

*   **PCSE/WOFOST Weather Gaps** : WOFOST plante si la météo a un trou ou si les dates débordent sur le futur.
    *   *Solution* : Limitation automatique de la fenêtre temporelle de la météo Open-Meteo à la date du jour (`today`) si la culture n'est pas encore récoltée.
*   **Bruits de fond sur les photos de feuilles** : La terre ou le feuillage mort faussent la vision par ordinateur et le calcul de sévérité.
    *   *Solution* : Le masque HSV isole le spectre de couleur vert/jaune/marron et exclut le noir/sombre et le blanc (surexposition) pour ne comparer que le rapport tissu malade/tissu sain sur le corps foliaire visible.
*   **APIs Tierces Non Répondantes** : Si l'API Google Earth Engine ou SoilGrids subit une interruption de service.
    *   *Solution* : Implémentation systématique de valeurs de repli (fallbacks) physiques cohérentes (NDVI moyen à $0.45$, sol par défaut sur Beni Mellal) pour garantir que l'application reste fonctionnelle.

---

## 7. Guide de Lancement et d'Installation

1.  Installez les dépendances :
    ```bash
    pip install fastapi uvicorn tensorflow numpy pillow pandas requests scikit-learn pcse
    ```
2.  Authentifiez Google Earth Engine :
    ```bash
    earthengine authenticate
    ```
3.  Lancez le serveur FastAPI :
    ```bash
    python main.py
    ```
4.  Ouvrez l'interface sur `http://127.0.0.1:8000`.

---

## 8. Améliorations Récentes et Optimisations Techniques

Dans le cadre de l'amélioration continue de l'expérience utilisateur et de la robustesse agronomique du pipeline, plusieurs modifications majeures ont été apportées :

### 8.1. Résolution des bugs de démarrage et de type
*   **Correction du NameError (Optional)** : Résolution d'un crash au chargement de l'API FastAPI dans `main.py` où le type de donnée `Optional` n'était pas défini. Importation systématique depuis le module natif `typing`.

### 8.2. Robustesse du Cycle de Simulation Biophysique (WOFOST)
*   **Sécurisation de la fenêtre temporelle** : Auparavant, si l'utilisateur ne spécifiait pas de date de semis ou si le semis était trop récent (ex: le jour même), la date de récolte estimée était bridée à la date courante (`today`), entraînant un cycle de simulation de 0 jour. Cela provoquait un plantage critique de WOFOST (`crop_end_date before or equal to crop_start_date`), bypassant l'étape de calibration historique et faussant la prédiction finale du Random Forest (rendement WOFOST de `0.0`).
*   **Solution de décalage adaptatif** : Dans [simulation_runner.py](file:///C:/Users/hp/OneDrive/Desktop/terroirai/step4_simulation/simulation_runner.py), une vérification automatique applique un décalage minimum de **30 jours** vers le passé si l'intervalle simulé est trop court, garantissant que la simulation biophysique s'exécute toujours et que le facteur de calibration est correctement appliqué.

### 8.3. Chatbot Agronome Interactif & Remplissage Intelligent (Intake)
Le chatbot a été transformé d'une simple boîte de dialogue post-diagnostic en un **assistant conversationnel d'intégration** :
*   **Accessibilité Immédiate (Split UI)** : Le chat virtuel est désormais affiché dans sa propre carte persistante dès le chargement de la page, tandis que le tableau de bord des diagnostics reste en attente de traitement.
*   **Extraction Automatique par Gemini** : Le point d'accès `/chat` dans `main.py` utilise une nouvelle fonction `ask_agronomist_interactive` de [llm_service.py](file:///C:/Users/hp/OneDrive/Desktop/terroirai/step5_llm/llm_service.py) configurée pour renvoyer un format structuré JSON contenant les paramètres identifiés au fil de la discussion (type de culture, date de semis, localisation, rendements historiques).
*   **Modifications dynamiques de la page** :
    *   **Autofill des Inputs** : Dès que l'IA détecte une information (ex: `"l9am7"` ou `"seme le 12 mai"`), le dropdown de la culture ou le champ de date sur la gauche est mis à jour instantanément, accompagné d'un effet visuel (flash de surbrillance vert).
    *   **Géocodage Géospatial OSM** : Si le fermier mentionne un lieu au Maroc (ex: *"Béni Mellal"*, *"Afourer"*), le JavaScript lance en arrière-plan une requête de géocodage Nominatim (OpenStreetMap), déplace le marqueur de la carte Leaflet et configure automatiquement les coordonnées GPS.
    *   **Lancement Automatisé (Auto-Submit)** : Si toutes les variables (culture, semis, localisation) sont renseignées et que l'utilisateur a sélectionné sa photo de feuille, le chat informe l'utilisateur et lance automatiquement la prédiction sans besoin de cliquer sur le bouton manuel.

### 8.4. Architecture Multi-Modèles de Diagnostic Végétal
Pour améliorer la précision et la couverture agronomique du diagnostic sanitaire, TerroirAI intègre désormais un système multi-modèles dynamique :
*   **Lazy Loading (Chargement paresseux)** : Afin de préserver la mémoire vive du serveur et d'accélérer le démarrage de l'API FastAPI, les modèles ne sont pas chargés tous en même temps. Chaque modèle (ex: le modèle olivier ou agrumes) est importé et compilé en cache uniquement au moment où le premier diagnostic de cette culture est demandé par un utilisateur.

### 8.5. Calibration Locale, Variables Agronomiques Étendues et Sortie Confiance-Aware
Pour rendre la prédiction plus réaliste et plus transparente pour l'utilisateur, le pipeline a été enrichi d'un niveau d'intelligence agronomique plus fin :

*   **Entrées agronomiques enrichies** : le point d'entrée `/predict` accepte désormais des variables supplémentaires qui améliorent la qualité de la fusion finale :
    *   `soil_moisture_pct`
    *   `season_rainfall_mm`
    *   `growth_window_avg_temp`
    *   `vegetative_stage`
    *   `irrigation_level`
    *   `fertilization_level`
    *   `variety`
  Ces variables sont normalisées dans [step6_ml/yield_predictor.py](step6_ml/yield_predictor.py) avant d'être injectées dans le Random Forest, ce qui permet d'aligner la prédiction sur le contexte réel de la parcelle.

*   **Calibration locale et pondérée** : la simulation WOFOST calcule maintenant un **facteur de calibration local** à partir des rendements historiques disponibles et de la zone géographique. Cette calibration est ajustée selon :
    *   la fiabilité du sol local,
    *   le biais spécifique à la culture,
    *   l'historique de rendement de la zone,
    *   la cohérence des ratios simulés / observés.
  Le facteur est borné dans une plage raisonnable pour garantir une stabilité numérique et éviter les sur-corrections.

*   **Niveau de confiance explicite** : la sortie expose maintenant une ou plusieurs informations de confiance au niveau de la parcelle :
    *   `photo_quality_score`
    *   `ndvi_signal_quality`
    *   `wofost_calibration_factor`
    *   `wofost_calibration_confidence`
    *   `final_prediction_confidence`
  Cela permet de distinguer le niveau de fiabilité de :
    1. la qualité de la photo d'entrée,
    2. la qualité du signal satellite,
    3. la calibration biophysique WOFOST,
    4. la fusion finale de rendement.

*   **Amélioration de la précision de détection de maladie** : la sévérité foliaire a été rendue plus stable par :
    *   normalisation RGB pour réduire la variance liée aux ombres, à l'exposition et aux reflets,
    *   masquage HSV plus robuste sur les feuilles réelles,
    *   exclusion des pixels trop sombres, trop clairs ou peu saturés,
    *   bornage de la sévérité pour éviter les extrêmes de bruit sur les captures terrain.

*   **Pourquoi c'est utile en pratique** : cette évolution transforme la sortie de l'API d'un simple rendement brut en un rapport de décision agronomique plus interprétable. L'utilisateur peut désormais voir si la prédiction est fiable parce que la photo est bonne, le signal satellite stable, la calibration locale cohérente et le modèle de fusion confiant.

### 8.6. Sortie de l'API enrichie
La réponse JSON retournée par `/predict` expose désormais explicitement l'ensemble des métadonnées utiles pour le diagnostic et la prise de décision, avec un contrat de sortie plus lisible pour le front-end et les intégrations externes :

```json
{
  "parcel_id": "...",
  "crop_type": "tomato",
  "disease_class": "Leaf Mold",
  "disease_confidence": 0.82,
  "disease_severity": 0.24,
  "photo_quality_score": 0.76,
  "ndvi": 0.58,
  "ndvi_signal_quality": 0.71,
  "wofost_yield_kg_ha": 3200.0,
  "wofost_calibration_factor": 1.08,
  "wofost_calibration_confidence": 0.82,
  "final_yield_estimate": 2940.0,
  "confidence_interval_low": 2380.0,
  "confidence_interval_high": 3510.0,
  "final_prediction_confidence": 0.81
}
```

Cette sortie permet d'éviter l'ambiguïté sur le niveau de confiance réel de la recommandation agronomique et facilite le développement d'une interface de suivi plus explicite pour l'agriculteur.
*   **Support Linguistique et Normalisation** : L'API résout et normalise les requêtes de cultures en français et en anglais (avec ou sans accents) pour cibler le bon modèle de classification.
*   **Amélioration de la Détection de Sévérité sur Photos de Terrain** : La mesure de sévérité n'est plus fondée uniquement sur une estimation brute par HSV. Elle passe désormais par une normalisation d'exposition (`_normalize_rgb_for_segmentation`) suivie d'une conversion en espace HSV robuste. Cela réduit considérablement les erreurs dues aux ombres, aux reflets et à la variation lumineuse sur les feuilles observées en conditions réelles.
*   **Modèles Intégrés et Mappings** :
    *   **Modèle Général (PlantVillage)** : 38 classes (Apple, Corn, Grape, Orange, Peach, Pepper bell, Potato, Strawberry, Squash, Tomato, etc.).
    *   **Modèle Olivier (OliveLeaf)** : 3 classes (`Bacterial leaf spot`, `Peacock spot`, `healthy`).
    *   **Modèle Agrumes (citrus_leaf)** : 9 classes (`Anthracnose`, `Bacterial Blight`, `Citrus Canker`, `Curl Virus`, `Deficiency Leaf`, `Dry Leaf`, `healthy`, `Sooty Mould`, `Spider Mites`).
    *   **Modèle Céréales (weatdiseases)** : 15 classes (`Aphid`, `Mite`, `Stem Fly`, `Black Rust`, `Brown Rust`, `Yellow Rust`, `Smut`, `Common Root Rot`, `Leaf Blight`, `Wheat Blast`, `Fusarium Head Blight`, `Septoria Leaf Blotch`, `Spot Blotch`, `Tan Spot`, `healthy`).
*   **Cohérence des Fichiers et Dossiers** : Le système utilise les chemins d'accès relatifs dynamiques et gère de manière transparente les divergences de nommage physiques (comme la structure `models/weatdiseases/wheat_diseases.h5` ou `models/citrus_leaf/citrus_leaf_p1.h5`).

### 8.5. Modélisation de l'Olivier (Estimation de Rendement sans Drone)
PCSE/WOFOST ne modélise pas nativement l'olivier (qui est une culture pérenne sujette à l'alternance de rendement biennale). TerroirAI pallie cette limite par un modèle agronomique biophysique-empirique intelligent basé sur des données physiologiques déclarées :
1.  **Formule de Rendement par Arbre** :
    $$\text{Rendement} = \text{Rendement de Base (Âge)} \times \text{Facteur Taille} \times \text{Facteur Cycle (Alternance)} \times \text{Facteur Variété} \times \text{Facteur NDVI} \times \text{Facteur Maladie}$$
2.  **Paramètres Physiologiques et Proxies** :
    *   *Âge des Arbres* : Jeunes ($8\text{ kg/arbre}$), Matures ($40\text{ kg/arbre}$), Vieux ($25\text{ kg/arbre}$).
    *   *Volume / Taille* : Petit ($0.7$), Moyen ($1.0$), Grand ($1.3$).
    *   *Cycle biennal (Alternance de charge)* : $0.6$ (si l'année passée était productive $\rightarrow$ repos cette année) ou $1.4$ (si l'année passée était en repos $\rightarrow$ charge cette année).
    *   *Variété* : Picholine Marocaine ($1.1$) ou autre ($1.0$).
3.  **Correction Satellite (NDVI)** :
    *   Le NDVI mesuré par Sentinel-2 sert de proxy pour la densité foliaire et le stress hydrique. Un NDVI inférieur à $0.7$ applique une pénalité progressive ($\text{Facteur NDVI} = \min(1.0, \frac{\text{NDVI}}{0.7})$).
4.  **Correction Diagnostic Maladie** :
    *   La présence de l'oeil de paon (`Peacock spot`) détectée sur la photo de feuille réduit le rendement proportionnellement à la sévérité ($\text{Facteur Maladie} = 1.0 - 0.75 \times \text{Sévérité}$).
5.  **Rendement en Huile** :
    *   En plus du rendement total en olives (kg), TerroirAI estime automatiquement le rendement en huile d'olive brute basé sur un taux moyen d'extraction régional de $20\%$.
## 9. Améliorations récentes documentées
Cette section résume les améliorations opérationnelles et d'expérience utilisateur apportées récemment dans le projet.

### 9.1. Auto-remplissage du formulaire via `/chat`
*   Le chatbot agronome virtuel peut maintenant extraire des informations de l'utilisateur et remplir automatiquement le formulaire de prédiction.
*   Exemple de données extraites : `crop_type`, `location`, `sowing_date`, `historical_yields`.
*   Si tous les champs nécessaires sont remplis par le chat et qu'une photo est téléchargée, la prédiction se lance automatiquement.
*   Le endpoint `/chat` retourne désormais un JSON structuré avec `response` et `extracted_data`.

### 9.2. Détection de champs manquants et arrière-plan dégradé
*   Le service LLM est capable de demander explicitement les informations manquantes une par une au fermier.
*   Si Gemini n'est pas disponible, le système bascule en mode dégradé et continue d'offrir une aide basée sur les informations déjà présentes.
*   Les messages du chatbot sont formulés de manière simple pour les agriculteurs marocains, avec un style Darija/Français accessible.

### 9.3. Suivi des étapes et rétroaction visible dans l'interface
*   L'interface web dispose maintenant d'un wizard pas-à-pas mobile-friendly en 5 étapes : culture, photo, localisation, date de semis, lancement de l'analyse.
*   Une barre de progression et des cartes d'étapes montrent l'état d'avancement de la saisie et du traitement.
*   Le tableau de bord affiche désormais les stages de traitement (`process_stages`) et des notifications de fallback (`fallback_notice`).

### 9.4. Gestion explicite des modes dégradés pour les services externes
*   `step2_soil/soil_api.py` expose maintenant explicitement le `source` du sol et le flag `degraded` pour SoilGrids.
*   `step3_satellite/satellite_ndvi.py` gère mieux les erreurs GEE en retournant un NDVI par défaut et un marqueur de dégradé.
*   Le backend enrichit la réponse `/predict` avec un objet `services` détaillant l'état de `soil`, `ndvi`, `wofost` et `ml`.
*   Les messages d'alerte sont plus transparents : l'utilisateur sait quand la prédiction est basée sur des valeurs par défaut ou quand la confiance est réduite.

### 9.5. Robustesse et validation
*   Le code a été compilé et vérifié sans erreur pour `main.py` et `step5_llm/llm_service.py`.
*   Un test direct du flux `/chat` a démontré que la structure JSON fonctionne et que le traitement extrait correctement `crop_type` et `location`.

### 9.6. Expérience d'utilisation
*   Le système propose un message d'accueil clair, un chat persistant et un dashboard de résultats visible dès le chargement.
*   La page guide l'utilisateur étape par étape, avec des suggestions de saisie et des boutons dédiés pour les coordonnées de test (ex: Béni Mellal).
*   Le service supporte l'affichage bilingue et simplifié pour des agriculteurs qui préfèrent le français ou l'arabe local.


