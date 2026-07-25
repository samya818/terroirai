# Plan de Correction - TerroirAI Frontend

## Problème 1 : Boutons de navigation manquants dans le formulaire

Les étapes 1 à 4 n'ont aucun bouton "Suivant" / "Précédent" pour valider et naviguer.
Actuellement, seuls les "step pills" en haut sont cliquables, ce qui n'est pas intuitif.

**Actions :**
- [x] Ajouter un bouton "Suivant →" / "← التالي" aux étapes 1, 2, 3, 4
- [x] Ajouter un bouton "← Précédent" / "السابق →" aux étapes 2, 3, 4, 5
- [x] Ajouter les clés de traduction `nextBtn`, `prevBtn` dans l'objet `translations`

## Problème 2 : Traductions arabes incomplètes

### 2a. Clés data-i18n utilisées dans le HTML mais absentes de l'objet `translations`
- `cropHint` (étape 1)
- `photoHint` (étape 2)
- `sowingHint` (étape 4)
- `readyHint` (étape 5)
- `backBtn` (bouton Retour étape 5)

### 2b. Éléments HTML avec texte en français sans attribut `data-i18n`
- Titres et sous-titres des 5 étapes du wizard
- Labels des "step pills" (1. Culture, 2. Photo, etc.)
- Texte du bouton recherche 🔍
- Placeholders des champs de rendement historique
- Texte d'aide dans `goToStep()` (déjà bilingue mais pas via translations)

### 2c. Texte statique dans `goToStep()` - utiliser les traductions
- Le helper text en bas est mis à jour par la fonction `goToStep()` - déjà en bilingue

## Actions détaillées :

### HTML - Navigation buttons
- Étape 1 : Ajouter `button-row` avec bouton "Suivant → / التالي ←"
- Étape 2 : Ajouter `button-row` avec "← Précédent / السابق →" et "Suivant → / التالي ←"
- Étape 3 : Ajouter `button-row` avec "← Précédent / السابق →" et "Suivant → / التالي ←"
- Étape 4 : Ajouter `button-row` avec "← Précédent / السابق →" et "Suivant → / التالي ←"
- Étape 5 : Remplacer le bouton "Retour" simple par un "← Précédent / السابق →" avec data-i18n

### HTML - Ajouter data-i18n aux éléments manquants
- Titres/sous-titres : `step1Title`, `step1Subtitle`, etc.
- Step pills : `stepPill1` à `stepPill5`
- Bouton recherche : `searchBtn`
- Placeholders : `yearPlaceholder`, `yieldPlaceholder`
- Ajouter les clés manquantes : `cropHint`, `photoHint`, `sowingHint`, `readyHint`, `backBtn`

### Translations - Ajouter toutes les clés
- Ajouter ~25 nouvelles clés dans `translations.fr` et `translations.ar`
- Vérifier la cohérence des traductions arabes

