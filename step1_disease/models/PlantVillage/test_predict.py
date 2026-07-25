import os
import numpy as np
from PIL import Image
import tensorflow as tf

def test_prediction():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model.h5')
    data_dir = os.path.join(script_dir, '..', '..', 'PlantVillageData')
    
    print("Chargement du modèle...")
    if not os.path.exists(model_path):
        print(f"Erreur : Le modèle '{model_path}' n'existe pas.")
        return
        
    model = tf.keras.models.load_model(model_path)
    print("Modèle chargé avec succès.")
    
    # Liste complète des 38 classes de PlantVillage dans l'ordre alphabétique standard
    classes = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites_Two_spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_YellowLeaf___Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]
    
    print(f"Classes configurées ({len(classes)}).")
    
    # Trouver une image de test à partir des dossiers existants
    if not os.path.exists(data_dir):
        print(f"Erreur : Le dossier de données '{data_dir}' n'existe pas.")
        return
        
    local_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != 'PlantVillage'])
    test_image_path = None
    true_class = None
    
    for c in local_folders:
        class_folder = os.path.join(data_dir, c)
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            test_image_path = os.path.join(class_folder, images[0])
            # Normaliser le nom du dossier pour correspondre avec la liste de classes si nécessaire
            true_class = c
            break
            
    if not test_image_path:
        print("Erreur : Aucune image de test trouvée.")
        return
        
    print(f"Image sélectionnée pour le test : {test_image_path} (Classe réelle : {true_class})")
    
    # Prétraitement de l'image
    img = Image.open(test_image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prédiction
    print("Exécution de la prédiction...")
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_idx]
    
    predicted_class = classes[predicted_idx] if predicted_idx < len(classes) else "Inconnue"
    
    def normalize(name):
        return "".join([c for c in name.lower() if c.isalnum()])
        
    print("\n--- RÉSULTAT DU TEST ---")
    print(f"Classe réelle : {true_class}")
    print(f"Classe prédite : {predicted_class} (Index: {predicted_idx})")
    print(f"Confiance : {confidence:.2%}")
    if normalize(predicted_class) == normalize(true_class):
        print("Résultat : SUCCÈS - La prédiction est correcte !")
    else:
        print("Résultat : DIFFÉRENT (Peut arriver selon le niveau d'entraînement)")

if __name__ == "__main__":
    test_prediction()
