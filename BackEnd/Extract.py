#dans mon dossier img, j'ai un dossier WithLabel et un dossier WithoutLabel
#WithLabel contient deux dossiers : Clean et Dirty
import os
from PIL import Image
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt

#Preprocessus de l'image pour s'assurer qu'elle est au format RGB
#et pour la convertir en tableau numpy
def preprocess_image(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

# Fonction pour traiter les images dans un répertoire donné
def process_images_in_directory(directory):
    features_list = []
    # Parcours des sous-dossiers et des images
    for subdir in ['WithLabel/Clean', 'WithLabel/Dirty', 'WithoutLabel']:
        full_path = os.path.join(directory, subdir)
        if os.path.exists(full_path):
            image_paths = glob(os.path.join(full_path, '*.*'))
            for img_path in image_paths:
                try:
                    img = preprocess_image(img_path)
                    features = extract_features(img, path=img_path)
                    features['path'] = img_path  # Ajoute le chemin de l'image aux features
                    features_list.append(features)
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {img_path}: {e}")
    return features_list

# Cette fonction extrait les caractéristiques d'une image et retourne un dictionnaire de features
def extract_features(image, path=None):
    features = {}
    # Redimensionne pour homogénéiser les features
    #img = cv2.resize(img, (128, 128))
    # Dimension de l'image
    height, width = image.shape[:2]
    features['width'] = width
    features['height'] = height
    # Moyenne et écart-type par canal
    features['mean'] = np.mean(image, axis=(0,1))
    features['std'] = np.std(image, axis=(0,1))
    # Contraste global
    features['contrast'] = image.std()
    # Histogramme des couleurs
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        features[f'hist_{color}'] = hist.flatten().tolist()  # Convertit en liste pour JSON
    # Histogramme de luminance
    luminance = 0.3 * image[:, :, 0] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 2]
    hist_luminance = cv2.calcHist([luminance.astype(np.uint8)], [0], None, [256], [0, 256])
    features['hist_luminance'] = hist_luminance.flatten().tolist()
    # Détection des contours avec Canny
    edges = cv2.Canny(image, 100, 200)
    features['edges'] = edges.flatten().tolist()  # Convertit en liste pour JSON
    # Median par canal
    features['median'] = np.median(image, axis=(0,1))
    # Min/Max par canal
    features['min'] = np.min(image, axis=(0,1))
    features['max'] = np.max(image, axis=(0,1))
    # Taille du fichier
    if path:
        features['file_size'] = os.path.getsize(path)
    else:
        features['file_size'] = 0
    return features

# Fonction pour créer un fichier csv avec les caractéristiques extraites
def save_features_to_csv(features_list, output_file='features.csv'):
    import pandas as pd
    if not features_list:
        print("Aucune caractéristique à sauvegarder.")
        return
    
    # Convertit la liste de dictionnaires en DataFrame
    df = pd.DataFrame(features_list)
    
    # Sauvegarde le DataFrame dans un fichier CSV
    df.to_csv(output_file, index=False)
    print(f"Caractéristiques sauvegardées dans {output_file}")

