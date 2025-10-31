# Object Detection with YOLO v3

## 📋 Description

Ce projet implémente un système de **détection d'objets** en utilisant l'algorithme **YOLO v3** (You Only Look Once version 3). YOLO est un réseau de neurones convolutif capable de détecter et classifier plusieurs objets dans une image en temps réel.

Le projet utilise le modèle **Darknet** pré-entraîné sur le dataset **COCO** (Common Objects in Context) qui peut reconnaître 80 classes d'objets différents (personnes, animaux, véhicules, objets du quotidien, etc.).

---

## 🎯 Objectifs

Apprendre à :
- Charger et utiliser un modèle YOLO pré-entraîné
- Traiter les sorties brutes d'un réseau de neurones
- Filtrer les prédictions avec des seuils de confiance
- Appliquer la suppression non-maximale (NMS)
- Afficher et sauvegarder les résultats de détection

---

## 🛠️ Technologies

- **Python 3.x**
- **TensorFlow / Keras** : Chargement et utilisation du modèle
- **NumPy** : Calculs matriciels
- **OpenCV (cv2)** : Traitement d'images et affichage

---

## 📁 Structure du projet

```
object_detection/
│
├── 0-yolo.py          # Initialisation de la classe Yolo
├── 1-yolo.py          # Traitement des sorties du modèle
├── 2-yolo.py          # Filtrage des boîtes par seuil
├── 3-yolo.py          # Suppression non-maximale (NMS)
├── 4-yolo.py          # Chargement d'images depuis un dossier
├── 5-yolo.py          # Prétraitement des images
├── 6-yolo.py          # Affichage des boîtes de détection
├── 7-yolo.py          # Pipeline complet de prédiction
│
├── yolo.h5            # Modèle YOLO pré-entraîné
├── coco_classes.txt   # Liste des 80 classes COCO
│
├── yolo_images/       # Dossier contenant les images de test
│   └── yolo/
│       ├── dog.jpg
│       ├── eagle.jpg
│       ├── giraffe.jpg
│       ├── horses.jpg
│       ├── person.jpg
│       └── takagaki.jpg
│
└── detections/        # Dossier où sont sauvegardées les détections
```

---

## 🚀 Fonctionnalités

### Exercice 0 : Initialisation
Création de la classe `Yolo` qui charge :
- Le modèle Darknet (`.h5`)
- Les noms des classes (`.txt`)
- Les paramètres de filtrage

### Exercice 1 : Traitement des sorties
Conversion des sorties brutes du réseau en :
- Coordonnées de boîtes `[x1, y1, x2, y2]`
- Scores de confiance
- Probabilités de classe

### Exercice 2 : Filtrage des boîtes
Élimination des détections avec un score de confiance trop faible.

### Exercice 3 : Suppression Non-Maximale (NMS)
Élimination des boîtes redondantes qui se chevauchent trop (même objet détecté plusieurs fois).

### Exercice 4 : Chargement d'images
Chargement de toutes les images d'un dossier avec OpenCV.

### Exercice 5 : Prétraitement
Redimensionnement et normalisation des images pour le modèle :
- Resize à 416×416 pixels (interpolation cubique)
- Normalisation des pixels entre [0, 1]

### Exercice 6 : Affichage des détections
Affichage des images avec :
- Rectangles bleus autour des objets
- Labels rouges (classe + score de confiance)
- Sauvegarde en appuyant sur la touche 's'

### Exercice 7 : Pipeline complet
Détection automatique sur toutes les images d'un dossier.

---

## 📊 Format des données

### Sorties du modèle YOLO

Pour chaque image, YOLO retourne un **tuple de 3 éléments** :

```python
(boxes, box_classes, box_scores)
```

#### 1. `boxes` : Coordonnées des boîtes
```python
array([[x1, y1, x2, y2],  # Boîte 1
       [x1, y1, x2, y2],  # Boîte 2
       ...])
```
- `x1, y1` : Coin supérieur gauche
- `x2, y2` : Coin inférieur droit

#### 2. `box_classes` : Indices des classes
```python
array([1, 7, 16])  # bicycle, truck, dog
```

#### 3. `box_scores` : Scores de confiance
```python
array([0.995, 0.914, 0.998])  # 99.5%, 91.4%, 99.8%
```

---

## 🎨 Classes COCO

Le modèle peut détecter **80 classes** d'objets :

```
0: person          16: dog            32: sports ball
1: bicycle         17: horse          33: kite
2: car             18: sheep          ...
3: motorcycle      19: cow            79: toothbrush
4: airplane        20: elephant
5: bus             21: bear
6: train           22: zebra
7: truck           23: giraffe
...
```

*(Liste complète dans `coco_classes.txt`)*

---

## 💻 Utilisation

### Installation des dépendances

```bash
pip install tensorflow opencv-python numpy
```

### Exécution

```bash
# Tester la détection sur toutes les images
./7-main.py

# Résultat : affiche chaque image avec les détections
# Appuyez sur 's' pour sauvegarder, autre touche pour continuer
```

### Exemple de code

```python
import numpy as np
from yolo import Yolo

# Initialisation
anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]])

yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)

# Prédiction sur un dossier d'images
predictions, image_paths = yolo.predict('yolo_images/yolo/')

# Affichage des résultats pour une image
print(predictions[0])
# (boxes, box_classes, box_scores)
```

---

## 🧮 Concepts clés

### 1. YOLO (You Only Look Once)
- Algorithme de détection d'objets en **temps réel**
- Divise l'image en grille et prédit les boîtes directement
- Une seule passe dans le réseau (rapide !)

### 2. Anchor Boxes
Boîtes de référence prédéfinies de différentes tailles et proportions pour détecter des objets de formes variées.

### 3. IoU (Intersection over Union)
Métrique pour mesurer le chevauchement entre deux boîtes :
```
IoU = Aire d'intersection / Aire d'union
```

### 4. NMS (Non-Maximum Suppression)
Élimine les détections multiples du même objet en gardant uniquement la boîte avec le meilleur score.

### 5. Seuils
- **class_t** (0.6) : Seuil minimum de confiance pour garder une détection
- **nms_t** (0.5) : Seuil IoU pour la suppression non-maximale

---

## 📈 Résultats attendus

Exemple sur `dog.jpg` :

```python
boxes = [[119.10, 118.64, 567.89, 440.59],  # bicycle
         [468.68,  84.48, 695.97, 168.01],  # truck
         [124.11, 220.44, 319.46, 542.40]]  # dog

box_classes = [1, 7, 16]
box_scores = [0.995, 0.914, 0.998]
```

**Interprétation :**
- 🚲 Vélo détecté avec 99.5% de confiance
- 🚚 Camion détecté avec 91.4% de confiance
- 🐕 Chien détecté avec 99.8% de confiance

---

## 🐛 Débogage

### Problème : Rectangles mal placés
- Vérifiez que vous utilisez les **bonnes images** (celles du ZIP fourni)
- Les dimensions doivent correspondre aux coordonnées hardcodées

### Problème : Erreur de dimensions
- Vérifiez l'ordre `input_w` et `input_h` (pas inversés)
- Pour YOLO : `shape[1]` = largeur, `shape[2]` = hauteur

### Problème : Images non trouvées
```bash
mkdir -p yolo_images/yolo
# Décompresser le ZIP d'images dans ce dossier
```

---

## 🎓 Compétences acquises

- ✅ Utilisation de modèles pré-entraînés
- ✅ Traitement des sorties de réseaux de neurones
- ✅ Algorithmes de filtrage et NMS
- ✅ Manipulation d'images avec OpenCV
- ✅ Pipeline complet de computer vision
- ✅ Gestion des fichiers et dossiers

---

