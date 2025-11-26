#!/usr/bin/env python3

import numpy as np

pca = __import__('0-pca').pca

# Charger les données depuis les fichiers texte en ignorant les en-têtes XML
def load_data_skip_xml(filename):
    """Charge les données depuis un fichier en ignorant les en-têtes XML"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Trouver où commencent les données réelles (ignorer les lignes XML)
    data_lines = []
    for line in lines:
        line = line.strip()
        # Ignorer les lignes vides et les lignes XML
        if line and not line.startswith('<') and not line.startswith('<?'):
            data_lines.append(line)
    
    return data_lines

# Charger et analyser les données
X_lines = load_data_skip_xml('mnist2500_X.txt')
X = np.array([list(map(float, line.split())) for line in X_lines])

labels_lines = load_data_skip_xml('mnist2500_labels.txt')
labels = np.array([int(line) for line in labels_lines])

# Standardiser les données
X_m = X - np.mean(X, axis=0)

# Appliquer PCA
W = pca(X_m)

# Projeter les données sur les composantes principales
T = np.matmul(X_m, W)

# Reconstruire les données
X_t = np.matmul(T, W.T)

# Calculer l'erreur de reconstruction
m = X.shape[0]
reconstruction_error = np.sum(np.square(X_m - X_t)) / m

print(f"Forme des données: {X.shape}")
print(f"Forme des composantes principales: {W.shape}")
print(f"Forme des données projetées: {T.shape}")
print(f"Erreur de reconstruction: {reconstruction_error}")