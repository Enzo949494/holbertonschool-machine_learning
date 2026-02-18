# QA Bot - Question Answering System

## Description

QA Bot est un système de réponse aux questions basé sur l'IA qui utilise des modèles BERT et la recherche sémantique pour extraire des réponses pertinentes à partir de documents de référence.

## Caractéristiques

- **Modèle BERT**: Utilise un modèle BERT fine-tuné sur le dataset SQuAD pour l'extraction de réponses
- **Recherche sémantique**: Trouve le document le plus pertinent avant d'extraire la réponse
- **Fallback intelligent**: Utilise une recherche par mots-clés en cas d'échec du modèle BERT
- **Post-traitement** : Nettoie et formate les réponses pour une meilleure qualité

## Structure du projet

```
qa_bot/
├── 0-qa.py                 # Modèle BERT pour l'extraction de réponses
├── 1-loop.py               # Boucle interactive simple (template de base)
├── 2-qa.py                 # Boucle de questions sur un texte simple
├── 3-semantic_search.py    # Recherche sémantique sur corpus de documents
├── 4-qa.py                 # QA Bot complet avec recherche sémantique et fallback
├── corpus/                 # Dossier contenant les documents de référence (.md)
└── README.md              # Ce fichier
```

## Installation

### Prérequis
- Python 3.11+
- TensorFlow
- Transformers
- NumPy

### Dépendances

```bash
pip install tensorflow tensorflow-hub transformers numpy
```

## Utilisation

### 1. Utilisation simple (0-qa.py + 2-qa.py)

Répondre à des questions sur un texte de référence direct:

```python
from qa_bot import answer_loop

reference = "Your reference text here..."
answer_loop(reference)
```

### 2. Utilisation avancée (4-qa.py)

Utiliser le QA Bot complet avec recherche sémantique sur un corpus:

```bash
python3 4-main.py
```

Puis saisir vos questions:
```
Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm
```

## Fichiers principaux

### 0-qa.py
**Fonction**: `question_answer(question, reference)`
- Prend une question et un texte de référence
- Retourne un snippet du texte qui répond à la question
- Utilise le modèle BERT fine-tuné sur SQuAD

### 3-semantic_search.py
**Fonction**: `semantic_search(corpus_path, sentence)`
- Effectue une recherche sémantique sur un corpus de documents
- Retourne le contenu du document le plus similaire
- Utilise le modèle "all-MiniLM-L6-v2" pour les embeddings

### 4-qa.py
**Fonction**: `question_answer(corpus_path)`
- Boucle interactive QA Bot
- Combine recherche sémantique + extraction BERT
- Utilise un fallback intelligent en cas d'échec

## Exemple d'utilisation

```bash
# Lancer le QA Bot
python3 4-main.py

# Poser des questions
Q: What are PLDs?
A: peer learning days

Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm

Q: What are Mock Interviews?
A: help you train for technical interviews

Q: EXIT
A: Goodbye
```

## Format du corpus

Les documents doivent être au format Markdown (.md) dans un `dossier/`:

```
dossier/
├── document1.md
├── document2.md
└── document3.md
```

## Modèles utilisés

- **BERT QA**: `bert-large-uncased-whole-word-masking-finetuned-squad`
  - Source: TensorFlow Hub
  - Fine-tuné sur le dataset SQuAD

- **Sentence Encoder**: `sentence-transformers/all-MiniLM-L6-v2`
  - Léger et efficace pour la recherche sémantique
  - Embedding de 384 dimensions

## Performance

- **Temps de réponse**: ~2-5 secondes (incluant le chargement des modèles)
- **Longueur max des réponses**: Sans limite artificielle
- **Qualité**: Dépend de la pertinence du corpus et de la clarté de la question
