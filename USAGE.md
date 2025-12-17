# Guide d’utilisation du dépôt IES_DPE

## Règles générales

- Ne jamais travailler directement sur la branche `main`
- Chaque équipe travaille uniquement sur sa branche dédiée
- Chaque dossier correspond à une responsabilité précise
- Toute intégration vers `main` se fait via Pull Request

Branches officielles :
- `main` → version stable
- `analyse` → Data Analyse
- `science` → Data Science
- `viz` → Data Visualisation

---
## Installation de Git sous Windows

Git est indispensable pour cloner le dépôt et collaborer en équipe.

1. Télécharger Git

Télécharger Git pour Windows depuis le site officiel :
https://git-scm.com/download/win

Lancer l’installateur (`.exe`) une fois le téléchargement terminé.

---

2. Options d’installation recommandées

Lors de l’installation, conserver les options par défaut, en particulier :

- Éditeur par défaut : **Use Vim** (ou **Use Visual Studio Code** si installé)
- PATH : **Git from the command line and also from 3rd-party software**
- HTTPS transport backend : **OpenSSL**
- Line ending conversions : **Checkout Windows-style, commit Unix-style**
- Terminal emulator : **Use MinTTY (default)**
- Git Pull behavior : **Default (fast-forward or merge)**

Ces options garantissent une compatibilité maximale avec le projet.

---

### 3. Vérifier l’installation
N'oubliez pas de relancer le terminal utilisé pour être que sûr que **Git** soit bien installé.

Ouvrir **Git Bash**, **PowerShell**,  **Windows Terminal (cmd)** ou **Anaconda Prompt** (préférer l'utilisation de Anaconda Prompt), puis exécuter :

```bash
git --version
```

Résultat attendu :

```bash
git version 2.x.x
```

---

## Installation initiale (tous les groupes)

**Alternative recommandée : Utilisation de GitHub Desktop (interface graphique)**

```bash
git clone https://github.com/Gjuuuy/IES_DPE.git
cd IES_DPE
```

## Installation des dépendances

Ce projet utilise Python 3.10 pour avoir moins de bugs, plus de temps pour le modèle et l’interface.

### Création d’un environnement virtuel (recommandé)

```bash
conda create -n dpe python=3.10
````

Activation de l’environnement :

```bash
conda activate dpe
```

Après l'activation de l'environnemnent, vous devez avoir sur le terminal le résultat suivant : 

Avant : 
```bash
(base) C:\Chemin/de/ton/dossier/IES_DPE
```

Après : 
```bash
(dpe) C:\Chemin/de/ton/dossier/IES_DPE
```

### Installation des requirements

À la racine du projet `IES_DPE` :

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Vérification de l’installation

```bash
pip list
```

Les dépendances du projet doivent apparaître dans la liste.

Vérifier les branches :

```bash
git branch -a
```

---

## La bonne procédure (première fois uniquement) par branche

1. Récupérer les branches distantes

```bash
git fetch origin
```

Résultat attendu dans le terminal : 

```bash
main
remotes/origin/HEAD -> origin/main
remotes/origin/analyse
remotes/origin/main
remotes/origin/science
remotes/origin/viz
```

2. Créer la branche locale à partir de GitHub

Pour analyse :

```bash
git checkout -b analyse origin/analyse
```

Pour science :

```bash
git checkout -b science origin/science
```

Pour viz :

```bash
git checkout -b viz origin/viz
```

---

## Workflow standard

Se placer sur la branche de son équipe :

```bash
git checkout analyse   # ou science / viz
```

Mettre à jour avec `main` :

```bash
git pull origin main
```

Ajouter les fichiers :

```bash
git add .
```

Commit :

```bash
git commit -m "[TYPE] description courte"
```

Push :

```bash
git push origin analyse   # ou science / viz
```

---

## Workflow par équipe

### Data Analyse (`analyse`)

Dossiers autorisés :

* `analyse/` : données brutes (lecture seule)
* `data/processed/` : données nettoyées
* `data/features/` : données prêtes pour modèles

Commit type :

```bash
git commit -m "[EDA] analyse exploratoire des variables"
```

---

### Data Science (`science`)

Dossiers autorisés :

* `science/`
* `data/features/`

Commit type :

```bash
git commit -m "[MODEL] entraînement modèle régression"
```

---

### Data Visualisation (`viz`)

Dossiers autorisés :

* `viz/`

Commit type :

```bash
git commit -m "[VIZ] ajout graphiques avant/après rénovation"
```

---

## Pull Request vers `main`

* Aucune modification directe sur `main`
* Passer par GitHub → New Pull Request
* Branche source : `analyse`, `science` ou `viz`
* Branche cible : `main`
* Validation collective obligatoire

---

## Convention de commits

Format obligatoire :

```
[type] description courte
```

## Types de commits

- `[EDA]` : Analyse exploratoire des données (statistiques, distributions, corrélations, visualisations initiales).
- `[DATA]` : Manipulation des données (nettoyage, transformation, feature engineering, préparation des jeux de données).
- `[MODEL]` : Développement, entraînement, évaluation ou amélioration des modèles de machine learning.
- `[VIZ]` : Création ou amélioration des visualisations, graphiques, dashboards et restitutions visuelles.
- `[FRONT]` : Développement ou modification de l’interface utilisateur (front-end).
- `[DOC]` : Ajout ou mise à jour de la documentation (README, guides, méthodologie).
- `[FIX]` : Correction de bugs, erreurs, incohérences ou problèmes techniques.

Exemples :

```python 
git commit -m "[MODEL] classification DPE avec XGBoost"
git commit -m "[VIZ] ajout graphiques comparatifs DPE"
git commit -m "[DOC] mise à jour méthodologie"
```

---

## Gestion des données

* `data/raw` : données brutes (lecture seule)
* `data/processed` : données nettoyées
* `data/features` : données prêtes pour modèles

Les données brutes ne doivent jamais être modifiées.


## Commandes de secours (erreurs fréquentes)

1. Voir l’état des fichiers
```bash 
git status
```
2. Annuler un fichier ajouté par erreur

```bash 
git restore --staged nom_du_fichier
```

3. Revenir au dernier commit

```bash 
git restore .
```

4. Voir l’historique

```bash
git log --oneline
```