# ✈️ AeroPredict — Optimisation de profils aérodynamiques assistée par ML

> **MGA 802 · Introduction à la programmation avec Python**
> École de Technologie Supérieure (ÉTS) — Université du Québec · Session Été 2026
> Responsable du cours : **Ilyass Tabiai**

---

## Équipe

| Membre | Code permanent | Contribution principale |
|---|---|---|
| **Flavien Blanchard** | BLAF63330301 | Pipeline XFoil, architecture du réseau de neurones |
| **Milissa Mechref** | MECM76290102 | Validation, revue de code, dashboard Streamlit |
| **Vincent Condette** | CONV76330301 | Dataset géométrique, tests, documentation |

---

## But du projet

L'objectif principal est d'**identifier le profil aérodynamique maximisant la finesse CL/CD** (portance / traînée) pour des conditions de vol données (nombre de Reynolds, plage d'angles d'attaque), en exploitant un modèle de Machine Learning entraîné sur ~401 000 simulations XFoil.

La démarche suit quatre phases :

1. **Construction d'un dataset** géométrique à partir des familles NACA 4 chiffres et UIUC.
2. **Enrichissement aérodynamique** des profils via le solveur XFoil (CL, CD, CM).
3. **Entraînement d'un réseau de neurones multi-tâches** (TensorFlow) qui prédit simultanément CL, CD et CM.
4. **Validation** des prédictions contre des simulations CFD Ansys Fluent indépendantes.

---

## Architecture du pipeline

```
data_profils.py        →  calcul_Xfoil.py        →  modele_ML.py
(géométrie NACA/UIUC)     (simulations XFoil)       (entraînement NN)
        ↓                         ↓                         ↓
 dataset_profil.csv       dataset_aeroXfoil.csv     naca_multitask_model.keras
                                                     preprocessor.pkl
                                  ↓
                       Application_modele_ML.py
                       (inférence sur dataset complet)
                                  ↓
                    dataset_aeroXfoil_avec_predictions.csv
                                  ↓
                    dashboard.py  /  audit_dataset.py
                    (visualisation interactive + rapports PDF)
```

Toutes les étapes sont orchestrées par `main.py` via un **menu interactif**, configuré par le fichier `deck.yaml`.

---

## Structure des fichiers

```
AeroPredict/
├── main.py                                    # Menu interactif — point d'entrée
├── deck.yaml                                  # Configuration du pipeline (chemins, XFoil)
│
├── data_profils.py                            # Étape 1 : dataset géométrique (AeroSandbox)
├── calcul_Xfoil.py                            # Étape 2 : simulations XFoil parallèles
├── modele_ML.py                               # Étape 3 : préprocesseur + entraînement NN
├── Application_modele_ML.py                   # Étape 4 : inférence + export CSV
├── audit_dataset.py                           # Étape 5 : audit + rapports PDF
├── dashboard.py                               # Étape 6 : dashboard Streamlit interactif
│
├── requirements.txt                           # Dépendances Python
├── naca_multitask_model.keras                 # Modèle entraîné (pré-calculé)
├── preprocessor.pkl                           # Préprocesseur sérialisé (pré-calculé)
│
├── dataset_projet/
│   ├── dataset_profil.csv                     # Dataset géométrique (Étape 1)
│   ├── dataset_aeroXfoil.csv                  # Dataset enrichi XFoil (Étape 2)
│   └── dataset_aeroXfoil_avec_predictions.csv # Dataset + prédictions ML (Étape 4)
│   └── dataset_fluent_csv                     # Dataset de validation via FLUENT      
├              
|
|
│
└── tests/
    ├── conftest.py
    ├── test_data_profils.py
    ├── test_calcul_xfoil.py
    ├── test_modele_ml.py
    └── test_main.py
```

---

## Technologies utilisées

| Bibliothèque | Version | Rôle dans le projet |
|---|---|---|
| **Python** | ≥ 3.9, ≤ 3.13 | Langage principal (contrainte TensorFlow) |
| **NumPy** | — | Calculs vectoriels, manipulations de tableaux |
| **Pandas** | — | Chargement, fusion et filtrage des datasets CSV |
| **AeroSandbox** | — | Génération des géométries NACA/UIUC + wrapper XFoil |
| **XFoil** | (externe) | Solveur aérodynamique 2D (calcul CL, CD, CM) |
| **TensorFlow / Keras** | ≥ 2.16 | Entraînement et inférence du réseau multi-tâches |
| **scikit-learn** | — | Normalisation (`StandardScaler`) et encodage (`LabelEncoder`) |
| **Plotly** | — | Graphiques interactifs dans le dashboard |
| **Streamlit** | — | Dashboard web interactif (déployable localement) |
| **PyYAML** | — | Lecture du fichier de configuration `deck.yaml` |
| **Matplotlib / Seaborn** | — | Graphiques statiques dans les rapports d'audit |
| **pytest** | — | Suite de tests unitaires (43 tests) |
| **pickle** | stdlib | Sérialisation du préprocesseur entraîné |

---

## Prérequis

### 1. Python

Installer Python **3.9 à 3.13** (TensorFlow ne supporte pas Python 3.14+).

```bash
python --version
```

### 2. XFoil (requis pour les étapes 1 et 2 uniquement)

Télécharger XFoil depuis [https://web.mit.edu/drela/Public/web/xfoil/](https://web.mit.edu/drela/Public/web/xfoil/) et noter le chemin de l'exécutable.

> Le dashboard et les étapes 3-6 fonctionnent **sans XFoil** grâce aux données pré-calculées fournies.

---

## Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd AeroPredict

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Installer les dépendances de base
pip install -r requirements.txt

# 4. TensorFlow (étapes 3 et 4 uniquement)
pip install tensorflow>=2.16

# 5. AeroSandbox (étapes 1 et 2 uniquement)
pip install aerosandbox
```

---

## Configuration — `deck.yaml`

Avant de lancer le pipeline, vérifier ou adapter le fichier `deck.yaml` :

```yaml
chemins:
  dataset_geometrie: dataset_projet/dataset_profil.csv
  dataset_xfoil: dataset_projet/dataset_aeroXfoil.csv
  dataset_predictions: dataset_projet/dataset_aeroXfoil_avec_predictions.csv
  dataset_fluent: dataset_projet/dataset_fluent.csv
  modele: naca_multitask_model.keras
  preprocesseur: preprocessor.pkl
  rapports_audit: audit_reports

xfoil:
  executable: C:\Xfoil\xfoil.exe   # ← adapter selon votre installation
  max_coeurs: 8                     # ← nombre de cœurs CPU pour la parallélisation

mode_test: false   # true = limite à 10 profils pour un test rapide
```

---

## Utilisation

### Option A — Menu interactif (pipeline complet)

```bash
python main.py
# ou avec un deck personnalisé :
python main.py mon_deck.yaml
```

Le menu propose les étapes suivantes :

```
══════════════════════════════════════════════════════
  AeroPredict — Pipeline aérodynamique + Machine Learning
══════════════════════════════════════════════════════
  1. Construire le dataset géométrique (AeroSandbox)   ← requiert AeroSandbox
  2. Enrichir le dataset via XFoil (CL, CD, CM)        ← requiert AeroSandbox + XFoil
  3. Entraîner le modèle ML (TensorFlow)               ← requiert TensorFlow
  4. Appliquer le modèle ML (prédictions)              ← requiert TensorFlow
  5. Auditer le dataset (rapports PDF)
  6. Lancer le dashboard interactif (Streamlit)        ← données pré-calculées suffisent
  7. Afficher la configuration
  0. Quitter
```

### Option B — Dashboard uniquement (données pré-calculées fournies)

Les fichiers `naca_multitask_model.keras`, `preprocessor.pkl` et `dataset_projet/dataset_aeroXfoil_avec_predictions.csv` sont déjà fournis dans le dépôt. Pour lancer directement le dashboard :

```bash
streamlit run dashboard.py
```

L'interface s'ouvre automatiquement dans le navigateur à `http://localhost:8501`.

### Option C — Étapes individuelles

Chaque script peut être exécuté directement :

```bash
python data_profils.py          # Étape 1
python calcul_Xfoil.py          # Étape 2
python modele_ML.py             # Étape 3
python Application_modele_ML.py # Étape 4
python audit_dataset.py         # Étape 5
```

---

## Dashboard — Pages disponibles

| Page | Description |
|---|---|
| **Polaires** | Courbes CL, CD, CM et finesse L/D pour un profil et un Reynolds donnés. Superposition XFoil / ML / Ansys Fluent. |
| **Performance ML** | Métriques globales R², MAE, RMSE par coefficient. Nuages de dispersion prédiction vs référence. |
| **Optimisation** | Classement des profils par finesse maximale CL/CD pour des conditions de vol choisies. |
| **Dataset** | Distribution des familles de profils, taux de convergence XFoil par Reynolds, histogrammes des features géométriques. |

La **disposition des graphiques** (2 colonnes ou 1 colonne) est configurable depuis la barre latérale.

---

## Modèle ML — Réseau de neurones multi-tâches

- **Entrées** : 12 features — géométrie du profil (t, camber, x_t, x_c, LE_radius, TE_angle, t_over_xt, area), conditions de vol (alpha, Re), encodage catégoriel (naca, source).
- **Architecture** : réseau dense partagé → 3 têtes de sortie indépendantes.
- **Sorties** : CL, CD, CM simultanément (apprentissage multi-tâches).
- **Préprocessing** : `StandardScaler` sur les features continues, `LabelEncoder` sur les catégories, `StandardScaler` sur chaque cible.
- **Dataset d'entraînement** : ~401 000 points issus de simulations XFoil convergées.

---

## Tests

```bash
pytest tests/ -v
```

43 tests couvrant :
- `test_data_profils.py` — construction du dataset géométrique
- `test_calcul_xfoil.py` — enrichissement XFoil
- `test_modele_ml.py` — préprocesseur, aller-retour `inverse_transform_target`
- `test_main.py` — lecture YAML, menu principal

---

## Documentation

Le code est documenté selon la convention **Sphinx (reStructuredText)** vue au
Module 10a : chaque fonction/classe possède une docstring avec les champs
`:param:`, `:type:`, `:return:`, `:rtype:` et `:raises:`.

```python
def add_numbers(a, b):
    """
    Additionne deux nombres.

    :param a: Premier nombre.
    :type a: int
    :param b: Deuxieme nombre.
    :type b: int
    :return: La somme des deux nombres.
    :rtype: int
    """
    return a + b
```

### Générer la documentation HTML

```bash
# 1. Installer Sphinx
pip install Sphinx

# 2. (déjà fait dans ce dépôt) créer le squelette du projet
python -m sphinx.cmd.quickstart docs --sep

# 3. Générer/régénérer les fichiers .rst à partir des docstrings
#    (-f pour écraser/rafraîchir les .rst existants, ex. nouveau module)
python -m sphinx.ext.apidoc -f -o docs/source . docs tests

# 4. Construire le site HTML
cd docs
python -m sphinx -b html source build/html

# 5. Ouvrir le résultat
docs/build/html/index.html
```

`docs/source/conf.py` active l'extension `sphinx.ext.autodoc` et pointe
(`sys.path`) vers la racine du dépôt pour importer les modules. `docs/build/`
est un dossier généré, à ne pas committer (voir `.gitignore`).

Sans `aerosandbox`/`tensorflow` installés, autodoc échoue à importer
`calcul_Xfoil.py`, `data_profils.py` et `modele_ML.py` (pages vides dans le
HTML généré). `conf.py` contient `autodoc_mock_imports = ['aerosandbox',
'tensorflow']` pour simuler ces dépendances et extraire quand même les
docstrings sans les installer.

---

## Validation — Ansys Fluent

Trois profils ont été simulés indépendamment avec Ansys Fluent (CFD haute fidélité) pour valider les prédictions ML :

| Fichier | Profil | Reynolds |
|---|---|---|
| `naca0106_Re50000_FLUENT.csv` | NACA 0106 | 50 000 |
| `goe101_Re500000_FLUENT.csv` | Göttingen 101 | 500 000 |
| `supermarine371ii_Re100000_FLUENT.csv` | Supermarine 371 II | 100 000 |

Ces fichiers sont chargés automatiquement par le dashboard (page Polaires) lorsqu'ils sont présents dans le répertoire racine.

---

## Dépendances optionnelles

| Dépendance | Quand nécessaire | Installation |
|---|---|---|
| `tensorflow>=2.16` | Étapes 3 et 4 (entraînement / inférence) | `pip install tensorflow` |
| `aerosandbox` | Étapes 1 et 2 (génération dataset + XFoil) | `pip install aerosandbox` |
| XFoil (exécutable) | Étape 2 uniquement | Téléchargement manuel (MIT) |

> Le dashboard (étape 6) et l'audit (étape 5) ne nécessitent **aucune** de ces dépendances.

---

## Licence

Projet académique — MGA 802, ÉTS Montréal, Session Été 2026.
