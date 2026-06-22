import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import tensorflow as tf
except ImportError:
    tf = None

"""
=============================================================================
TEST DE ROBUSTESSE AU BRUIT (ANALYSE DE SENSIBILITÉ)
=============================================================================

Ce test simule les incertitudes de fabrication réelles.
On fait varier des paramètres géométriques des profils 
(épaisseur, cambrure, rayon du bord d'attaque) de ±2%. 
L'objectif est de vérifier que l'IA reste
stable face à ces variations et continue de prédire correctement 
les coefficients aérodynamiques sans diverger.
=============================================================================
"""
# === CHEMINS ===
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset_aeroXfoil.csv"
MODEL_PATH = ROOT / "naca_multitask_model.keras"
PREP_PATH = ROOT / "preprocessor.pkl"
# =======================

# Vérification préliminaire des fichiers
if not (os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH) and os.path.exists(DATASET_PATH)):
    pytest.skip("Fichiers du modèle ou dataset introuvables.", allow_module_level=True)

from modele_ML import NACAAeroPreprocessor
sys.modules['__main__'].NACAAeroPreprocessor = NACAAeroPreprocessor
        
@pytest.fixture(scope="module")
def pipeline_ia():
    """Charge le modèle, le préprocesseur et un échantillon de données."""
    if tf is None:
        pytest.skip("TensorFlow est requis pour exécuter ce test.")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH) and os.path.exists(DATASET_PATH)):
        pytest.skip("Fichiers du modèle ou dataset introuvables. Lancez l'entraînement d'abord.")

    # Chargement du préprocesseur
    with open(PREP_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    
    # Chargement du modèle Keras
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # On charge un échantillon représentatif (ici 1000 points) pour garder un test rapide
    df_sample = pd.read_csv(DATASET_PATH).sample(n=1000, random_state=42).copy()
    
    return model, preprocessor, df_sample


@pytest.mark.validation
class TestRobustesseBruit:
    """Évaluation de la stabilité du réseau de neurones face aux incertitudes d'usinage."""

    def test_sensibilite_bruit_geometrique(self, pipeline_ia):
        model, preprocessor, df_orig = pipeline_ia
        
        # Sélection des caractéristiques géométriques à faire varier (±2% d'erreur)
        features_a_bruiter = ["t", "camber", "LE_radius"]
        
        # Création du dataset avec bruit
        df_bruit = df_orig.copy()
        np.random.seed(42) # Pour des tests reproductibles
        
        for col in features_a_bruiter:
            # Multiplicateur aléatoire entre 0.98 et 1.02
            bruit = np.random.uniform(0.98, 1.02, size=len(df_bruit))
            df_bruit[col] = df_bruit[col] * bruit
            
        # profil parfait
        X_orig = preprocessor.transform(df_orig)
        preds_orig_scaled = model.predict(X_orig, verbose=0)
        
        # profil avec variation géométrique
        X_bruit = preprocessor.transform(df_bruit)
        preds_bruit_scaled = model.predict(X_bruit, verbose=0)
        
        # On définit des tolérances pour chaque coefficient aérodynamique
        seuils_tolerance = {
            "CL": 0.025,  # Le CL ne doit pas varier de plus de 0.025 en moyenne
            "CD": 0.003,  # Le CD ne doit pas varier de plus de 0.003 en moyenne
            "CM": 0.010   # Moment de tangage ne doit pas varier de plus de 0.010 en moyenne
        }
        
        for i, coef in enumerate(["CL", "CD", "CM"]):
            val_orig = preprocessor.inverse_transform_target(preds_orig_scaled[i], coef).flatten()
            val_bruit = preprocessor.inverse_transform_target(preds_bruit_scaled[i], coef).flatten()
            
            # Calcul de la variation absolue moyenne causée par les variations
            variation_moyenne = np.mean(np.abs(val_orig - val_bruit))
            
            # Le test échoue si la variation moyenne dépasse le seuil toléré
            assert variation_moyenne < seuils_tolerance[coef], (
                f"Instabilité critique détectée sur {coef} ! "
                f"Une variation géométrique de ±2% a causé un écart moyen de {variation_moyenne:.4f} "
                f"(Tolérance max: {seuils_tolerance[coef]}). Le modèle sur-apprend le bruit."
            )
