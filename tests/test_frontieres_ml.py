import pandas as pd
import numpy as np
import pytest
import os

"""
=============================================================================
TEST DES LIMITES PHYSIQUES DU MODÈLE MACHINE LEARNING
=============================================================================

Évaluer la robustesse de l'Intelligence Artificielle aux frontières de son 
domaine d'entraînement, en ciblant les cas extrêmes.

On choisit :
    - Des angles d'attaque très élevés (alpha >= 15°). 
    - Le plus bas nombre de Reynolds simulé, là où les effets de viscosité dominent fortement l'écoulement.
    - Les profils les plus épais (épaisseur t >= 20%)
    
Pour chaque zone limite, on calcule l'erreur absolue moyenne entre les prédictions et la réalité.
On vérifie que cette déviation reste acceptable (MAE < seuil).
=============================================================================
"""

# === CHEMINS ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_XFOIL_PATH = os.path.join(ROOT_DIR, "dataset_aeroXfoil.csv")
DATASET_ML_PATH = os.path.join(ROOT_DIR, "dataset_aeroXfoil_avec_predictions.csv")
# =======================

@pytest.fixture(scope="module")
def df_predictions():
    """Charge et fusionne les datasets XFoil et ML pour la comparaison."""
    if not (os.path.exists(DATASET_XFOIL_PATH) and os.path.exists(DATASET_ML_PATH)):
        pytest.skip("Datasets introuvables. Exécutez le pipeline complet d'abord.")
        
    df_xfoil = pd.read_csv(DATASET_XFOIL_PATH)
    df_ml = pd.read_csv(DATASET_ML_PATH)
    
    # On garde toutes les colonnes de XFoil, et on y ajoute juste les 3 prédictions de ML
    cols_ml = ["naca", "alpha", "Re", "CL", "CD", "CM"]
    
    df_merge = df_xfoil.merge(
        df_ml[cols_ml],
        on=["naca", "alpha", "Re"],
        suffixes=("_xfoil", "_ml")
    )
    
    # On garde que les points où XFoil a convergé
    return df_merge[df_merge["converged"] == True]

def evaluer_erreur_frontiere(df_subset, coef, seuil_mae):
    """Fonction utilitaire pour calculer et vérifier la MAE."""
    vrai = df_subset[f"{coef}_xfoil"]  # Valeur XFoil
    pred = df_subset[f"{coef}_ml"]     # Prédiction ML
    mae = np.mean(np.abs(vrai - pred))
    
    assert mae <= seuil_mae, (
        f"Échec aux frontières : L'erreur MAE pour {coef} est de {mae:.4f}, "
        f"ce qui dépasse le seuil toléré de {seuil_mae}."
    )

@pytest.mark.validation
class TestLimitesPhysiquesML:
    """Évaluation du modèle IA sur les cas aux frontières du domaine."""

    def test_frontiere_haute_incidence_decrochage(self, df_predictions):
        """Teste la précision du modèle proche du décrochage (alpha >= 15°)."""
        df_frontiere = df_predictions[df_predictions["alpha"] >= 15.0]
        assert not df_frontiere.empty, "Aucune donnée de décrochage trouvée."
        
        # Tolérances plus souples car la zone non linéaire est difficile à prédire
        evaluer_erreur_frontiere(df_frontiere, "CL", seuil_mae=0.15)
        evaluer_erreur_frontiere(df_frontiere, "CD", seuil_mae=0.05)

    def test_frontiere_reynolds_critique(self, df_predictions):
        """Teste la précision sur le nombre de Reynolds le plus bas (viscosité dominante)."""
        min_re = df_predictions["Re"].min()
        df_frontiere = df_predictions[df_predictions["Re"] == min_re]
        
        # Tolérance standard
        evaluer_erreur_frontiere(df_frontiere, "CL", seuil_mae=0.10)

    def test_frontiere_geometrie_extreme(self, df_predictions):
        """Teste la précision sur les profils les plus épais et cambrés (ex: NACA 6624)."""
        # On cible les profils avec une épaisseur t >= 20%
        df_frontiere = df_predictions[df_predictions["t"] >= 0.20]
        assert not df_frontiere.empty, "Aucune donnée de géométrie extrême trouvée."
        
        evaluer_erreur_frontiere(df_frontiere, "CM", seuil_mae=0.03)