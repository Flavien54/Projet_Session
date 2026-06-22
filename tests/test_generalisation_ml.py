import pandas as pd
import numpy as np
import pytest
import os

"""
=============================================================================
TEST DE GÉNÉRALISATION DU MODÈLE MACHINE LEARNING 
=============================================================================
1. L'objectif du test est de que l'Intelligence Artificielle n'apprenne pas ses données d'entraînement par cœur sans comprendre la logique.
- Les profils NACA 4 chiffres sont générés par une équation mathématique stricte.
- Les profils UIUC sont des profils d'ailes réels, plus complexe.

Le fonctionnement du test :
    On sépare les prédictions en deux groupes (NACA vs UIUC).
    On calcule l'Erreur Absolue Moyenne du CL pour voir de combien l'IA se trompe par rapport à XFoil.
    L'erreur sur les profils réels (UIUC) ne doit pas dépasser l'erreur sur les profils théoriques de plus d'une marge tolérée de 0.05.

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
    
    # On garde toutes les colonnes XFoil, on ajoute les prédictions
    cols_ml = ["naca", "alpha", "Re", "CL", "CD", "CM"]
    df_merge = df_xfoil.merge(
        df_ml[cols_ml],
        on=["naca", "alpha", "Re"],
        suffixes=("_xfoil", "_ml")
    )
    
    return df_merge[df_merge["converged"] == True]

@pytest.mark.validation
class TestGeneralisationML:
    """Évaluation de la capacité du modèle à généraliser hors de sa zone de confort."""

    def test_generalisation_profils_uiuc(self, df_predictions):
        """Vérifie que l'IA performe bien sur des profils réels inconnus (UIUC)."""
        
        # On sépare le dataset selon la provenance géométrique (NACA vs UIUC)
        df_naca = df_predictions[df_predictions["source"] == "naca_grid"]
        df_uiuc = df_predictions[df_predictions["source"] == "uiuc"]
        
        assert not df_uiuc.empty, "Aucun profil UIUC trouvé dans le dataset de test."
        
        # On calcule l'erreur absolue moyenne du coefficient de portance (CL) pour les deux datasets
        mae_naca = np.mean(np.abs(df_naca["CL_xfoil"] - df_naca["CL_ml"]))
        mae_uiuc = np.mean(np.abs(df_uiuc["CL_xfoil"] - df_uiuc["CL_ml"]))
        
        # L'erreur sur les profils réels UIUC ne doit pas dépasser celle des NACA de plus de 0.05
        tolerance_generalisation = mae_naca + 0.05
        
        assert mae_uiuc <= tolerance_generalisation, (
            f"Problème d'overfitting détecté ! "
            f"Erreur NACA: {mae_naca:.4f} | Erreur UIUC: {mae_uiuc:.4f}. "
            f"Le modèle a sur-appris les mathématiques NACA et peine sur les profils réels."
        )
