"""
Application du modèle ML entraîné sur le dataset complet.

MGA 802 · AeroPredict

Ce script charge le modèle TensorFlow entraîné et le préprocesseur,
puis applique le modèle à l'ensemble du dataset pour prédire les
coefficients aérodynamiques CL, CD et CM.

Le traitement est effectué par lots pour gérer de grands volumes
de données avec un suivi d'avancement en temps réel.

Usage:
    python Application_modele_ML.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time


# ─────────────────────────────────────────────────────────────────────────────
# 1. REDÉFINITION DE LA CLASSE STRICTEMENT IDENTIQUE À L'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
class NACAAeroPreprocessor:
    """
    Préprocesseur pour les données aérodynamiques NACA.

    Cette classe doit correspondre exactement à l'objet sauvegardé
    dans preprocessor.pkl lors de l'entraînement. Elle gère la
    normalisation des features et des targets.

    :cvar GEOMETRIC_COLS: Noms des colonnes géométriques.
    :vartype GEOMETRIC_COLS: list
    :cvar AERO_COLS: Noms des colonnes aérodynamiques (conditions de vol).
    :vartype AERO_COLS: list
    :cvar TARGET_COLS: Noms des colonnes cibles (coefficients à prédire).
    :vartype TARGET_COLS: list

    :ivar feature_scaler: Scaler pour les features d'entrée.
    :vartype feature_scaler: sklearn.preprocessing.StandardScaler
    :ivar target_scalers: Scalers pour chaque cible (CL, CD, CM).
    :vartype target_scalers: dict
    :ivar is_fitted: Indique si le préprocesseur a été ajusté.
    :vartype is_fitted: bool
    """

    GEOMETRIC_COLS = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area"]
    AERO_COLS = ["alpha", "Re"]
    TARGET_COLS = ["CL", "CD", "CM"]

    def __init__(self):
        """Initialise le préprocesseur avec des scalers non ajustés."""
        self.feature_scaler = StandardScaler()
        self.target_scalers = {col: StandardScaler() for col in self.TARGET_COLS}
        self.is_fitted = False

    @property
    def input_cols(self) -> list:
        """
        Retourne la liste des colonnes d'entrée (géométrie + aéro).

        :return: Liste des noms de colonnes d'entrée.
        :rtype: list
        """
        return self.GEOMETRIC_COLS + self.AERO_COLS

    def fit_transform(self, df: pd.DataFrame):
        """
        Ajuste les scalers sur les données et transforme les features et targets.

        :param df: DataFrame contenant les colonnes d'entrée et les cibles.
        :type df: pandas.DataFrame
        :return: Tuple (X_scaled, y_dict) où X_scaled est la matrice des
                 features normalisées et y_dict un dictionnaire des targets normalisées.
        :rtype: tuple
        """
        X_raw = df[self.input_cols].values.astype(np.float32)
        X_scaled = self.feature_scaler.fit_transform(X_raw)
        y_dict = {}
        for col in self.TARGET_COLS:
            y_dict[col] = self.target_scalers[col].fit_transform(df[[col]].values).astype(np.float32)
        self.is_fitted = True
        return X_scaled, y_dict

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforme les features d'entrée en utilisant le scaler ajusté.

        :param df: DataFrame contenant les colonnes d'entrée.
        :type df: pandas.DataFrame
        :return: Matrice des features normalisées.
        :rtype: numpy.ndarray
        :raises AssertionError: Si le préprocesseur n'a pas été ajusté.
        """
        assert self.is_fitted, "Le préprocesseur doit d'abord être ajusté (fit)."
        X_raw = df[self.input_cols].values.astype(np.float32)
        return self.feature_scaler.transform(X_raw).astype(np.float32)

    def inverse_transform_target(self, y_scaled: np.ndarray, col: str) -> np.ndarray:
        """
        Transforme les prédictions normalisées vers les valeurs physiques.

        :param y_scaled: Valeurs normalisées à inverser.
        :type y_scaled: numpy.ndarray
        :param col: Nom de la cible ("CL", "CD" ou "CM").
        :type col: str
        :return: Valeurs physiques inversées.
        :rtype: numpy.ndarray
        """
        return self.target_scalers[col].inverse_transform(y_scaled)

    @property
    def input_dim(self) -> int:
        """
        Retourne la dimension de l'espace d'entrée.

        :return: Nombre de features d'entrée.
        :rtype: int
        """
        return len(self.input_cols)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCRIPT DE PRÉDICTION PROGRESSIVE AVEC SUIVI D'AVANCEMENT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Configuration des chemins
    fichiers_requis = ["naca_multitask_model.keras", "preprocessor.pkl", "dataset_aeroXfoil.csv"]
    for f in fichiers_requis:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Erreur : Le fichier requis '{f}' est introuvable.")

    print("=" * 60)
    print("  AeroPredict — Inférence Globale Progressive")
    print("=" * 60)

    print("[CHARGEMENT] Modèle, préprocesseur et BDD source...")
    t0_load = time.time()

    # Chargement du préprocesseur original via pickle
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    model = tf.keras.models.load_model("naca_multitask_model.keras")
    df = pd.read_csv("dataset_aeroXfoil.csv")

    # SUPPRESSION DE LA COLONNE converged SI ELLE EXISTE
    if "converged" in df.columns:
        df = df.drop(columns=["converged"])
        print("→ Colonne 'converged' supprimée du jeu de données")

    # SUPPRESSION DES COLONNES CL, CD, CM SI ELLES EXISTENT (pour éviter les conflits)
    cols_a_supprimer = ["CL", "CD", "CM"]
    cols_presentes = [col for col in cols_a_supprimer if col in df.columns]
    if cols_presentes:
        df = df.drop(columns=cols_presentes)
        print(f"→ Colonnes {cols_presentes} supprimées (seront remplacées par les prédictions)")

    print(f"→ Base chargée en {time.time() - t0_load:.2f}s | Lignes à traiter : {len(df):,}")

    # Préparation des conteneurs pour stocker les résultats finaux
    cl_predictions = []
    cd_predictions = []
    cm_predictions = []

    # Définition de la taille du bloc pour le suivi d'avancement
    BATCH_SIZE = 10000
    total_lignes = len(df)
    total_batches = int(np.ceil(total_lignes / BATCH_SIZE))

    print(f"\n[DÉBUT] Inférence par blocs de {BATCH_SIZE:,} lignes...")
    t0_inference = time.time()

    for i in range(total_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_lignes)

        # Extraction du bloc courant
        df_batch = df.iloc[start_idx:end_idx]

        # Encodage et normalisation des features du bloc (Uniquement géométrie et physique)
        X_batch_scaled = preprocessor.transform(df_batch)

        # Inférence via le réseau de neurones (multitâche)
        preds_scaled = model.predict(X_batch_scaled, verbose=0)

        # Inversion d'échelle vers les vraies valeurs physiques
        cl_phys = preprocessor.inverse_transform_target(preds_scaled[0], "CL").flatten()
        cd_phys = preprocessor.inverse_transform_target(preds_scaled[1], "CD").flatten()
        cm_phys = preprocessor.inverse_transform_target(preds_scaled[2], "CM").flatten()

        # Stockage
        cl_predictions.extend(cl_phys)
        cd_predictions.extend(cd_phys)
        cm_predictions.extend(cm_phys)

        # Calcul des temps d'avancement
        elapsed = time.time() - t0_inference
        lignes_traitees = end_idx
        vitesse = lignes_traitees / elapsed if elapsed > 0 else 0
        temps_restant = (total_lignes - lignes_traitees) / vitesse if vitesse > 0 else 0

        # Log d'avancement
        print(
            f"  [{i + 1:3d}/{total_batches}]  "
            f"Lignes: {start_idx:7,d} → {end_idx:7,d}  |  "
            f"Vitesse: {vitesse:6.0f} lig/s  |  "
            f"Reste: {temps_restant / 60:4.1f} min"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. ENREGISTREMENT ET RÉSUMÉ
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[FINISH] Intégration des coefficients prédits dans le jeu de données...")
    df["CL"] = cl_predictions
    df["CD"] = cd_predictions
    df["CM"] = cm_predictions

    # Sauvegarde dans un nouveau fichier pour préserver l'original
    output_path = "dataset_aeroXfoil_avec_predictions.csv"
    df.to_csv(output_path, index=False)

    temps_total = time.time() - t0_inference
    print(f"\n{'=' * 60}")
    print("  Prédiction terminée avec succès !")
    print(f"{'=' * 60}")
    print(f"  Fichier source original    : dataset_aeroXfoil.csv")
    print(f"  Fichier avec prédictions   : {output_path}")
    print(f"  Lignes totales traitées    : {len(df):,}")
    print(f"  Colonnes dans le nouveau fichier : {list(df.columns)[:5]}... + CL, CD, CM")
    print(f"  Temps d'inférence          : {temps_total / 60:.2f} min")
    print(f"  Vitesse moyenne            : {len(df) / temps_total:.0f} lig/s")

    # Afficher un aperçu des prédictions
    print(f"\n{'-' * 60}")
    print("  Aperçu des prédictions (5 premières lignes) :")
    print(f"{'-' * 60}")
    print(df[["naca", "alpha", "CL", "CD", "CM"]].head().to_string(index=False))

    print(f"\n{'=' * 60}")
