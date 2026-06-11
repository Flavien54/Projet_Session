import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os
import pickle

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRÉPROCESSEUR DU JEU DE DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
class NACAAeroPreprocessor:
    """
    Prétraitement des données aérodynamiques NACA pour l'apprentissage multi-tâches.

    Ce préprocesseur gère l'encodage des variables catégorielles (naca, source),
    la normalisation des variables continues et la standardisation des cibles.

    Attributes
    ----------
    CATEGORICAL_COLS : list
        Noms des colonnes catégorielles du dataset.
    CONTINUOUS_COLS : list
        Noms des colonnes continues à normaliser.
    TARGET_COLS : list
        Noms des colonnes cibles (CL, CD, CM).
    label_encoders : dict
        Dictionnaire des encodeurs LabelEncoder pour chaque colonne catégorielle.
    feature_scaler : StandardScaler
        Scaler pour les caractéristiques continues.
    target_scalers : dict
        Dictionnaire des scalers pour chaque colonne cible.
    is_fitted : bool
        Indique si le préprocesseur a été ajusté sur les données.
    """

    CATEGORICAL_COLS = ["naca", "source"]
    CONTINUOUS_COLS = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area", "alpha", "Re"]
    TARGET_COLS = ["CL", "CD", "CM"]

    def __init__(self):
        """Initialise le préprocesseur avec les encodeurs et scalers."""
        self.label_encoders = {col: LabelEncoder() for col in self.CATEGORICAL_COLS}
        self.feature_scaler = StandardScaler()
        self.target_scalers = {col: StandardScaler() for col in self.TARGET_COLS}
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame):
        """
        Ajuste le préprocesseur et transforme les données.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contenant les données brutes avec toutes les colonnes.

        Returns
        -------
        tuple
            X : np.ndarray
                Matrice des caractéristiques encodées et normalisées.
            y_dict : dict
                Dictionnaire des cibles standardisées pour CL, CD, CM.
        """
        X = self._encode_features(df, fit=True)
        y_dict = {}
        for col in self.TARGET_COLS:
            y_dict[col] = self.target_scalers[col].fit_transform(df[[col]].values).astype(np.float32)
        self.is_fitted = True
        return X, y_dict

    def transform(self, df: pd.DataFrame):
        """
        Transforme les données avec le préprocesseur déjà ajusté.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contenant les données brutes.

        Returns
        -------
        np.ndarray
            Matrice des caractéristiques encodées et normalisées.

        Raises
        ------
        AssertionError
            Si le préprocesseur n'a pas été ajusté avec fit_transform.
        """
        assert self.is_fitted, "Le préprocesseur doit d'abord être ajusté (fit)."
        return self._encode_features(df, fit=False)

    def inverse_transform_target(self, y_scaled: np.ndarray, col: str) -> np.ndarray:
        """
        Inverse la normalisation d'une colonne cible.

        Parameters
        ----------
        y_scaled : np.ndarray
            Valeurs cibles normalisées.
        col : str
            Nom de la colonne cible ('CL', 'CD' ou 'CM').

        Returns
        -------
        np.ndarray
            Valeurs cibles dans l'échelle d'origine.
        """
        return self.target_scalers[col].inverse_transform(y_scaled)
