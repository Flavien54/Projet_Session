import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import time


# ─────────────────────────────────────────────────────────────────────────────
# 1. REDÉFINITION DE LA CLASSE (Obligatoire pour que pickle puisse lire l'objet)
# ─────────────────────────────────────────────────────────────────────────────
class NACAAeroPreprocessor:
    CATEGORICAL_COLS = ["naca", "source"]
    CONTINUOUS_COLS = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area", "alpha", "Re"]
    TARGET_COLS = ["CL", "CD", "CM"]

    def __init__(self):
        self.label_encoders = {col: LabelEncoder() for col in self.CATEGORICAL_COLS}
        self.feature_scaler = StandardScaler()
        self.target_scalers = {col: StandardScaler() for col in self.TARGET_COLS}
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame):
        X = self._encode_features(df, fit=True)
        y_dict = {}
        for col in self.TARGET_COLS:
            y_dict[col] = self.target_scalers[col].fit_transform(df[[col]].values).astype(np.float32)
        self.is_fitted = True
        return X, y_dict

    def transform(self, df: pd.DataFrame):
        assert self.is_fitted, "Le préprocesseur doit d'abord être ajusté (fit)."
        return self._encode_features(df, fit=False)

    def inverse_transform_target(self, y_scaled: np.ndarray, col: str) -> np.ndarray:
        return self.target_scalers[col].inverse_transform(y_scaled)

    def _encode_features(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        encoded_parts = []
        for col in self.CATEGORICAL_COLS:
            if fit:
                enc = self.label_encoders[col].fit_transform(df[col].astype(str)).reshape(-1, 1)
            else:
                known = set(self.label_encoders[col].classes_)
                safe = df[col].astype(str).map(lambda x: x if x in known else self.label_encoders[col].classes_[0])
                enc = self.label_encoders[col].transform(safe).reshape(-1, 1)
            encoded_parts.append(enc.astype(np.float32))

        cont = df[self.CONTINUOUS_COLS].values.astype(np.float32)
        cont_scaled = self.feature_scaler.fit_transform(cont) if fit else self.feature_scaler.transform(cont)
        encoded_parts.append(cont_scaled)
        return np.hstack(encoded_parts).astype(np.float32)

    @property
    def input_dim(self) -> int:
        return len(self.CATEGORICAL_COLS) + len(self.CONTINUOUS_COLS)
