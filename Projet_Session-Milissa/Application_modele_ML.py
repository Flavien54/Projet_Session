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

        # Encodage et normalisation des features du bloc
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
