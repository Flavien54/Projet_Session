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
    """Prétraitement des données aérodynamiques NACA pour l'apprentissage multi-tâches.

    Ce préprocesseur gère l'encodage des variables catégorielles (naca, source),
    la normalisation des variables continues et la standardisation des cibles.

    :ivar CATEGORICAL_COLS: Noms des colonnes catégorielles du dataset.
    :vartype CATEGORICAL_COLS: list
    :ivar CONTINUOUS_COLS: Noms des colonnes continues à normaliser.
    :vartype CONTINUOUS_COLS: list
    :ivar TARGET_COLS: Noms des colonnes cibles (CL, CD, CM).
    :vartype TARGET_COLS: list
    :ivar label_encoders: Dictionnaire des encodeurs LabelEncoder pour chaque colonne catégorielle.
    :vartype label_encoders: dict
    :ivar feature_scaler: Scaler pour les caractéristiques continues.
    :vartype feature_scaler: StandardScaler
    :ivar target_scalers: Dictionnaire des scalers pour chaque colonne cible.
    :vartype target_scalers: dict
    :ivar is_fitted: Indique si le préprocesseur a été ajusté sur les données.
    :vartype is_fitted: bool
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
        """Ajuste le préprocesseur et transforme les données.

        :param df: DataFrame contenant les données brutes avec toutes les colonnes.
        :type df: pd.DataFrame
        :return: Tuple ``(X, y_dict)`` où ``X`` est la matrice des
            caractéristiques encodées et normalisées, et ``y_dict`` le
            dictionnaire des cibles standardisées pour CL, CD, CM.
        :rtype: tuple
        """
        X = self._encode_features(df, fit=True)
        y_dict = {}
        for col in self.TARGET_COLS:
            y_dict[col] = self.target_scalers[col].fit_transform(df[[col]].values).astype(np.float32)
        self.is_fitted = True
        return X, y_dict

    def transform(self, df: pd.DataFrame):
        """Transforme les données avec le préprocesseur déjà ajusté.

        :param df: DataFrame contenant les données brutes.
        :type df: pd.DataFrame
        :return: Matrice des caractéristiques encodées et normalisées.
        :rtype: np.ndarray
        :raises AssertionError: Si le préprocesseur n'a pas été ajusté avec fit_transform.
        """
        assert self.is_fitted, "Le préprocesseur doit d'abord être ajusté (fit)."
        return self._encode_features(df, fit=False)

    def inverse_transform_target(self, y_scaled: np.ndarray, col: str) -> np.ndarray:
        """Inverse la normalisation d'une colonne cible.

        :param y_scaled: Valeurs cibles normalisées.
        :type y_scaled: np.ndarray
        :param col: Nom de la colonne cible ('CL', 'CD' ou 'CM').
        :type col: str
        :return: Valeurs cibles dans l'échelle d'origine.
        :rtype: np.ndarray
        """
        return self.target_scalers[col].inverse_transform(y_scaled)

    def _encode_features(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        """Encode les caractéristiques catégorielles et continues.

        :param df: DataFrame contenant les données.
        :type df: pd.DataFrame
        :param fit: Si True, ajuste les encodeurs, sinon utilise les encodeurs existants.
        :type fit: bool
        :return: Matrice des caractéristiques encodées.
        :rtype: np.ndarray
        """
        encoded_parts = []

        # Encodage des variables catégorielles
        for col in self.CATEGORICAL_COLS:
            if fit:
                enc = self.label_encoders[col].fit_transform(df[col].astype(str)).reshape(-1, 1)
            else:
                known = set(self.label_encoders[col].classes_)
                safe = df[col].astype(str).map(lambda x: x if x in known else self.label_encoders[col].classes_[0])
                enc = self.label_encoders[col].transform(safe).reshape(-1, 1)
            encoded_parts.append(enc.astype(np.float32))

        # Normalisation des variables continues
        cont = df[self.CONTINUOUS_COLS].values.astype(np.float32)
        cont_scaled = self.feature_scaler.fit_transform(cont) if fit else self.feature_scaler.transform(cont)
        encoded_parts.append(cont_scaled)

        return np.hstack(encoded_parts).astype(np.float32)

    @property
    def input_dim(self) -> int:
        """Retourne la dimension de l'espace d'entrée.

        :return: Nombre total de caractéristiques après encodage.
        :rtype: int
        """
        return len(self.CATEGORICAL_COLS) + len(self.CONTINUOUS_COLS)
# ─────────────────────────────────────────────────────────────────────────────
# 2. ARCHITECTURE MULTI-TÂCHES PROFONDE
# ─────────────────────────────────────────────────────────────────────────────
class NACAMultiTaskModel:
    """Modèle de réseau de neurones multi-tâches pour la prédiction des coefficients aérodynamiques.

    Ce modèle prédit simultanément CL (coefficient de portance),
    CD (coefficient de traînée) et CM (coefficient de moment).

    :ivar input_dim: Dimension de l'espace d'entrée.
    :vartype input_dim: int
    :ivar learning_rate: Taux d'apprentissage pour l'optimiseur Adam.
    :vartype learning_rate: float
    :ivar model: Modèle Keras compilé.
    :vartype model: keras.Model
    """

    def __init__(self, input_dim: int, learning_rate: float = 5e-4):
        """Initialise le modèle multi-tâches.

        :param input_dim: Dimension des caractéristiques d'entrée.
        :type input_dim: int
        :param learning_rate: Taux d'apprentissage (défaut: 5e-4).
        :type learning_rate: float
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = self._build()

    def _build_expert_branch(self, base_tensor, name: str):
        """Construit une branche experte pour une tâche spécifique.

        :param base_tensor: Tenseur d'entrée provenant du tronc commun.
        :type base_tensor: tf.Tensor
        :param name: Nom de la branche (CL, CD ou CM).
        :type name: str
        :return: Tenseur de sortie pour la tâche spécifique.
        :rtype: tf.Tensor
        """
        # Première couche dense élargie
        x = layers.Dense(256, activation="swish", name=f"dense_{name}_1")(base_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        # Deuxième couche avec skip connection
        x_skip = layers.Dense(128, activation="swish", name=f"dense_{name}_2")(x)
        x_skip = layers.BatchNormalization()(x_skip)

        # Troisième couche
        x = layers.Dense(128, activation="swish", name=f"dense_{name}_3")(x_skip)
        x = layers.BatchNormalization()(x)

        # Connexion résiduelle pour stabiliser les gradients
        x = layers.add([x, x_skip], name=f"res_{name}")

        # Couche de sortie linéaire
        out = layers.Dense(1, activation="linear", name=name)(x)
        return out

    def _build(self) -> keras.Model:
        """Construit l'architecture complète du modèle multi-tâches.

        :return: Modèle Keras compilé avec les sorties CL, CD, CM.
        :rtype: keras.Model
        """
        # Entrée du modèle
        inp = keras.Input(shape=(self.input_dim,), name="features")

        # Tronc commun partagé (3 couches denses profondes)
        x = layers.Dense(512, activation="swish", name="shared_1")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(512, activation="swish", name="shared_2")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(256, activation="swish", name="shared_3")(x)
        x = layers.BatchNormalization()(x)

        # Branches expertes pour chaque coefficient aérodynamique
        out_cl = self._build_expert_branch(x, "CL")
        out_cd = self._build_expert_branch(x, "CD")
        out_cm = self._build_expert_branch(x, "CM")

        # Création du modèle
        model = keras.Model(inputs=inp, outputs=[out_cl, out_cd, out_cm], name="NACA_MultiTaskNet_Deep")

        # Compilation avec l'optimiseur Adam et la perte MSE
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={"CL": "mse", "CD": "mse", "CM": "mse"},
            metrics={"CL": "mae", "CD": "mae", "CM": "mae"}
        )
        return model
# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE PRINCIPAL D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Pipeline principal d'entraînement du modèle multi-tâches.

    Ce script charge les données, prétraite les caractéristiques,
    entraîne le modèle avec des callbacks d'early stopping et de réduction
    du taux d'apprentissage, puis sauvegarde les artefacts.
    """

    # Configuration des chemins de fichiers
    csv_file_path = "dataset_aeroXfoil.csv"

    # Vérification de l'existence du fichier de données
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Le fichier '{csv_file_path}' est introuvable.")

    print(f"Chargement de la base de données entière : {csv_file_path}")
    df_raw = pd.read_csv(csv_file_path)

    # Séparation des ensembles d'entraînement et de test (85% / 15%)
    df_train, df_test = train_test_split(df_raw, test_size=0.15, random_state=42)

    # Initialisation et application du préprocesseur
    preprocessor = NACAAeroPreprocessor()
    X_train_scaled, y_train_dict = preprocessor.fit_transform(df_train)

    # Séparation entraînement/validation pour l'early stopping
    idx_t, idx_v = train_test_split(np.arange(len(X_train_scaled)), test_size=0.15, random_state=42)
    X_t, X_v = X_train_scaled[idx_t], X_train_scaled[idx_v]
    y_t = {k: v[idx_t] for k, v in y_train_dict.items()}
    y_v = {k: v[idx_v] for k, v in y_train_dict.items()}

    # Création du modèle multi-tâches
    mt_model = NACAMultiTaskModel(input_dim=preprocessor.input_dim, learning_rate=5e-4)

    # Configuration des callbacks pour l'entraînement
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,  # Patience augmentée pour laisser converger
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,  # Réduction de moitié du taux d'apprentissage
            patience=6,  # Patience avant réduction
            min_lr=1e-6,  # Taux d'apprentissage minimum
            verbose=1
        )
    ]

    # Lancement de l'entraînement
    print("\n[ML] Lancement de l'apprentissage à haute capacité...")
    mt_model.model.fit(
        X_t, y_t,
        validation_data=(X_v, y_v),
        epochs=120,  # Nombre maximal d'époques
        batch_size=512,  # Taille de lot pour stabilité
        callbacks=cbs,
        verbose=1
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. EXPORTATION DES ARTIFACTS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[EXPORT] Sauvegarde des modèles et configurations...")

    # Sauvegarde du modèle Keras au format .keras
    mt_model.model.save("naca_multitask_model.keras")

    # Sauvegarde du préprocesseur (scalers et encodeurs) avec pickle
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Sauvegarde du jeu de test pour analyses ultérieures
    df_test.to_csv("dataset_test_isolated.csv", index=False)

    # Affichage des confirmations de sauvegarde
    print("→ 'naca_multitask_model.keras' enregistré.")
    print("→ 'preprocessor.pkl' enregistré.")
    print("→ 'dataset_test_isolated.csv' enregistré pour l'analyse.")
    print("Entraînement terminé avec succès.")
