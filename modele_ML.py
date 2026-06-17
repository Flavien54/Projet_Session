"""Entraînement du modèle de Machine Learning AeroPredict — MGA 802.

Ce module définit et entraîne un réseau de neurones multi-tâches prédisant
les trois coefficients aérodynamiques ``CL``, ``CD`` et ``CM`` d'un profil
NACA à partir de ses seules caractéristiques physiques et géométriques
(épaisseur, cambrure, position des extrema, rayon de bord d'attaque, angle
de bord de fuite, ratios et surface) et des conditions de vol (angle
d'attaque, nombre de Reynolds).

Le choix de n'utiliser que des descripteurs physiques continus — et
explicitement aucune variable catégorielle telle que le nom du profil
(``naca``) ou sa source (``source``) — est délibéré : il garantit que le
modèle généralise à des profils absents de la base d'entraînement, dès lors
que leur géométrie peut être décrite par les mêmes 8 paramètres.

Architecture du réseau : un tronc commun (``shared``) apprend les
interactions entre géométrie et écoulement, puis trois branches expertes
indépendantes (une par coefficient) se spécialisent chacune via un bloc
résiduel (connexion ``skip``), avant une sortie linéaire scalaire.


"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
import pickle

warnings.filterwarnings("ignore")

# Graines fixées pour la reproductibilité des résultats d'entraînement
# (initialisation des poids, mélange des données, dropout).
tf.random.set_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRÉPROCESSEUR DU JEU DE DONNÉES (PHYSIQUE & GÉOMÉTRIE UNIQUEMENT)
# ─────────────────────────────────────────────────────────────────────────────
class NACAAeroPreprocessor:
    """Prétraite les caractéristiques physiques et géométriques pour l'apprentissage.

    Les variables textuelles/catégorielles (``naca``, ``source``) sont
    délibérément exclues des entrées du modèle afin de garantir la
    généralisation aux profils hors de la base de données : seules des
    quantités physiques continues, valables pour n'importe quel profil,
    sont utilisées.

    .. important::
        Cette classe doit rester structurellement identique à celle
        utilisée côté inférence (voir ``dashboard.py`` et
        ``Application_modele_ML.py``), car son état (notamment les
        scalers ajustés) est sérialisé par ``pickle`` puis désérialisé
        dans ces autres contextes.

    :ivar GEOMETRIC_COLS: Caractéristiques intrinsèques de la géométrie
        du profil (8 colonnes).
    :vartype GEOMETRIC_COLS: list[str]
    :ivar AERO_COLS: Conditions physiques de l'écoulement (incidence et
        nombre de Reynolds).
    :vartype AERO_COLS: list[str]
    :ivar TARGET_COLS: Cibles aérodynamiques prédites (``CL``, ``CD``,
        ``CM``).
    :vartype TARGET_COLS: list[str]
    :ivar feature_scaler: Scaler ajusté sur l'ensemble des entrées
        physiques (concaténation de :attr:`GEOMETRIC_COLS` et
        :attr:`AERO_COLS`).
    :vartype feature_scaler: sklearn.preprocessing.StandardScaler
    :ivar target_scalers: Dictionnaire associant chaque nom de cible
        (clé) à son ``StandardScaler`` dédié (valeur), chaque cible étant
        normalisée indépendamment des autres.
    :vartype target_scalers: dict[str, sklearn.preprocessing.StandardScaler]
    :ivar is_fitted: Indique si :meth:`fit_transform` a déjà été appelée
        (et donc si les scalers sont prêts à être utilisés par
        :meth:`transform`).
    :vartype is_fitted: bool
    """

    #: Caractéristiques géométriques qui dictent la physique du profil.
    GEOMETRIC_COLS = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area"]

    #: Conditions physiques opérationnelles (angle d'attaque, Reynolds).
    AERO_COLS = ["alpha", "Re"]

    #: Coefficients aérodynamiques à prédire.
    TARGET_COLS = ["CL", "CD", "CM"]

    def __init__(self):
        """Initialise le préprocesseur physique avec des scalers non ajustés.

        Les scalers (:attr:`feature_scaler` et :attr:`target_scalers`)
        sont instanciés vides ; ils ne sont réellement ajustés (calcul
        des moyennes/écarts-types) que lors de l'appel à
        :meth:`fit_transform`.
        """
        self.feature_scaler = StandardScaler()
        self.target_scalers = {col: StandardScaler() for col in self.TARGET_COLS}
        self.is_fitted = False

    @property
    def input_cols(self) -> list:
        """Liste ordonnée de toutes les colonnes d'entrée utilisées par le modèle.

        :returns: Concaténation de :attr:`GEOMETRIC_COLS` et
            :attr:`AERO_COLS`, dans cet ordre exact — cet ordre doit être
            respecté à l'identique côté inférence.
        :rtype: list[str]
        """
        return self.GEOMETRIC_COLS + self.AERO_COLS

    def fit_transform(self, df: pd.DataFrame):
        """Ajuste les scalers sur les données d'entraînement et les normalise.

        Cette méthode doit être appelée une seule fois, exclusivement
        sur l'ensemble d'entraînement, afin d'éviter toute fuite de
        données (*data leakage*) depuis les ensembles de validation ou
        de test.

        :param df: DataFrame d'entraînement contenant au minimum les
            colonnes listées par :attr:`input_cols` et
            :attr:`TARGET_COLS`.
        :type df: pandas.DataFrame
        :returns: Un couple ``(X_scaled, y_dict)`` où :

            * ``X_scaled`` est le tableau ``float32`` des entrées
              standardisées, de forme ``(n_lignes, len(input_cols))`` ;
            * ``y_dict`` est un dictionnaire associant chaque nom de
              cible à un tableau ``float32`` standardisé de forme
              ``(n_lignes, 1)``.
        :rtype: tuple(numpy.ndarray, dict[str, numpy.ndarray])
        """
        X_raw = df[self.input_cols].values.astype(np.float32)
        X_scaled = self.feature_scaler.fit_transform(X_raw)

        y_dict = {}
        for col in self.TARGET_COLS:
            y_dict[col] = self.target_scalers[col].fit_transform(df[[col]].values).astype(np.float32)

        self.is_fitted = True
        return X_scaled, y_dict

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Standardise les entrées d'un profil inconnu à partir des scalers déjà ajustés.

        :param df: DataFrame contenant au minimum les colonnes listées
            par :attr:`input_cols` (profil quelconque, potentiellement
            absent de l'ensemble d'entraînement).
        :type df: pandas.DataFrame
        :returns: Tableau ``float32`` standardisé, prêt à être fourni en
            entrée du réseau de neurones.
        :rtype: numpy.ndarray
        :raises AssertionError: Si le préprocesseur n'a pas encore été
            ajusté via :meth:`fit_transform`.
        """
        assert self.is_fitted, "Le préprocesseur doit d'abord être ajusté (fit)."
        X_raw = df[self.input_cols].values.astype(np.float32)
        return self.feature_scaler.transform(X_raw).astype(np.float32)

    def inverse_transform_target(self, y_scaled: np.ndarray, col: str) -> np.ndarray:
        """Ramène une prédiction standardisée à son échelle physique d'origine.

        :param y_scaled: Valeurs standardisées, typiquement issues
            directement d'une sortie du modèle.
        :type y_scaled: numpy.ndarray
        :param col: Nom de la cible concernée (``"CL"``, ``"CD"`` ou
            ``"CM"``), doit être une clé de :attr:`target_scalers`.
        :type col: str
        :returns: Valeurs dans l'échelle physique d'origine de la cible.
        :rtype: numpy.ndarray
        """
        return self.target_scalers[col].inverse_transform(y_scaled)

    @property
    def input_dim(self) -> int:
        """Dimension du vecteur d'entrée du modèle.

        :returns: Nombre total de features d'entrée (géométriques +
            aérodynamiques), utilisé pour dimensionner la couche d'entrée
            de :class:`NACAMultiTaskModel`.
        :rtype: int
        """
        return len(self.input_cols)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ARCHITECTURE MULTI-TÂCHES PROFONDE
# ─────────────────────────────────────────────────────────────────────────────
class NACAMultiTaskModel:
    """Réseau de neurones multi-tâches prédisant CL, CD, CM à partir de la physique.

    L'architecture suit un schéma *tronc commun + branches expertes* :
    un tronc partagé de trois couches denses (512, 512, 256 neurones,
    activation ``swish``, avec ``BatchNormalization`` et ``Dropout``)
    apprend une représentation générique des interactions entre
    géométrie et conditions de vol ; cette représentation est ensuite
    transmise à trois branches indépendantes — une par coefficient — qui
    se spécialisent chacune via un bloc résiduel avant de produire une
    sortie scalaire linéaire.

    :ivar input_dim: Dimension du vecteur d'entrée du réseau.
    :vartype input_dim: int
    :ivar learning_rate: Taux d'apprentissage initial de l'optimiseur
        Adam.
    :vartype learning_rate: float
    :ivar model: Modèle Keras fonctionnel compilé, prêt pour
        l'entraînement (``model.fit``) ou l'inférence
        (``model.predict``).
    :vartype model: tensorflow.keras.Model
    """

    def __init__(self, input_dim: int, learning_rate: float = 5e-4):
        """Construit et compile le modèle multi-tâches.

        :param input_dim: Dimension du vecteur d'entrée (nombre de
            features physiques et géométriques), typiquement
            :attr:`NACAAeroPreprocessor.input_dim`.
        :type input_dim: int
        :param learning_rate: Taux d'apprentissage initial transmis à
            l'optimiseur Adam.
        :type learning_rate: float
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = self._build()

    def _build_expert_branch(self, base_tensor, name: str):
        """Construit une branche dédiée à la prédiction d'une force ou d'un moment.

        La branche enchaîne deux couches denses (256 puis 128 neurones)
        avec normalisation par batch et dropout, puis ajoute une
        connexion résiduelle (``layers.add``) entre la sortie de la
        deuxième couche dense (``x_skip``) et celle d'une troisième
        couche dense de même largeur, avant de produire une sortie
        scalaire linéaire portant le nom de la cible (``name``).

        :param base_tensor: Tenseur de sortie du tronc commun, partagé
            entre les trois branches expertes.
        :type base_tensor: tensorflow.Tensor
        :param name: Nom de la cible associée à cette branche (``"CL"``,
            ``"CD"`` ou ``"CM"``) ; utilisé pour nommer les couches et la
            sortie finale, ce qui permet à Keras d'associer correctement
            chaque sortie à sa fonction de perte lors de la compilation.
        :type name: str
        :returns: Tenseur de sortie scalaire (activation linéaire) pour
            la cible considérée.
        :rtype: tensorflow.Tensor
        """
        x = layers.Dense(256, activation="swish", name=f"dense_{name}_1")(base_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        x_skip = layers.Dense(128, activation="swish", name=f"dense_{name}_2")(x)
        x_skip = layers.BatchNormalization()(x_skip)

        x = layers.Dense(128, activation="swish", name=f"dense_{name}_3")(x_skip)
        x = layers.BatchNormalization()(x)

        x = layers.add([x, x_skip], name=f"res_{name}")
        out = layers.Dense(1, activation="linear", name=name)(x)
        return out

    def _build(self) -> keras.Model:
        """Assemble le tronc commun, les trois branches expertes, et compile le modèle.

        Le modèle est compilé avec une fonction de perte ``"mse"`` et une
        métrique ``"mae"`` identiques pour les trois sorties, et
        l'optimiseur Adam paramétré par :attr:`learning_rate`. Comme les
        trois cibles sont standardisées indépendamment en amont (via
        :class:`NACAAeroPreprocessor`), elles contribuent à parts égales
        à la perte totale sans pondération additionnelle.

        :returns: Modèle Keras fonctionnel compilé, à trois entrées
            nommées ``"CL"``, ``"CD"``, ``"CM"`` (une seule entrée
            ``"physical_features"`` mais trois sorties).
        :rtype: tensorflow.keras.Model
        """
        inp = keras.Input(shape=(self.input_dim,), name="physical_features")

        # Tronc commun : apprentissage des interactions géométrie / écoulement
        x = layers.Dense(512, activation="swish", name="shared_1")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(512, activation="swish", name="shared_2")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(256, activation="swish", name="shared_3")(x)
        x = layers.BatchNormalization()(x)

        # Branches de spécialisation physique
        out_cl = self._build_expert_branch(x, "CL")
        out_cd = self._build_expert_branch(x, "CD")
        out_cm = self._build_expert_branch(x, "CM")

        model = keras.Model(inputs=inp, outputs=[out_cl, out_cd, out_cm], name="NACA_PhysicsNet")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={"CL": "mse", "CD": "mse", "CM": "mse"},
            metrics={"CL": "mae", "CD": "mae", "CM": "mae"}
        )
        return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE D'ENTRAÎNEMENT GÉNÉRALISABLE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Cette section constitue le script d'entraînement exécuté lorsque le
    # module est lancé directement (et non importé). Elle n'est pas
    # documentée via une docstring de fonction puisqu'il s'agit du corps
    # du script ``__main__`` ; chaque étape est commentée individuellement
    # ci-dessous pour faciliter la génération d'une documentation narrative
    # (par exemple via une directive Sphinx ``literalinclude``).

    csv_file_path = "dataset_aeroXfoil.csv"

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Le fichier '{csv_file_path}' est introuvable.")

    print(f"Chargement de la base de données entière : {csv_file_path}")
    df_raw = pd.read_csv(csv_file_path)

    # Séparation des ensembles d'entraînement et de test (85% / 15%).
    # Le jeu de test (df_test) est isolé dès cette étape et n'intervient à
    # aucun moment dans l'ajustement des scalers ni dans l'entraînement du
    # réseau, afin de fournir une estimation non biaisée de la performance
    # de généralisation.
    df_train, df_test = train_test_split(df_raw, test_size=0.15, random_state=42)

    # Initialisation du préprocesseur basé sur la géométrie pure et
    # ajustement de ses scalers sur le seul ensemble d'entraînement.
    preprocessor = NACAAeroPreprocessor()
    X_train_scaled, y_train_dict = preprocessor.fit_transform(df_train)

    # Séparation entraînement/validation (sur les données déjà standardisées)
    # pour l'early stopping et la réduction du taux d'apprentissage.
    idx_t, idx_v = train_test_split(np.arange(len(X_train_scaled)), test_size=0.15, random_state=42)
    X_t, X_v = X_train_scaled[idx_t], X_train_scaled[idx_v]
    y_t = {k: v[idx_t] for k, v in y_train_dict.items()}
    y_v = {k: v[idx_v] for k, v in y_train_dict.items()}

    # Création du modèle multi-tâches physique, dimensionné selon le
    # nombre de features défini par le préprocesseur.
    mt_model = NACAMultiTaskModel(input_dim=preprocessor.input_dim, learning_rate=5e-4)

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,  # Patience augmentée pour stabiliser l'apprentissage des lois physiques
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n[ML] Lancement de l'apprentissage à haute capacité ({preprocessor.input_dim} descripteurs physiques)...")
    mt_model.model.fit(
        X_t, y_t,
        validation_data=(X_v, y_v),
        epochs=150,
        batch_size=512,
        callbacks=cbs,
        verbose=1
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. EXPORTATION DES ARTIFACTS (NOMS DE FICHIERS IDENTIQUES)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[EXPORT] Sauvegarde des modèles et configurations...")

    # Sauvegarde du modèle Keras au format .keras. Ce chemin doit rester
    # identique à MODEL_PATH dans dashboard.py pour que le dashboard puisse
    # charger ce modèle sans modification de configuration.
    mt_model.model.save("naca_multitask_model.keras")

    # Sauvegarde du préprocesseur (scalers physiques) avec pickle. La classe
    # NACAAeroPreprocessor doit rester importable (ou reconstruite à
    # l'identique, comme c'est le cas dans dashboard.py) côté
    # désérialisation pour que pickle.load fonctionne.
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Sauvegarde du jeu de test isolé pour analyses ultérieures (audit,
    # calcul de métriques de généralisation hors échantillon d'entraînement).
    df_test.to_csv("dataset_test_isolated.csv", index=False)

    print("→ 'naca_multitask_model.keras' enregistré.")
    print("→ 'preprocessor.pkl' enregistré.")
    print("→ 'dataset_test_isolated.csv' enregistré pour l'analyse.")
    print("Entraînement terminé avec succès.")
