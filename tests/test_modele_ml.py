"""Tests unitaires du préprocesseur ML (NACAAeroPreprocessor).

Vérifie l'encodage des variables catégorielles, la normalisation des
features continues, la standardisation des cibles et l'inversion
d'échelle. Le réseau de neurones lui-même n'est pas testé (nécessite
TensorFlow et un entraînement complet).
"""

import numpy as np
import pandas as pd
import pytest

from modele_ML import NACAAeroPreprocessor


def _dataframe_exemple(n_par_profil: int = 10) -> pd.DataFrame:
    """Construit un petit DataFrame synthétique au format dataset_aeroXfoil."""
    rng = np.random.default_rng(42)
    lignes = []
    for nom, source in [("naca2412", "naca_grid"), ("e205", "uiuc")]:
        for i in range(n_par_profil):
            lignes.append({
                "naca": nom,
                "source": source,
                "t": 0.12, "camber": 0.02, "x_t": 0.30, "x_c": 0.40,
                "LE_radius": 0.015, "TE_angle": 12.0,
                "t_over_xt": 0.40, "area": 0.08,
                "alpha": float(i), "Re": 1e5,
                "CL": float(rng.normal(0.5, 0.3)),
                "CD": float(rng.uniform(0.01, 0.05)),
                "CM": float(rng.normal(-0.05, 0.02)),
            })
    return pd.DataFrame(lignes)


@pytest.fixture
def df():
    return _dataframe_exemple()


@pytest.fixture
def preprocesseur():
    return NACAAeroPreprocessor()


class TestFitTransform:
    def test_dimensions_sortie(self, preprocesseur, df):
        X, y_dict = preprocesseur.fit_transform(df)
        # 2 colonnes catégorielles + 10 continues = 12 features
        assert X.shape == (len(df), 12)
        assert preprocesseur.input_dim == 12

    def test_cibles_standardisees(self, preprocesseur, df):
        _, y_dict = preprocesseur.fit_transform(df)
        assert set(y_dict) == {"CL", "CD", "CM"}
        for col, y in y_dict.items():
            assert y.shape == (len(df), 1)
            # StandardScaler : moyenne ~0, écart-type ~1
            assert abs(float(y.mean())) < 1e-6
            assert float(y.std()) == pytest.approx(1.0, abs=0.05)

    def test_type_float32(self, preprocesseur, df):
        X, y_dict = preprocesseur.fit_transform(df)
        assert X.dtype == np.float32
        assert all(v.dtype == np.float32 for v in y_dict.values())

    def test_marque_comme_ajuste(self, preprocesseur, df):
        assert preprocesseur.is_fitted is False
        preprocesseur.fit_transform(df)
        assert preprocesseur.is_fitted is True


class TestTransform:
    def test_refuse_sans_fit(self, preprocesseur, df):
        with pytest.raises(AssertionError):
            preprocesseur.transform(df)

    def test_coherent_avec_fit_transform(self, preprocesseur, df):
        X_fit, _ = preprocesseur.fit_transform(df)
        X_again = preprocesseur.transform(df)
        np.testing.assert_allclose(X_fit, X_again, rtol=1e-5)

    def test_categorie_inconnue_geree(self, preprocesseur, df):
        preprocesseur.fit_transform(df)
        df_nouveau = df.copy()
        df_nouveau["naca"] = "profil_jamais_vu"
        # Ne doit pas lever : catégorie inconnue rabattue sur la 1re classe
        X = preprocesseur.transform(df_nouveau)
        assert X.shape == (len(df), 12)


class TestInverseTransform:
    def test_aller_retour_exact(self, preprocesseur, df):
        _, y_dict = preprocesseur.fit_transform(df)
        for col in ["CL", "CD", "CM"]:
            valeurs_originales = df[[col]].values
            reconstruites = preprocesseur.inverse_transform_target(y_dict[col], col)
            np.testing.assert_allclose(reconstruites, valeurs_originales, rtol=1e-5)
