"""Tests unitaires de data_profils.DatasetBuilder.

Vérifie la génération combinatoire des profils NACA, le balayage des
conditions de vol et la structure du DataFrame produit. L'extraction
géométrique AeroSandbox est remplacée par un stub (aucune simulation).
"""

import numpy as np
import pandas as pd
import pytest

from data_profils import DatasetBuilder


# Géométrie factice retournée à la place d'AeroSandbox
GEOMETRIE_STUB = {
    "t": 0.12,
    "camber": 0.02,
    "x_t": 0.30,
    "x_c": 0.40,
    "LE_radius": 0.015,
    "TE_angle": 12.0,
    "t_over_xt": 0.40,
    "area": 0.08,
}


@pytest.fixture
def builder():
    return DatasetBuilder(output_path="dataset_inexistant.csv")


# ── Grille NACA ──────────────────────────────────────────────────

class TestNacaProfiles:
    def test_nombre_total_de_profils(self, builder):
        # m=0 : 1 seul p (symétrique) x 8 épaisseurs = 8
        # m=1..6 : 6 p x 8 t = 48 chacun -> 288
        profils = builder._naca_profiles()
        assert len(profils) == 296

    def test_pas_de_doublons(self, builder):
        noms = [p["name"] for p in builder._naca_profiles()]
        assert len(noms) == len(set(noms))

    def test_format_des_noms(self, builder):
        noms = [p["name"] for p in builder._naca_profiles()]
        # 4 chiffres après 'naca', épaisseur toujours sur 2 chiffres
        assert "naca0106" in noms      # m=0, p=1, t=6 -> '06'
        assert "naca6624" in noms      # m=6, p=6, t=24
        for nom in noms:
            assert nom.startswith("naca")
            assert len(nom) == 8
            assert nom[4:].isdigit()

    def test_profils_symetriques_un_seul_p(self, builder):
        # Pour m=0, p n'a aucun effet géométrique : une seule valeur gardée
        noms = [p["name"] for p in builder._naca_profiles()]
        symetriques = [n for n in noms if n[4] == "0"]
        assert len(symetriques) == len(builder.T_RANGE)
        # Tous avec p = première valeur de P_RANGE
        assert all(n[5] == str(builder.P_RANGE[0]) for n in symetriques)

    def test_source_naca_grid(self, builder):
        assert all(p["source"] == "naca_grid" for p in builder._naca_profiles())


# ── Plages de conditions de vol ──────────────────────────────────

class TestPlagesConditions:
    def test_alpha_range(self):
        # 60 valeurs de -6.0 à 23.5 par pas de 0.5
        assert len(DatasetBuilder.ALPHA_RANGE) == 60
        assert DatasetBuilder.ALPHA_RANGE[0] == -6.0
        assert DatasetBuilder.ALPHA_RANGE[-1] == 23.5
        assert np.allclose(np.diff(DatasetBuilder.ALPHA_RANGE), 0.5)

    def test_re_range(self):
        assert DatasetBuilder.RE_RANGE == [5e4, 1e5, 5e5, 1e6, 5e6]


# ── Génération des combinaisons ──────────────────────────────────

class TestGenerateCombinations:
    def test_300_lignes_par_profil(self, builder, monkeypatch):
        monkeypatch.setattr(builder, "_get_geometry", lambda name: GEOMETRIE_STUB)
        df = builder._generate_combinations({"name": "naca2412", "source": "naca_grid"})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 60 * 5  # 60 alpha x 5 Re

    def test_colonnes_attendues(self, builder, monkeypatch):
        monkeypatch.setattr(builder, "_get_geometry", lambda name: GEOMETRIE_STUB)
        df = builder._generate_combinations({"name": "naca2412", "source": "naca_grid"})
        colonnes = ["naca", "source", "t", "camber", "x_t", "x_c",
                    "LE_radius", "TE_angle", "t_over_xt", "area", "alpha", "Re"]
        assert list(df.columns) == colonnes

    def test_geometrie_constante_par_profil(self, builder, monkeypatch):
        monkeypatch.setattr(builder, "_get_geometry", lambda name: GEOMETRIE_STUB)
        df = builder._generate_combinations({"name": "naca2412", "source": "naca_grid"})
        for cle, valeur in GEOMETRIE_STUB.items():
            assert (df[cle] == valeur).all()

    def test_toutes_les_combinaisons_presentes(self, builder, monkeypatch):
        monkeypatch.setattr(builder, "_get_geometry", lambda name: GEOMETRIE_STUB)
        df = builder._generate_combinations({"name": "naca2412", "source": "naca_grid"})
        assert df["alpha"].nunique() == 60
        assert df["Re"].nunique() == 5
        assert not df.duplicated(subset=["alpha", "Re"]).any()

    def test_erreur_geometrie_retourne_none(self, builder, monkeypatch):
        def geometrie_en_erreur(name):
            raise ValueError("profil illisible")
        monkeypatch.setattr(builder, "_get_geometry", geometrie_en_erreur)
        resultat = builder._generate_combinations({"name": "casse", "source": "uiuc"})
        assert resultat is None


# ── Limitation du nombre de profils (mode test) ──────────────────

class TestMaxProfils:
    def test_troncature(self, monkeypatch, capsys):
        builder = DatasetBuilder(output_path="x.csv", max_profils=10)
        profils = builder._all_profiles()
        assert len(profils) == 10
