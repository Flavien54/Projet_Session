"""Tests unitaires de main.py (LecteurYAML et AeroPredictApp)."""

import pytest

from main import AeroPredictApp, LecteurYAML


class TestLecteurYAML:
    def test_lecture_deck_valide(self, tmp_path):
        deck = tmp_path / "deck.yaml"
        deck.write_text(
            "chemins:\n"
            "  dataset_xfoil: data.csv\n"
            "xfoil:\n"
            "  max_coeurs: 4\n"
            "mode_test: true\n",
            encoding="utf-8",
        )
        contenu = LecteurYAML(str(deck)).read_yaml()
        assert contenu["chemins"]["dataset_xfoil"] == "data.csv"
        assert contenu["xfoil"]["max_coeurs"] == 4
        assert contenu["mode_test"] is True

    def test_fichier_inexistant(self):
        with pytest.raises(FileNotFoundError):
            LecteurYAML("deck_qui_nexiste_pas.yaml").read_yaml()


class TestAeroPredictApp:
    def test_configuration_chargee(self):
        config = {"chemins": {"dataset_xfoil": "x.csv"}, "mode_test": True}
        app = AeroPredictApp(config)
        assert app.chemins["dataset_xfoil"] == "x.csv"
        assert app.mode_test is True

    def test_config_vide_valeurs_par_defaut(self):
        app = AeroPredictApp({})
        assert app.chemins == {}
        assert app.mode_test is False

    def test_audit_dataset_introuvable(self, capsys):
        # Fichier absent : message clair, pas d'exception
        app = AeroPredictApp({"chemins": {"dataset_xfoil": "inexistant.csv"}})
        app.auditer_dataset()
        sortie = capsys.readouterr().out
        assert "introuvable" in sortie
