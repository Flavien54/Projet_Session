"""AeroPredict — Point d'entrée principal du module.

MGA 802 · Optimisation de profils aérodynamiques assistée par Machine Learning

Ce script orchestre l'ensemble du pipeline du projet via un menu interactif :
  1. Construction du dataset géométrique (NACA + UIUC, via AeroSandbox)
  2. Enrichissement aérodynamique par XFoil (CL, CD, CM)
  3. Entraînement du modèle de Machine Learning (TensorFlow)
  4. Inférence du modèle sur le dataset complet
  5. Audit et rapports PDF du dataset
  6. Lancement du dashboard interactif (Streamlit)

La configuration (chemins des fichiers, paramètres XFoil) est lue depuis un
deck YAML (deck.yaml par défaut), conformément à l'interface utilisateur
recommandée dans le cours (Module 9).

Usage:
    python main.py                 # deck.yaml par défaut
    python main.py mon_deck.yaml   # deck personnalisé
"""

import os
import subprocess
import sys

import yaml


class LecteurYAML:
    """Lit et valide le deck de configuration YAML du projet.

    Attributes:
        file_path (str): Chemin du fichier YAML à lire.
    """

    def __init__(self, file_path: str) -> None:
        """Initialise le lecteur.

        Args:
            file_path: Chemin du fichier YAML de configuration.
        """
        self.file_path = file_path

    def read_yaml(self) -> dict:
        """Charge le contenu du fichier YAML.

        Returns:
            Le contenu du deck sous forme de dictionnaire Python.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            yaml.YAMLError: Si le fichier est mal formé.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Deck YAML introuvable : '{self.file_path}'. "
                "Complétez deck.yaml avec vos paramètres avant de lancer le programme."
            )
        with open(self.file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def get_int_input(prompt: str) -> int:
    """Demande un entier à l'utilisateur en boucle jusqu'à saisie valide.

    Args:
        prompt: Message affiché à l'utilisateur.

    Returns:
        L'entier saisi.
    """
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Entrée invalide, veuillez saisir un nombre entier.")


class AeroPredictApp:
    """Application principale AeroPredict : menu interactif du pipeline.

    Chaque étape du pipeline est encapsulée dans une méthode. Les dépendances
    lourdes (AeroSandbox, TensorFlow) sont importées paresseusement afin que
    les étapes de visualisation restent utilisables sans elles.

    Attributes:
        config (dict): Configuration chargée depuis le deck YAML.
    """

    def __init__(self, config: dict) -> None:
        """Initialise l'application.

        Args:
            config: Dictionnaire de configuration issu du deck YAML.
        """
        self.config = config
        self.chemins = config.get("chemins", {})
        self.mode_test = bool(config.get("mode_test", False))

    # ── Étapes du pipeline ───────────────────────────────────────

    def construire_dataset(self) -> None:
        """Étape 1 : génère le dataset géométrique NACA + UIUC."""
        try:
            from data_profils import DatasetBuilder
        except ImportError as exc:
            print(f"\n  AeroSandbox requis pour cette étape : {exc}")
            print("  Installation : pip install aerosandbox")
            return

        builder = DatasetBuilder(
            output_path=self.chemins.get("dataset_geometrie", "dataset_profil.csv"),
            max_profils=10 if self.mode_test else None,
        )
        builder.build()

    def enrichir_xfoil(self) -> None:
        """Étape 2 : enrichit le dataset avec les coefficients XFoil."""
        try:
            from calcul_Xfoil import XFoilDatasetBuilder
        except ImportError as exc:
            print(f"\n  AeroSandbox requis pour cette étape : {exc}")
            return

        xfoil_cfg = self.config.get("xfoil", {})
        if not os.path.exists(xfoil_cfg.get("executable", "")):
            print(f"\n  Exécutable XFoil introuvable : {xfoil_cfg.get('executable')}")
            print("  Vérifiez le chemin 'xfoil: executable' dans le deck YAML.")
            return

        builder = XFoilDatasetBuilder(
            input_csv=self.chemins.get("dataset_geometrie", "dataset_profil.csv"),
            output_csv=self.chemins.get("dataset_xfoil", "dataset_aeroXfoil.csv"),
            max_cores=int(xfoil_cfg.get("max_coeurs", 8)),
        )
        builder.build()

    def entrainer_modele(self) -> None:
        """Étape 3 : entraîne le réseau de neurones multi-tâches."""
        self._lancer_script("modele_ML.py", besoin="TensorFlow")

    def appliquer_modele(self) -> None:
        """Étape 4 : applique le modèle entraîné sur le dataset complet."""
        self._lancer_script("Application_modele_ML.py", besoin="TensorFlow")

    def auditer_dataset(self) -> None:
        """Étape 5 : génère les rapports PDF d'audit du dataset."""
        from audit_dataset import audit_dataset

        fichier = self.chemins.get("dataset_xfoil", "dataset_aeroXfoil.csv")
        if not os.path.exists(fichier):
            print(f"\n  Dataset introuvable : {fichier}")
            return
        audit_dataset(fichier, self.chemins.get("rapports_audit", "audit_reports"))

    def lancer_dashboard(self) -> None:
        """Étape 6 : démarre le dashboard Streamlit dans le navigateur."""
        print("\n  Lancement du dashboard (Ctrl+C pour arrêter)...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
            check=False,
        )

    # ── Utilitaires ──────────────────────────────────────────────

    def _lancer_script(self, script: str, besoin: str) -> None:
        """Exécute un script du pipeline dans un sous-processus.

        Args:
            script: Nom du fichier Python à exécuter.
            besoin: Nom de la dépendance principale (pour le message d'erreur).
        """
        print(f"\n  Exécution de {script} ({besoin} requis)...")
        resultat = subprocess.run([sys.executable, script], check=False)
        if resultat.returncode != 0:
            print(f"\n  {script} s'est terminé avec une erreur "
                  f"(code {resultat.returncode}). {besoin} est-il installé ?")

    def afficher_configuration(self) -> None:
        """Affiche les paramètres du deck pour validation par l'utilisateur."""
        print("\n  Configuration chargée :")
        for section, valeurs in self.config.items():
            if isinstance(valeurs, dict):
                print(f"    {section}:")
                for cle, val in valeurs.items():
                    print(f"      {cle}: {val}")
            else:
                print(f"    {section}: {valeurs}")

    # ── Boucle principale ────────────────────────────────────────

    MENU = """
══════════════════════════════════════════════════════
  AeroPredict — Pipeline aérodynamique + Machine Learning
══════════════════════════════════════════════════════
  1. Construire le dataset géométrique (AeroSandbox)
  2. Enrichir le dataset via XFoil (CL, CD, CM)
  3. Entraîner le modèle ML (TensorFlow)
  4. Appliquer le modèle ML (prédictions)
  5. Auditer le dataset (rapports PDF)
  6. Lancer le dashboard interactif (Streamlit)
  7. Afficher la configuration
  0. Quitter
"""

    def run(self) -> None:
        """Boucle du menu interactif principal."""
        actions = {
            1: self.construire_dataset,
            2: self.enrichir_xfoil,
            3: self.entrainer_modele,
            4: self.appliquer_modele,
            5: self.auditer_dataset,
            6: self.lancer_dashboard,
            7: self.afficher_configuration,
        }

        while True:
            print(self.MENU)
            choix = get_int_input("  Votre choix : ")
            if choix == 0:
                print("  Au revoir !")
                break
            action = actions.get(choix)
            if action is None:
                print("  Choix invalide, options 0 à 7.")
                continue
            action()


if __name__ == "__main__":
    deck_path = sys.argv[1] if len(sys.argv) > 1 else "deck.yaml"

    lecteur = LecteurYAML(deck_path)
    configuration = lecteur.read_yaml()

    app = AeroPredictApp(configuration)
    app.afficher_configuration()
    app.run()
