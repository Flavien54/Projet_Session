"""AeroPredict — Point d'entrée principal du module.

MGA 802 · Optimisation de profils aérodynamiques assistée par Machine Learning

Ce script orchestre l'ensemble du pipeline du projet via un menu interactif :

#. Construction du dataset géométrique (NACA + UIUC, via AeroSandbox) ;
#. Enrichissement aérodynamique par XFoil (``CL``, ``CD``, ``CM``) ;
#. Entraînement du modèle de Machine Learning (TensorFlow) ;
#. Inférence du modèle sur le dataset complet ;
#. Audit et rapports PDF du dataset ;
#. Lancement du dashboard interactif (Streamlit).

La configuration (chemins des fichiers, paramètres XFoil) est lue depuis un
deck YAML (``deck.yaml`` par défaut), conformément à l'interface utilisateur
recommandée dans le cours (Module 9).

Ce fichier est documenté au format reStructuredText (compatible
`Sphinx <https://www.sphinx-doc.org/>`_ et son extension ``autodoc``). Pour
générer la documentation HTML, il suffit d'inclure ce module dans la
configuration Sphinx (``conf.py``) puis d'exécuter ``sphinx-build``.

:Usage:

    .. code-block:: bash

        python main.py                 # deck.yaml par défaut
        python main.py mon_deck.yaml   # deck personnalisé
"""

import os
import subprocess
import sys

import yaml


class LecteurYAML:
    """Lit et valide le deck de configuration YAML du projet.

    :ivar file_path: Chemin du fichier YAML à lire.
    :vartype file_path: str
    """

    def __init__(self, file_path: str) -> None:
        """Initialise le lecteur avec le chemin du fichier de configuration.

        :param file_path: Chemin du fichier YAML de configuration.
        :type file_path: str
        """
        self.file_path = file_path

    def read_yaml(self) -> dict:
        """Charge et désérialise le contenu du fichier YAML.

        :returns: Le contenu du deck sous forme de dictionnaire Python,
            tel que produit par ``yaml.safe_load``.
        :rtype: dict
        :raises FileNotFoundError: Si :attr:`file_path` ne correspond à
            aucun fichier existant.
        :raises yaml.YAMLError: Si le fichier existe mais n'est pas un
            YAML syntaxiquement valide.
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

    Toute saisie non convertible en ``int`` (par exemple une chaîne vide
    ou du texte) déclenche un message d'erreur et une nouvelle demande,
    sans jamais lever d'exception vers l'appelant.

    :param prompt: Message affiché à l'utilisateur lors de la demande de
        saisie (transmis directement à ``input``).
    :type prompt: str
    :returns: L'entier saisi par l'utilisateur.
    :rtype: int
    """
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Entrée invalide, veuillez saisir un nombre entier.")


class AeroPredictApp:
    """Application principale AeroPredict : menu interactif du pipeline.

    Chaque étape du pipeline (construction du dataset, calcul XFoil,
    entraînement et application du modèle ML, audit, dashboard) est
    encapsulée dans une méthode dédiée. Les dépendances lourdes
    (`AeroSandbox <https://aerosandbox.readthedocs.io/>`_, TensorFlow) sont
    importées paresseusement (à l'intérieur des méthodes, et non en tête
    de module) afin que les étapes ne les nécessitant pas — comme
    l'affichage de la configuration ou le lancement du dashboard — restent
    utilisables même si ces paquets ne sont pas installés.

    :ivar config: Dictionnaire de configuration complet issu du deck
        YAML.
    :vartype config: dict
    :ivar chemins: Sous-section ``"chemins"`` de :attr:`config` (chemins
        des fichiers d'entrée/sortie du pipeline) ; dictionnaire vide si
        absente du deck.
    :vartype chemins: dict
    :ivar mode_test: Indique si le pipeline doit s'exécuter en mode
        test réduit (par exemple, nombre de profils limité lors de la
        construction du dataset). Lu depuis la clé ``"mode_test"`` du
        deck, ``False`` par défaut.
    :vartype mode_test: bool
    """

    def __init__(self, config: dict) -> None:
        """Initialise l'application à partir de la configuration chargée.

        :param config: Dictionnaire de configuration issu du deck YAML
            (typiquement produit par :meth:`LecteurYAML.read_yaml`).
        :type config: dict
        """
        self.config = config
        self.chemins = config.get("chemins", {})
        self.mode_test = bool(config.get("mode_test", False))

    # ── Étapes du pipeline ───────────────────────────────────────

    def construire_dataset(self) -> None:
        """Étape 1 — génère le dataset géométrique NACA + UIUC.

        Délègue la construction à
        :class:`data_profils.DatasetBuilder` (import paresseux, car cette
        classe dépend d'AeroSandbox). En :attr:`mode_test`, le nombre de
        profils générés est limité à 10 afin d'accélérer les itérations
        de développement.

        :returns: Rien. En cas d'absence d'AeroSandbox, affiche un
            message d'aide à l'installation et retourne sans lever
            d'exception (l'étape est simplement ignorée).
        :rtype: None
        """
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
        """Étape 2 — enrichit le dataset géométrique avec les coefficients XFoil.

        Lit le chemin de l'exécutable XFoil ainsi que le nombre de cœurs
        à utiliser pour la parallélisation depuis la section ``"xfoil"``
        du deck YAML (clés ``"executable"`` et ``"max_coeurs"``), puis
        délègue le calcul à
        :class:`calcul_Xfoil.XFoilDatasetBuilder` (import paresseux, car
        cette classe dépend d'AeroSandbox pour piloter XFoil).

        :returns: Rien. Affiche un message d'erreur et retourne sans
            lever d'exception si AeroSandbox n'est pas installé ou si
            l'exécutable XFoil configuré est introuvable sur le disque.
        :rtype: None
        """
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
        """Étape 3 — entraîne le réseau de neurones multi-tâches.

        Délègue l'exécution au script externe ``modele_ML.py`` via
        :meth:`_lancer_script`, dans un sous-processus Python séparé
        (et non par import direct), afin d'isoler l'empreinte mémoire de
        TensorFlow du reste de l'application.

        :returns: Rien.
        :rtype: None
        """
        self._lancer_script("modele_ML.py", besoin="TensorFlow")

    def appliquer_modele(self) -> None:
        """Étape 4 — applique le modèle ML entraîné sur le dataset complet.

        Délègue l'exécution au script externe
        ``Application_modele_ML.py`` via :meth:`_lancer_script`, qui
        génère typiquement le fichier de prédictions consommé ensuite
        par le dashboard (voir ``CSV_ML`` dans ``dashboard.py``).

        :returns: Rien.
        :rtype: None
        """
        self._lancer_script("Application_modele_ML.py", besoin="TensorFlow")

    def auditer_dataset(self) -> None:
        """Étape 5 — génère les rapports PDF d'audit du dataset XFoil.

        Délègue l'analyse à la fonction
        :func:`audit_dataset.audit_dataset`, en lui passant le chemin du
        dataset XFoil (clé ``"dataset_xfoil"`` du deck) et le répertoire
        de sortie des rapports (clé ``"rapports_audit"``, par défaut
        ``"audit_reports"``).

        :returns: Rien. Affiche un message d'erreur et retourne sans
            lever d'exception si le dataset XFoil configuré est
            introuvable sur le disque.
        :rtype: None
        """
        from audit_dataset import audit_dataset

        fichier = self.chemins.get("dataset_xfoil", "dataset_aeroXfoil.csv")
        if not os.path.exists(fichier):
            print(f"\n  Dataset introuvable : {fichier}")
            return
        audit_dataset(fichier, self.chemins.get("rapports_audit", "audit_reports"))

    def lancer_dashboard(self) -> None:
        """Étape 6 — démarre le dashboard Streamlit dans le navigateur.

        Exécute ``streamlit run dashboard.py`` dans un sous-processus
        bloquant (la méthode ne retourne qu'une fois le serveur
        Streamlit arrêté, par exemple via ``Ctrl+C``).

        :returns: Rien.
        :rtype: None
        """
        print("\n  Lancement du dashboard (Ctrl+C pour arrêter)...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
            check=False,
        )

    # ── Utilitaires ──────────────────────────────────────────────

    def _lancer_script(self, script: str, besoin: str) -> None:
        """Exécute un script Python du pipeline dans un sous-processus.

        Utilise ``subprocess.run`` avec l'interpréteur courant
        (``sys.executable``), garantissant l'exécution dans le même
        environnement (et donc le même ``virtualenv``/``conda env``) que
        le processus appelant. Le code de retour est inspecté pour
        signaler une éventuelle erreur, sans pour autant interrompre le
        menu principal.

        :param script: Nom (ou chemin) du fichier Python à exécuter.
        :type script: str
        :param besoin: Nom de la dépendance principale requise par ce
            script, utilisé uniquement à des fins d'affichage dans les
            messages utilisateur.
        :type besoin: str
        :returns: Rien.
        :rtype: None
        """
        print(f"\n  Exécution de {script} ({besoin} requis)...")
        resultat = subprocess.run([sys.executable, script], check=False)
        if resultat.returncode != 0:
            print(f"\n  {script} s'est terminé avec une erreur "
                  f"(code {resultat.returncode}). {besoin} est-il installé ?")

    def afficher_configuration(self) -> None:
        """Affiche dans la console les paramètres du deck YAML chargé.

        Parcourt :attr:`config` à un seul niveau d'imbrication : les
        valeurs de type ``dict`` (sections, par exemple ``"chemins"`` ou
        ``"xfoil"``) sont affichées clé par clé avec une indentation
        supplémentaire, les autres valeurs sont affichées directement.
        Cette fonction permet à l'utilisateur de valider visuellement la
        configuration avant de lancer une étape potentiellement longue
        du pipeline.

        :returns: Rien.
        :rtype: None
        """
        print("\n  Configuration chargée :")
        for section, valeurs in self.config.items():
            if isinstance(valeurs, dict):
                print(f"    {section}:")
                for cle, val in valeurs.items():
                    print(f"      {cle}: {val}")
            else:
                print(f"    {section}: {valeurs}")

    # ── Boucle principale ────────────────────────────────────────

    #: Texte du menu interactif affiché à chaque itération de
    #: :meth:`run`. Les numéros listés ici doivent rester synchronisés
    #: avec les clés du dictionnaire ``actions`` défini dans :meth:`run`.
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
        """Exécute la boucle du menu interactif principal.

        Affiche :data:`MENU`, demande un choix entier via
        :func:`get_int_input`, puis route vers la méthode d'étape
        correspondante (table de dispatch ``actions``). La boucle se
        poursuit indéfiniment jusqu'à ce que l'utilisateur saisisse
        ``0`` (sortie normale) ; tout choix hors de l'intervalle ``[0,
        7]`` redemande simplement une saisie sans interrompre la boucle.

        :returns: Rien. Cette méthode ne retourne que lorsque
            l'utilisateur choisit de quitter (option ``0``).
        :rtype: None
        """
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
    # Le chemin du deck YAML peut être surchargé en argument de ligne de
    # commande (ex. "python main.py mon_deck.yaml") ; "deck.yaml" est
    # utilisé par défaut.
    deck_path = sys.argv[1] if len(sys.argv) > 1 else "deck.yaml"

    lecteur = LecteurYAML(deck_path)
    configuration = lecteur.read_yaml()

    app = AeroPredictApp(configuration)
    app.afficher_configuration()
    app.run()
