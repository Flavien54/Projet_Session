import aerosandbox as asb
import numpy as np
import pandas as pd
import pathlib
import time
import argparse


class DatasetBuilder:
    """
    Construit le dataset complet (NACA + UIUC) pour AeroPredict.

    Paramètres de simulation
    ------------------------
    ALPHA_RANGE : angles d'attaque en degrés
    RE_RANGE    : nombres de Reynolds
    

    Grille NACA
    -----------
    m_range : cambrure max        (1er chiffre NACA / 100)
    p_range : position cambrure   (2e  chiffre NACA / 10)
    t_range : épaisseur relative  (3e+4e chiffres   / 100)

    Familles UIUC retenues
    ----------------------
    Eppler (e), Wortmann FX (fx), NREL S (s), AG (ag),
    MH (mh), HQ (hq), Selig-Donovan (sd), Clark Y, Göttingen (goe)
    """

    # ── Paramètres de simulation ────────────────────────────────────
    ALPHA_RANGE = np.arange(-6, 24, 0.5, dtype=float)
    RE_RANGE    = [5e4, 1e5, 5e5, 1e6, 5e6]
    

    # ── Grille NACA ─────────────────────────────────────────────────
    M_RANGE = [0, 1, 2, 3, 4, 5, 6]
    P_RANGE = [1, 2, 3, 4, 5, 6]
    T_RANGE = [6, 8, 10, 12, 15, 18, 21, 24]

    # ── Familles UIUC ───────────────────────────────────────────────
    UIUC_FAMILIES = ["e", "fx", "s", "ag", "mh", "hq", "sd", "clarky", "goe"]

    def __init__(
        self,
        output_path : str   = "dataset.csv",
        max_profils : int   = None,     # None = tous, entier = limite test
    ):
        self.output_path = output_path
        self.max_profils = max_profils
      
    def _naca_profiles(self):
        """
        Retourne la liste des profils NACA 4 chiffres avec leurs
        paramètres m, p, t.
        Les profils m=0 sont symétriques quelle que soit p → afin de pas générer de bruit, aucune modèle symétrique est utilisé.
        """
        profiles = []

        for m in self.M_RANGE:
            for p in self.P_RANGE:

                # Profil symétrique (m=0) : p n'a aucun effet géométrique
                # → on ne garde qu'une seule valeur de p pour éviter les doublons
                if m == 0 and p != self.P_RANGE[0]:
                    continue

                for t in self.T_RANGE:
                    profiles.append({
                        "name"   : f"naca{m}{p}{t:02d}",
                        "m"      : m / 100,
                        "p"      : p / 10,
                        "t"      : t / 100,
                        "camber" : None,       # calculé analytiquement via m
                        "source" : "naca_grid",
                    })

        return profiles
