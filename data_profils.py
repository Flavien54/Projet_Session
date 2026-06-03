"""
DatasetBuilder — Construction complète du dataset AeroPredict
MGA 802 · AeroPredict

Un seul script qui :
  1. Génère la grille paramétrique NACA 4 chiffres
  2. Charge les profils UIUC embarqués dans AeroSandbox
  3. Sauvegarde progressivement dans un seul dataset.csv

Usage :
    python dataset_builder.py             # dataset complet
    python dataset_builder.py --test      # 10 profils, test rapide
"""

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
    MODEL_SIZE  : précision NeuralFoil ('small','medium','large','xlarge')

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
    MODEL_SIZE  = "large"

    # ── Grille NACA ─────────────────────────────────────────────────
    M_RANGE = [0, 1, 2, 3, 4, 5, 6]
    P_RANGE = [1, 2, 3, 4, 5, 6]
    T_RANGE = [6, 8, 10, 12, 15, 18, 21, 24]

    # ── Familles UIUC ───────────────────────────────────────────────
    UIUC_FAMILIES = ["e", "fx", "s", "ag", "mh", "hq", "sd", "clarky", "goe"]

    def __init__(
        self,
        output_path : str   = "dataset.csv",
        model_size  : str   = None,
        max_profils : int   = None,     # None = tous, entier = limite test
    ):
        self.output_path = output_path
        self.model_size  = model_size or self.MODEL_SIZE
        self.max_profils = max_profils
