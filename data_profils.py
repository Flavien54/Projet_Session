import pandas
import numpy as np
import aerosandbox as asb
import pathlib

class Dataset_profil:
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

    # Paramètres de simulation
    Alpha_range = np.arange(-6,20,2,dtype=float)
    Re_range = np.array([5e4,1e5,5e5,1e6])
    MODEL_SIZE = "large"

    # Grille NACA
    M_range = np.array([0,1,2,3,4,5,6])
    P_range = np.array([1,2,3,4,5,6])
    T_range = np.array([6,8,10,12,15,18,21,24])

    # ── Familles UIUC ───────────────────────────────────────────────
    UIUC_FAMILIES = ["e", "fx", "s", "ag", "mh", "hq", "sd", "clarky", "goe"]

    def __init__(self,
                 output_path = "dataset.csv",
                 model_size = None,
                 max_profils = None,
                 ):
        self.output_path = output_path
        self.model_size = model_size or self.MODEL_SIZE
        self.max_profils = max_profils


    


