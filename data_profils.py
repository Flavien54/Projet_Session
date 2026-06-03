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
        
    def _uiuc_profiles(self):
        """
        Retourne la liste des profils UIUC disponibles dans AeroSandbox,
        filtrés par famille, hors profils NACA.
        """
        pkg_path  = pathlib.Path(asb.__file__).parent
        dat_files = pkg_path.rglob("*.dat")
        all_names = [f.stem for f in dat_files]

        selected = []
        for name in sorted(set(all_names)):
            if name.startswith("naca"):
                continue                    # déjà couverts par la grille NACA
            for famille in self.UIUC_FAMILIES:
                if name.startswith(famille):
                    selected.append({
                        "name"   : name,
                        "m"      : None,    # pas de paramètre NACA
                        "p"      : None,
                        "t"      : None,    # calculé via max_thickness()
                        "camber" : None,    # calculé via max_camber()
                        "source" : "uiuc",
                    })
                    break
        return selected
        
    def _all_profiles(self):
        """Fusionne NACA + UIUC et applique la limite max_profils."""
        naca = self._naca_profiles()
        uiuc = self._uiuc_profiles()
        all_p = naca + uiuc

        if self.max_profils:
            all_p = all_p[: self.max_profils]

        print(f"\n  Profils NACA  : {len(naca)}")
        print(f"  Profils UIUC  : {len(uiuc)}")
        print(f"  Total          : {len(all_p)}"
              + (f"  (limité à {self.max_profils})" if self.max_profils else ""))
        sims = len(all_p) * len(self.ALPHA_RANGE) * len(self.RE_RANGE)
        print(f"  Combinaisons   : ~{sims:,}  "
              f"({len(self.ALPHA_RANGE)} α × {len(self.RE_RANGE)} Re)")
        return all_p

     def _generate_combinations(self, data):
        """
        Génère toutes les combinaisons (alpha, Re) pour un profil.

        Pour les profils UIUC :
            - m = None
            - p = None
            - t = None
            - camber calculé par AeroSandbox

        Retourne un DataFrame avec les paramètres géométriques et conditions.
        """
        name = data["name"]

        try:

            if data["source"] == "uiuc":
                af = asb.Airfoil(name)

                m_val = "None"
                p_val = "None"
                t_val = "None"

                camber_val = round(float(af.max_camber()), 6)

            else:
                m_val = data["m"]
                p_val = data["p"]
                t_val = data["t"]
                camber_val = data["camber"] if data["camber"] is not None else 0.0

            rows = []

            for re in self.RE_RANGE:
                for alpha in self.ALPHA_RANGE:
                    rows.append({
                        "naca": name,
                        "m": m_val,
                        "p": p_val,
                        "t": t_val,
                        "camber": camber_val,
                        "alpha": float(alpha),
                        "Re": float(re),
                        "source": data["source"],
                    })

            return pd.DataFrame(rows)

        except Exception as exc:
            print(f"      {name} : {exc}")

    def build(self):
        """
        Génère toutes les combinaisons de profils et sauvegarde
        progressivement dans output_path.

        Reprise automatique : si le CSV existe déjà, les profils
        déjà présents sont ignorés.
        """      
        print("=" * 60)

        all_profiles = self._all_profiles()

        # ── Reprise après interruption ──────────────────────────
        deja_faits = set()
        first_write = True
        try:
            df_exist    = pd.read_csv(self.output_path)
            deja_faits  = set(df_exist["naca"].unique())
            first_write = False
            print(f"\n  Reprise : {len(deja_faits)} profils déjà générés")
        except FileNotFoundError:
            print(f"\n  Nouveau fichier : {self.output_path}")

        restants = [p for p in all_profiles if p["name"] not in deja_faits]
        print(f"  Restants : {len(restants)} profils\n")

        # ── Boucle de génération ────────────────────────────────
        n_ok  = len(deja_faits)
        n_err = 0
        t0    = time.time()

        for i, data in enumerate(restants, 1):
            df_profil = self._generate_combinations(data)

            if df_profil is not None and not df_profil.empty:
                df_profil.to_csv(
                    self.output_path,
                    mode   = "a",
                    header = first_write,
                    index  = False,
                )
                first_write = False
                n_ok += 1

                elapsed = time.time() - t0
                reste   = (elapsed / i) * (len(restants) - i)
                print(
                    f"  [{i:4d}/{len(restants)}]  "
                    f"{data['name']:22s}  "
                    f"[{data['source']:9s}]  "
                    f"{len(df_profil):3d} lignes  "
                    f"~{reste/60:.1f} min restantes"
                )
            else:
                n_err += 1

        # ── Résumé final ────────────────────────────────────────
        df_final = pd.read_csv(self.output_path)
        self._summary(df_final, n_ok, n_err, time.time() - t0)
        return df_final

    # ══════════════════════════════════════════════════════════════
    # 4. RÉSUMÉ ET STATISTIQUES
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _summary(df: pd.DataFrame, n_ok: int, n_err: int, elapsed: float):
        print(f"\n{'=' * 60}")
        print(f"  Dataset généré avec succès")
        print(f"{'=' * 60}")
        print(f"  Profils générés      : {n_ok}")
        print(f"  Profils en erreur    : {n_err}")
        print(f"  Lignes totales       : {len(df):,}")
        print(f"  Colonnes             : {list(df.columns)}")
        print(f"  Temps total          : {elapsed/60:.1f} min")
        print(f"\n  Par source :")
        print(df.groupby("source").agg(
            profils = ("naca",  "nunique"),
            lignes  = ("naca",  "count"),
        ).to_string())
        print(f"\n  Plages de valeurs :")
        print(f"  Alpha : [{df['alpha'].min():.1f}°, {df['alpha'].max():.1f}°]")
        print(f"  Re    : [{df['Re'].min():.0e}, {df['Re'].max():.0e}]")
        if df['t'].notna().any():
            print(f"  Épaisseur (t) : [{df['t'].min():.3f}, {df['t'].max():.3f}]")
        if df['camber'].notna().any():
            print(f"  Cambrure (camber) : [{df['camber'].min():.3f}, {df['camber'].max():.3f}]")


# ══════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Construit le dataset AeroPredict")
    parser.add_argument("--test",    action="store_true", help="Mode test : 10 profils")
    parser.add_argument("--output",  default="dataset.csv", help="Fichier de sortie")
    parser.add_argument("--size",    default="large",
                        choices=["small", "medium", "large", "xlarge"],
                        )
    args = parser.parse_args()

    builder = DatasetBuilder(
        output_path = "dataset_test.csv" if args.test else args.output,
        max_profils = 10 if args.test else None,
    )

    df = builder.build()
            return None    
    
