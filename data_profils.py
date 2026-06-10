"""
DatasetBuilder — Construction complète du dataset AeroPredict
MGA 802 · AeroPredict

Un seul script qui :
  1. Génère la grille paramétrique NACA 4 chiffres
  2. Charge les profils UIUC embarqués dans AeroSandbox
  3. Extrait 8 features géométriques via AeroSandbox pour chaque profil
  4. Génère toutes les combinaisons (alpha × Re)
  5. Sauvegarde progressivement dans un seul dataset.csv

Les colonnes CL, CD, CM seront ajoutées ultérieurement via XFoil.

Colonnes produites (14) :
    naca        nom du profil
    source      naca_grid ou uiuc
    t           épaisseur relative maximale
    camber      cambrure maximale
    x_t         position du max d'épaisseur
    x_c         position du max de cambrure
    LE_radius   rayon de bord d'attaque
    TE_angle    angle de bord de fuite
    t_over_xt   rapport t / x_t
    area        aire de la section
    alpha       angle d'attaque
    Re          nombre de Reynolds

Résultat attendu :
    1 337 profils × 60 alpha × 5 Re = ~401 100 lignes

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
    Construit le dataset géométrique complet (NACA + UIUC) pour AeroPredict.

    Grille NACA
    -----------
    M_RANGE : cambrure max        (1er chiffre NACA / 100)
    P_RANGE : position cambrure   (2e  chiffre NACA / 10)
    T_RANGE : épaisseur relative  (3e+4e chiffres   / 100)

    Familles UIUC retenues
    ----------------------
    Eppler (e), Wortmann FX (fx), NREL S (s), AG (ag),
    MH (mh), HQ (hq), Selig-Donovan (sd), Clark Y, Göttingen (goe).
    """

    # ── Plages de conditions de vol ─────────────────────────────────
    ALPHA_RANGE = np.arange(-6, 24, 0.5, dtype=float)   # 60 valeurs
    RE_RANGE    = [5e4, 1e5, 5e5, 1e6, 5e6]             # 5 valeurs

    # ── Grille NACA ─────────────────────────────────────────────────
    M_RANGE = [0, 1, 2, 3, 4, 5, 6]
    P_RANGE = [1, 2, 3, 4, 5, 6]
    T_RANGE = [6, 8, 10, 12, 15, 18, 21, 24]

    # ── Familles UIUC ───────────────────────────────────────────────
    UIUC_FAMILIES = ["e", "fx", "s", "ag", "mh", "hq", "sd", "clarky", "goe"]

    def __init__(
        self,
        output_path : str = "dataset.csv",
        max_profils : int = None,
    ):
        self.output_path = output_path
        self.max_profils = max_profils

    # ══════════════════════════════════════════════════════════════
    # 1. GÉNÉRATION DE LA LISTE DES PROFILS
    # ══════════════════════════════════════════════════════════════

    def _naca_profiles(self) -> list:
        """
        Retourne la liste des profils NACA 4 chiffres.
        Les profils m=0 sont symétriques quelle que soit p →
        on n'en garde qu'un seul (p fixé à p_range[0]).
        """
        profiles = []

        for m in self.M_RANGE:
            for p in self.P_RANGE:

                if m == 0 and p != self.P_RANGE[0]:
                    continue

                for t in self.T_RANGE:
                    profiles.append({
                        "name"  : f"naca{m}{p}{t:02d}",
                        "source": "naca_grid",
                    })

        return profiles

    def _uiuc_profiles(self) -> list:
        """
        Retourne les profils UIUC disponibles dans AeroSandbox,
        filtrés par famille, hors profils NACA.
        """
        pkg_path  = pathlib.Path(asb.__file__).parent
        dat_files = pkg_path.rglob("*.dat")
        all_names = [f.stem for f in dat_files]

        selected = []
        for name in sorted(set(all_names)):
            if name.startswith("naca"):
                continue
            for famille in self.UIUC_FAMILIES:
                if name.startswith(famille):
                    selected.append({
                        "name"  : name,
                        "source": "uiuc",
                    })
                    break

        return selected

    def _all_profiles(self) -> list:
        """Fusionne NACA + UIUC et applique la limite max_profils."""
        naca  = self._naca_profiles()
        uiuc  = self._uiuc_profiles()
        all_p = naca + uiuc

        if self.max_profils:
            all_p = all_p[: self.max_profils]

        n_alpha = len(self.ALPHA_RANGE)
        n_re    = len(self.RE_RANGE)
        lignes  = len(all_p) * n_alpha * n_re

        print(f"\n  Profils NACA   : {len(naca)}")
        print(f"  Profils UIUC   : {len(uiuc)}")
        print(f"  Total profils  : {len(all_p)}"
              + (f"  (limité à {self.max_profils})" if self.max_profils else ""))
        print(f"  Alpha          : {n_alpha} valeurs  ({self.ALPHA_RANGE[0]}° → {self.ALPHA_RANGE[-1]}°)")
        print(f"  Re             : {n_re} valeurs  {self.RE_RANGE}")
        print(f"  Lignes totales : ~{lignes:,}  ({n_alpha} α × {n_re} Re × {len(all_p)} profils)")

        return all_p

    # ══════════════════════════════════════════════════════════════
    # 2. EXTRACTION DES 8 FEATURES GÉOMÉTRIQUES
    # ══════════════════════════════════════════════════════════════

    def _get_geometry(self, name: str) -> dict:
        """
        Extrait les 8 features géométriques d'un profil via AeroSandbox.
        Fonctionne pour NACA et UIUC sans aucune simulation.

        Features retournées :
            t           épaisseur relative maximale
            camber      cambrure maximale
            x_t         position du max d'épaisseur (extrados)
            x_c         position du max de cambrure (ligne de cambrure)
            LE_radius   rayon de bord d'attaque
            TE_angle    angle de bord de fuite
            t_over_xt   rapport t / x_t
            area        aire de la section transversale
        """
        af = asb.Airfoil(name)

        # 1. Épaisseur relative maximale
        t = float(af.max_thickness())

        # 2. Cambrure maximale
        camber = float(af.max_camber())

        # 3. Position du maximum d'épaisseur sur l'extrados
        coords  = af.coordinates
        n       = len(coords) // 2
        x_upper = coords[:n, 0]
        y_upper = coords[:n, 1]
        x_t     = float(x_upper[np.argmax(y_upper)])

        # 4. Position du maximum de cambrure
        # Ligne de cambrure = moyenne extrados / intrados
        x_lower  = coords[n:, 0]
        y_lower  = coords[n:, 1]
        y_camber = (y_upper + np.interp(x_upper, x_lower[::-1], y_lower[::-1])) / 2
        x_c      = float(x_upper[np.argmax(np.abs(y_camber))])

        # 5. Rayon de bord d'attaque
        LE_radius = float(af.LE_radius())

        # 6. Angle de bord de fuite — calcul direct sur les coordonnées
        # af.TE_angle() retourne des valeurs aberrantes (>800°)
        # Angle entre vecteurs tangents extrados/intrados au bord de fuite
        v_upper  = coords[0]  - coords[1]
        v_lower  = coords[-1] - coords[-2]
        cos_a    = np.dot(v_upper, v_lower) / (
                   np.linalg.norm(v_upper) * np.linalg.norm(v_lower) + 1e-12)
        TE_angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

        # 7. Rapport épaisseur / position
        t_over_xt = t / max(x_t, 0.01)

        # 8. Aire de la section transversale
        area = float(af.area())

        return {
            "t"         : round(t,          6),
            "camber"    : round(camber,     6),
            "x_t"       : round(x_t,        4),
            "x_c"       : round(x_c,        4),
            "LE_radius" : round(LE_radius,  6),
            "TE_angle"  : round(TE_angle,   6),
            "t_over_xt" : round(t_over_xt,  6),
            "area"      : round(area,       6),
        }

    # ══════════════════════════════════════════════════════════════
    # 3. GÉNÉRATION DES LIGNES POUR UN PROFIL
    # ══════════════════════════════════════════════════════════════

    def _generate_combinations(self, meta: dict) -> pd.DataFrame:
        """
        Génère toutes les combinaisons (alpha × Re) pour un profil,
        avec ses 8 features géométriques.

        Retourne un DataFrame de 300 lignes (60 α × 5 Re) ou None.
        """
        name   = meta["name"]
        source = meta["source"]

        try:
            # Extraire les 8 features géométriques une seule fois par profil
            geom = self._get_geometry(name)

            rows = []

            for re in self.RE_RANGE:
                for alpha in self.ALPHA_RANGE:
                    rows.append({
                        # Identifiant
                        "naca"      : name,
                        "source"    : source,
                        # 8 features géométriques
                        "t"         : geom["t"],
                        "camber"    : geom["camber"],
                        "x_t"       : geom["x_t"],
                        "x_c"       : geom["x_c"],
                        "LE_radius" : geom["LE_radius"],
                        "TE_angle"  : geom["TE_angle"],
                        "t_over_xt" : geom["t_over_xt"],
                        "area"      : geom["area"],
                        # Conditions de vol
                        "alpha"     : float(alpha),
                        "Re"        : float(re),
                    })

            return pd.DataFrame(rows) if rows else None

        except Exception as exc:
            print(f"    ✗  {name} : {exc}")
            return None

    # ══════════════════════════════════════════════════════════════
    # 4. CONSTRUCTION COMPLÈTE
    # ══════════════════════════════════════════════════════════════

    def build(self) -> pd.DataFrame:
        """
        Construit le dataset complet et sauvegarde progressivement.
        Reprise automatique si le CSV existe déjà.
        """
        print("=" * 60)
        print("  DatasetBuilder — AeroPredict")
        print("=" * 60)

        all_profiles = self._all_profiles()

        # ── Reprise après interruption ──────────────────────────
        deja_faits  = set()
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

        # ── Boucle principale ────────────────────────────────────
        n_ok  = len(deja_faits)
        n_err = 0
        t0    = time.time()

        for i, meta in enumerate(restants, 1):
            df_profil = self._generate_combinations(meta)

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
                    f"{meta['name']:22s}  "
                    f"[{meta['source']:9s}]  "
                    f"{len(df_profil):4d} lignes  "
                    f"~{reste/60:.1f} min restantes"
                )
            else:
                n_err += 1

        # ── Résumé final ─────────────────────────────────────────
        df_final = pd.read_csv(self.output_path)
        self._summary(df_final, n_ok, n_err, time.time() - t0)
        return df_final

    # ══════════════════════════════════════════════════════════════
    # 5. RÉSUMÉ
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _summary(df: pd.DataFrame, n_ok: int, n_err: int, elapsed: float):
        print(f"\n{'=' * 60}")
        print(f"  Dataset construit")
        print(f"{'=' * 60}")
        print(f"  Profils générés  : {n_ok}")
        print(f"  Profils en erreur: {n_err}")
        print(f"  Lignes totales   : {len(df):,}")
        print(f"  Colonnes ({len(df.columns):2d})    : {list(df.columns)}")
        print(f"  Temps total      : {elapsed/60:.1f} min")
        print(f"\n  Par source :")
        print(df.groupby("source").agg(
            profils = ("naca", "nunique"),
            lignes  = ("naca", "count"),
        ).to_string())
        print(f"\n  Plages des 8 features géométriques :")
        print(df[["t", "camber", "x_t", "x_c",
                  "LE_radius", "TE_angle", "t_over_xt", "area"]].describe().round(4).to_string())


# ══════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Construit le dataset AeroPredict")
    parser.add_argument("--test",   action="store_true", help="Mode test : 10 profils")
    parser.add_argument("--output", default="dataset.csv", help="Fichier de sortie")
    args = parser.parse_args()

    builder = DatasetBuilder(
        output_path = "dataset_test.csv" if args.test else args.output,
        max_profils = 10 if args.test else None,
    )

    df = builder.build()
