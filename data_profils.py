"""DatasetBuilder — Construction complète du dataset AeroPredict.



Un seul script qui :
  1. Génère la grille paramétrique NACA 4 chiffres
  2. Charge les profils UIUC embarqués dans AeroSandbox
  3. Extrait 8 features géométriques via AeroSandbox pour chaque profil
  4. Génère toutes les combinaisons (alpha × Re)
  5. Sauvegarde progressivement dans un seul dataset.csv

Les colonnes CL, CD, CM seront ajoutées ultérieurement via XFoil.

Usage:
    python dataset_builder.py           # dataset complet
    python dataset_builder.py --test    # 10 profils, test rapide
"""

import argparse
import pathlib
import time
from typing import Dict, List, Optional, Set
from pandas.errors import EmptyDataError

import aerosandbox as asb
import numpy as np
import pandas as pd


class DatasetBuilder:
    """Construit le dataset géométrique complet (NACA + UIUC) pour AeroPredict.

    Attributes:
        ALPHA_RANGE (np.ndarray): Plage des angles d'attaque (60 valeurs).
        RE_RANGE (List[float]): Liste des nombres de Reynolds (5 valeurs).
        M_RANGE (List[int]): Chiffres de cambrure max (1er chiffre NACA / 100).
        P_RANGE (List[int]): Chiffres de position cambrure (2e chiffre NACA / 10).
        T_RANGE (List[int]): Chiffres d'épaisseur relative (3e+4e chiffres / 100).
        UIUC_FAMILIES (List[str]): Familles de profils UIUC retenues.
        output_path (str): Chemin du fichier CSV de sortie.
        max_profils (Optional[int]): Nombre maximum de profils à traiter (mode test).
    """

    #: Plages de conditions de vol (60 valeurs de -6° à 23.5° par pas de 0.5°)
    ALPHA_RANGE: np.ndarray = np.arange(-6, 24, 0.5, dtype=float)
    
    #: Liste des nombres de Reynolds retenus
    RE_RANGE: List[float] = [5e4, 1e5, 5e5, 1e6, 5e6]

    #: Grille NACA : Cambrure maximale
    M_RANGE: List[int] = [0, 1, 2, 3, 4, 5, 6]
    
    #: Grille NACA : Position de la cambrure maximale
    P_RANGE: List[int] = [1, 2, 3, 4, 5, 6]
    
    #: Grille NACA : Épaisseur relative
    T_RANGE: List[int] = [6, 8, 10, 12, 15, 18, 21, 24]

    #: Préfixes des familles de profils UIUC sélectionnés
    UIUC_FAMILIES: List[str] = ["e", "fx", "s", "ag", "mh", "hq", "sd", "clarky", "goe"]

    def __init__(self, output_path: str = "dataset_profil.csv", max_profils: Optional[int] = None) -> None:
        """Initialise le constructeur de dataset.

        Args:
            output_path: Chemin du fichier CSV de sortie. En fonction du mode,
                il peut être écrasé ou complété.
            max_profils: Limite le nombre de profils traités si renseigné.
        """
        self.output_path: str = output_path
        self.max_profils: Optional[int] = max_profils

    def _naca_profiles(self) -> List[Dict[str, str]]:
        """Génère la liste combinatoire des profils NACA à 4 chiffres.

        Les profils m=0 étant symétriques, la position de la cambrure (p) n'a pas
        d'impact géométrique. On ne garde donc qu'une seule valeur de p pour m=0.

        Returns:
            Une liste de dictionnaires contenant le nom du profil et sa source.
            Exemple: [{'name': 'naca0106', 'source': 'naca_grid'}, ...]
        """
        profiles = []

        for m in self.M_RANGE:
            for p in self.P_RANGE:
                if m == 0 and p != self.P_RANGE[0]:
                    continue

                for t in self.T_RANGE:
                    profiles.append({
                        "name": f"naca{m}{p}{t:02d}",
                        "source": "naca_grid",
                    })

        return profiles

    def _uiuc_profiles(self) -> List[Dict[str, str]]:
        """Recherche et filtre les profils UIUC disponibles dans AeroSandbox.

        Exclut les profils dont le nom commence par 'naca' pour éviter les doublons.

        Returns:
            Une liste de dictionnaires contenant les métadonnées des profils UIUC.
        """
        pkg_path = pathlib.Path(asb.__file__).parent
        dat_files = pkg_path.rglob("*.dat")
        all_names = [f.stem for f in dat_files]

        selected = []
        for name in sorted(set(all_names)):
            if name.startswith("naca"):
                continue
            for famille in self.UIUC_FAMILIES:
                if name.startswith(famille):
                    selected.append({
                        "name": name,
                        "source": "uiuc",
                    })
                    break

        return selected

    def _all_profiles(self) -> List[Dict[str, str]]:
        """Fusionne les listes de profils NACA et UIUC et applique les filtres.

        Affiche un résumé statistique global des volumes de données à générer dans la console.

        Returns:
            La liste combinée et potentiellement tronquée des profils.
        """
        naca = self._naca_profiles()
        uiuc = self._uiuc_profiles()
        all_p = naca + uiuc

        if self.max_profils:
            all_p = all_p[: self.max_profils]

        n_alpha = len(self.ALPHA_RANGE)
        n_re = len(self.RE_RANGE)
        lignes = len(all_p) * n_alpha * n_re

        print(f"\n  Profils NACA   : {len(naca)}")
        print(f"  Profils UIUC   : {len(uiuc)}")
        print(f"  Total profils  : {len(all_p)}" + (f"  (limité à {self.max_profils})" if self.max_profils else ""))
        print(f"  Alpha          : {n_alpha} valeurs  ({self.ALPHA_RANGE[0]}° → {self.ALPHA_RANGE[-1]}°)")
        print(f"  Re             : {n_re} valeurs  {self.RE_RANGE}")
        print(f"  Lignes totales : ~{lignes:,}  ({n_alpha} α × {n_re} Re × {len(all_p)} profils)")

        return all_p

    def _get_geometry(self, name: str) -> Dict[str, float]:
        """Extrait les 8 features géométriques d'un profil via AeroSandbox.

        Cette méthode est purement géométrique et n'exécute aucune simulation aérodynamique.

        Args:
            name: Nom du profil aérodynamique reconnu par AeroSandbox.

        Returns:
            Un dictionnaire contenant les caractéristiques géométriques calculées :
                - t : Épaisseur relative maximale.
                - camber : Cambrure maximale.
                - x_t : Position du maximum d'épaisseur sur l'extrados.
                - x_c : Position du maximum de cambrure.
                - LE_radius : Rayon du bord d'attaque.
                - TE_angle : Angle géométrique du bord de fuite (en degrés).
                - t_over_xt : Rapport t / x_t.
                - area : Aire de la section transversale du profil.
        """
        af = asb.Airfoil(name)

        # 1. Épaisseur relative maximale
        t = float(af.max_thickness())

        # 2. Cambrure maximale
        camber = float(af.max_camber())

        # 3. Position du maximum d'épaisseur sur l'extrados
        coords = af.coordinates
        n = len(coords) // 2
        x_upper = coords[:n, 0]
        y_upper = coords[:n, 1]
        x_t = float(x_upper[np.argmax(y_upper)])

        # 4. Position du maximum de cambrure
        x_lower = coords[n:, 0]
        y_lower = coords[n:, 1]
        y_camber = (y_upper + np.interp(x_upper, x_lower[::-1], y_lower[::-1])) / 2
        x_c = float(x_upper[np.argmax(np.abs(y_camber))])

        # 5. Rayon de bord d'attaque
        LE_radius = float(af.LE_radius())

        # 6. Angle de bord de fuite
        v_upper = coords[0] - coords[1]
        v_lower = coords[-1] - coords[-2]
        cos_a = np.dot(v_upper, v_lower) / (np.linalg.norm(v_upper) * np.linalg.norm(v_lower) + 1e-12)
        TE_angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

        # 7. Rapport épaisseur / position
        t_over_xt = t / max(x_t, 0.01)

        # 8. Aire de la section transversale
        area = float(af.area())

        return {
            "t": round(t, 6),
            "camber": round(camber, 6),
            "x_t": round(x_t, 4),
            "x_c": round(x_c, 4),
            "LE_radius": round(LE_radius, 6),
            "TE_angle": round(TE_angle, 6),
            "t_over_xt": round(t_over_xt, 6),
            "area": round(area, 6),
        }

    def _generate_combinations(self, meta: Dict[str, str]) -> Optional[pd.DataFrame]:
        """Génère toutes les combinaisons de conditions de vol pour un profil donné.

        Associe la géométrie extraite à la matrice croisée (ALPHA_RANGE × RE_RANGE).

        Args:
            meta: Dictionnaire contenant les clés 'name' et 'source' du profil.

        Returns:
            Un DataFrame pandas de 300 lignes contenant toutes les variantes,
            ou None si une erreur survient lors de l'extraction géométrique.
        """
        name = meta["name"]
        source = meta["source"]

        try:
            geom = self._get_geometry(name)
            rows = []

            for re in self.RE_RANGE:
                for alpha in self.ALPHA_RANGE:
                    rows.append({
                        "airfoil": name,
                        "source": source,
                        "t": geom["t"],
                        "camber": geom["camber"],
                        "x_t": geom["x_t"],
                        "x_c": geom["x_c"],
                        "LE_radius": geom["LE_radius"],
                        "TE_angle": geom["TE_angle"],
                        "t_over_xt": geom["t_over_xt"],
                        "area": geom["area"],
                        "alpha": float(alpha),
                        "Re": float(re),
                    })

            return pd.DataFrame(rows) if rows else None

        except Exception as exc:
            print(f"    ✗  {name} : {exc}")
            return None

    def build(self) -> pd.DataFrame:
        """Déclenche la construction complète du dataset et gère la sauvegarde.

        Vérifie si un fichier existe déjà au chemin `output_path` pour reprendre
        le travail là où il s'est arrêté (tolérance aux pannes).

        Returns:
            Le DataFrame final contenant l'ensemble des données consolidées.
        """
        print("=" * 60)
        print("  DatasetBuilder — AeroPredict")
        print("=" * 60)

        all_profiles = self._all_profiles()

        deja_faits: Set[str] = set()
        first_write = True
        try:
            df_exist = pd.read_csv(self.output_path)
            deja_faits = set(df_exist["airfoil"].unique())
            first_write = False
            print(f"\n  Reprise : {len(deja_faits)} profils déjà générés")
        except (FileNotFoundError, EmptyDataError):
            print(f"\n  Nouveau fichier : {self.output_path}")

        restants = [p for p in all_profiles if p["name"] not in deja_faits]
        print(f"  Restants : {len(restants)} profils\n")

        n_ok = len(deja_faits)
        n_err = 0
        t0 = time.time()

        for i, meta in enumerate(restants, 1):
            df_profil = self._generate_combinations(meta)

            if df_profil is not None and not df_profil.empty:
                df_profil.to_csv(
                    self.output_path,
                    mode="a",
                    header=first_write,
                    index=False,
                    encoding="utf-8"
                )
                first_write = False
                n_ok += 1

                elapsed = time.time() - t0
                reste = (elapsed / i) * (len(restants) - i)
                print(
                    f"  [{i:4d}/{len(restants)}]  "
                    f"{meta['name']:22s}  "
                    f"[{meta['source']:9s}]  "
                    f"{len(df_profil):4d} lignes  "
                    f"~{reste/60:.1f} min restantes"
                )
            else:
                n_err += 1

        df_final = pd.read_csv(self.output_path)
        self._summary(df_final, n_ok, n_err, time.time() - t0)
        return df_final

    @staticmethod
    def _summary(df: pd.DataFrame, n_ok: int, n_err: int, elapsed: float) -> None:
        """Génère et affiche un rapport d'exécution détaillé dans la console.

        Args:
            df: Le DataFrame final chargé depuis le disque.
            n_ok: Nombre de profils traités avec succès.
            n_err: Nombre de profils ayant levé une exception.
            elapsed: Temps total écoulé en secondes.
        """
        print(f"\n{'=' * 60}")
        print("  Dataset construit")
        print(f"{'=' * 60}")
        print(f"  Profils générés  : {n_ok}")
        print(f"  Profils en erreur: {n_err}")
        print(f"  Lignes totales   : {len(df):,}")
        print(f"  Colonnes ({len(df.columns):2d})     : {list(df.columns)}")
        print(f"  Temps total      : {elapsed/60:.1f} min")
        print("\n  Par source :")
        print(df.groupby("source").agg(
            profils=("airfoil", "nunique"),
            lignes=("airfoil", "count"),
        ).to_string())
        print("\n  Plages des 8 features géométriques :")
        print(df[["t", "camber", "x_t", "x_c",
                  "LE_radius", "TE_angle", "t_over_xt", "area"]].describe().round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construit le dataset AeroPredict")
    parser.add_argument("--test", action="store_true", help="Mode test : limite à 10 profils")
    parser.add_argument("--output", default="dataset_profil.csv", help="Fichier de sortie CSV")
    args = parser.parse_args()

    builder = DatasetBuilder(
        output_path="dataset_test.csv" if args.test else args.output,
        max_profils=10 if args.test else None,
    )

    df_dataset = builder.build()
