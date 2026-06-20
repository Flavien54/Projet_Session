"""Dashboard interactif AeroPredict — MGA 802.

Ce module construit un tableau de bord `Streamlit <https://streamlit.io/>`_
permettant d'explorer et de valider les résultats du pipeline aérodynamique
couplé à un modèle de Machine Learning. Il propose les six sections
suivantes, accessibles depuis la barre latérale de navigation :

#. **Polaires** — comparaison des polaires par profil (XFoil vs modèle ML
   vs Ansys Fluent) ;
#. **Performance ML** — métriques globales du modèle ML (R², MAE, RMSE,
   dispersion) ;
#. **Validation ML** — analyse fine du comportement du modèle (dérivée
   ``dCl/dalpha``, régression polynomiale ``Cd(Cl)``) ;
#. **Optimisation** — recherche du profil maximisant la finesse
   ``CL/CD`` ;
#. **Dataset** — exploration statistique du corpus (familles de profils,
   taux de convergence, distribution des features géométriques) ;
#. **Prédiction ML** — génération à la volée des polaires d'un profil
   quelconque défini par l'utilisateur.

Ce fichier est documenté au format reStructuredText (compatible
`Sphinx <https://www.sphinx-doc.org/>`_ et son extension ``autodoc``). Pour
générer la documentation HTML, il suffit d'inclure ce module dans la
configuration Sphinx (``conf.py``) puis d'exécuter ``sphinx-build``.

:Usage:

    .. code-block:: bash

        streamlit run dashboard.py

:Auteurs: Blanchard / Mechref / Condette
:Cours: MGA 802
"""

import glob
import os
import re as regex

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats, optimize
import base64 as _b64


def _logo_base64(path: str = "logos_ets.png") -> str:
    """Encode le logo ETS en base64 pour une injection HTML inline.

    Cette fonction est utilisée pour afficher le logo directement dans
    une balise ``<img>`` via ``data:image/png;base64,...`` sans dépendre
    d'un chemin statique servi par Streamlit.

    :param path: Chemin relatif ou absolu vers le fichier image du logo.
    :type path: str
    :returns: Représentation base64 du contenu binaire de l'image, ou une
        chaîne vide si le fichier n'existe pas (le logo est alors
        simplement masqué côté HTML via ``onerror``).
    :rtype: str
    """
    try:
        with open(path, "rb") as f:
            return _b64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

# ── Configuration générale ───────────────────────────────────────

st.set_page_config(
    page_title="AeroPredict — Dashboard Scientifique",
    page_icon="logos_ets.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

#: Chemin du fichier CSV contenant les résultats bruts XFoil.
CSV_XFOIL = "dataset_aeroXfoil.csv"

#: Chemin du fichier CSV contenant les résultats XFoil enrichis des
#: prédictions du modèle ML (mêmes clés ``(naca, alpha, Re)`` que
#: :data:`CSV_XFOIL`).
CSV_ML    = "dataset_aeroXfoil_avec_predictions.csv"

#: Chemin du modèle de réseau de neurones multi-tâches sauvegardé au
#: format Keras (``.keras``).
MODEL_PATH       = "naca_multitask_model.keras"

#: Chemin du préprocesseur (``StandardScaler`` sur les features et sur
#: chaque sortie) sérialisé via ``pickle``.
PREPROCESSOR_PATH = "preprocessor.pkl"

#: Palette de couleurs scientifique moderne utilisée pour l'ensemble des
#: graphiques Plotly et des composants HTML du dashboard. Les clés
#: ``xfoil``, ``ml`` et ``fluent`` désignent les couleurs associées à
#: chaque source de données afin d'assurer une cohérence visuelle sur
#: toutes les pages.
COLORS = {
    "primary": "#0F172A",
    "secondary": "#1E293B",
    "accent": "#3B82F6",
    "accent_light": "#60A5FA",
    "xfoil": "#2563EB",
    "ml": "#DC2626",
    "fluent": "#059669",
    "background": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "text": "#1E293B",
    "text_light": "#64748B",
    "success": "#10B981",
    "warning": "#F59E0B",
    "info": "#3B82F6",
    "sidebar_bg": "#0F172A",
    "sidebar_text": "#E2E8F0",
}

#: Ordre exact des 8 features géométriques attendues en entrée du
#: ``StandardScaler`` (et donc du modèle ML). Cet ordre doit impérativement
#: correspondre à celui utilisé lors de l'entraînement, sous peine de
#: prédictions erronées silencieuses.
FEATURES_ORDRE = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area"]


# ── Helpers ──────────────────────────────────────────────────────

def re_label(r: float) -> str:
    """Formate un nombre de Reynolds en notation scientifique lisible.

    Le résultat est destiné aux libellés de widgets Streamlit (menus,
    sliders) ; il utilise des exposants Unicode pour un rendu compact.

    :param r: Nombre de Reynolds à formater.
    :type r: float
    :returns: Chaîne du type ``"Re = 1×10⁶"`` ou ``"Re = 2.5×10⁵"``. Si le
        coefficient est proche de 1 (à 5 % près), il est omis pour
        simplifier l'affichage (``"Re = 10⁶"``).
    :rtype: str
    """
    exp      = int(np.floor(np.log10(r)))
    coef     = r / 10**exp
    exposant = str(exp).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    if abs(coef - 1.0) < 0.05:
        return f"Re = 10{exposant}"
    return f"Re = {coef:.1f}×10{exposant}"


def re_tick(r: float) -> str:
    """Formate un nombre de Reynolds pour les étiquettes d'axe Plotly.

    Variante de :func:`re_label` produisant un texte pur (sans le préfixe
    ``"Re = "``), adapté aux étiquettes d'axes catégoriels des graphiques
    en barres.

    :param r: Nombre de Reynolds à formater.
    :type r: float
    :returns: Chaîne du type ``"1.0×10⁶"``.
    :rtype: str
    """
    exp      = int(np.floor(np.log10(r)))
    coef     = r / 10**exp
    exposant = str(exp).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{coef:.1f}×10{exposant}"


# ── Chargement des données (mis en cache) ────────────────────────

@st.cache_data(show_spinner="Chargement des datasets...")
def charger_donnees():
    """Charge les datasets XFoil et ML, et les fusionne sur leurs clés communes.

    La fusion (``merge`` interne) s'effectue sur le triplet
    ``(naca, alpha, Re)`` qui identifie de façon unique un point de calcul
    aérodynamique. Les colonnes de coefficients (``CL``, ``CD``, ``CM``)
    issues de chaque source sont renommées avec les suffixes
    ``_xfoil`` / ``_ml`` afin d'éviter toute ambiguïté en aval.

    .. note::
        Cette fonction est décorée par ``st.cache_data`` : Streamlit met en
        cache le résultat tant que les fichiers source ne changent pas,
        évitant ainsi une relecture disque à chaque interaction utilisateur.

    :returns: Un triplet ``(df_xfoil, df_ml, df_merge)`` où :

        * ``df_xfoil`` est le DataFrame brut issu de :data:`CSV_XFOIL` ;
        * ``df_ml`` est le DataFrame brut issu de :data:`CSV_ML` ;
        * ``df_merge`` est la fusion des deux, avec colonnes suffixées
          ``_xfoil`` et ``_ml`` pour ``CL``, ``CD``, ``CM``.
    :rtype: tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
    """
    df_xfoil = pd.read_csv(CSV_XFOIL)
    df_ml    = pd.read_csv(CSV_ML)

    cles     = ["naca", "alpha", "Re"]
    df_merge = df_xfoil[cles + ["source", "CL", "CD", "CM", "converged"]].merge(
        df_ml[cles + ["CL", "CD", "CM"]],
        on=cles,
        suffixes=("_xfoil", "_ml"),
    )
    return df_xfoil, df_ml, df_merge


@st.cache_data
def charger_fluent():
    """Charge les données de référence Ansys Fluent, toutes sources confondues.

    Deux sources de données sont combinées :

    #. un fichier unique multi-profils (``dataset_fluent.csv``), dont les
       colonnes peuvent porter des noms hétérogènes selon la version du
       pipeline ayant généré le fichier (``CL_ML``/``CFD_CL``, etc.) — ces
       colonnes sont renommées de façon défensive vers le format canonique
       ``CL``/``CD``/``CM``/``LD`` ;
    #. des fichiers individuels par profil et par Reynolds, nommés selon
       le motif ``"{profil}_Re{reynolds}_FLUENT.csv"``, utilisés en
       complément si le profil n'est pas déjà présent dans le fichier
       groupé.

    :returns: Dictionnaire associant chaque couple ``(profil, Re)`` (clé)
        à un ``pandas.DataFrame`` contenant les colonnes aérodynamiques
        Fluent correspondantes (valeur).
    :rtype: dict[tuple(str, float), pandas.DataFrame]
    """
    donnees = {}

    # Source 1 : fichier groupé multi-profils recalculé par ML
    fichier_bruite = "dataset_fluent.csv"
    if os.path.exists(fichier_bruite):
        df_flu = pd.read_csv(fichier_bruite)

        if "alpha" in df_flu.columns:
            df_flu = df_flu.rename(columns={"alpha": "alpha_deg"})

        # Renommage défensif : certaines variantes du pipeline exportent
        # les coefficients ML sous les noms CL_ML / CD_ML / CM_ML / LD_ML.
        rename_ml = {}
        for src, dst in [("CL_ML", "CL"), ("CD_ML", "CD"), ("CM_ML", "CM"), ("LD_ML", "LD")]:
            if src in df_flu.columns and dst not in df_flu.columns:
                rename_ml[src] = dst
        if rename_ml:
            df_flu = df_flu.rename(columns=rename_ml)

        # Renommage défensif (variante 2) : export brut CFD avec un
        # mapping de colonnes décalé (CFD_CL -> CL, CFD_CM -> CD, CFD_CD -> CM).
        rename_cfd = {}
        if "CFD_CL" in df_flu.columns and "CL" not in df_flu.columns:
            rename_cfd["CFD_CL"] = "CL"
        if "CFD_CM" in df_flu.columns and "CD" not in df_flu.columns:
            rename_cfd["CFD_CM"] = "CD"
        if "CFD_CD" in df_flu.columns and "CM" not in df_flu.columns:
            rename_cfd["CFD_CD"] = "CM"
        if rename_cfd:
            df_flu = df_flu.rename(columns=rename_cfd)

        for (profil, re_val), sous_df in df_flu.groupby(["naca", "Re"]):
            donnees[(profil, float(re_val))] = sous_df.copy()

    # Source 2 : fichiers individuels "{profil}_Re{reynolds}_FLUENT.csv"
    for chemin in glob.glob("*_FLUENT.csv"):
        m = regex.match(r"(.+)_Re(\d+)_FLUENT\.csv", os.path.basename(chemin))
        if m:
            profil, re_val = m.group(1), float(m.group(2))
            if (profil, re_val) not in donnees:
                donnees[(profil, re_val)] = pd.read_csv(chemin)

    return donnees


@st.cache_resource(show_spinner="Chargement du modèle ML…")
def charger_modele():
    """Charge le modèle Keras de prédiction aérodynamique et son préprocesseur.

    Le chargement du modèle est tenté successivement via ``tensorflow.keras``
    puis, en cas d'échec, via le paquet ``keras`` autonome — ceci afin de
    rester compatible avec différents environnements d'exécution.

    Le préprocesseur sérialisé (``pickle``) référence une classe
    :class:`NACAAeroPreprocessor` définie au moment de l'entraînement.
    Comme cette classe n'est pas importable depuis ce module au moment du
    ``unpickle``, elle est reconstruite ici à l'identique et injectée dans
    l'espace de noms ``__main__`` avant désérialisation.

    .. note::
        Cette fonction est décorée par ``st.cache_resource`` : le modèle
        et le préprocesseur ne sont chargés qu'une seule fois par session
        Streamlit, ce qui évite un rechargement coûteux à chaque
        interaction.

    :returns: Un couple ``(model, preprocessor)``. Si l'un des fichiers
        requis (:data:`MODEL_PATH`, :data:`PREPROCESSOR_PATH`) est absent,
        ou si le chargement échoue, retourne ``(None, None)``.
    :rtype: tuple
    """
    import pickle

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        return None, None

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        try:
            import keras
            model = keras.models.load_model(MODEL_PATH)
        except Exception as exc:
            st.error(f"Impossible de charger le modèle : {exc}")
            return None, None

    from sklearn.preprocessing import StandardScaler as _SS
    import numpy as _np

    class NACAAeroPreprocessor:
        """Reconstruction locale du préprocesseur utilisé à l'entraînement.

        Cette classe doit être strictement identique (en termes
        d'attributs et de structure) à celle utilisée lors de la
        sérialisation du fichier :data:`PREPROCESSOR_PATH`, afin que
        ``pickle.load`` puisse restaurer correctement l'état des scalers
        (moyennes, variances apprises).

        :cvar GEOMETRIC_COLS: Les 8 colonnes géométriques d'entrée.
        :vartype GEOMETRIC_COLS: list[str]
        :cvar AERO_COLS: Les colonnes de conditions de vol (angle
            d'attaque et nombre de Reynolds).
        :vartype AERO_COLS: list[str]
        :cvar TARGET_COLS: Les colonnes cibles prédites par le modèle.
        :vartype TARGET_COLS: list[str]
        """

        GEOMETRIC_COLS = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area"]
        AERO_COLS      = ["alpha", "Re"]
        TARGET_COLS    = ["CL", "CD", "CM"]

        def __init__(self):
            """Initialise les scalers (non encore ajustés à ce stade).

            Les scalers réels (avec leurs paramètres appris) sont
            restaurés juste après par ``pickle.load``, qui réassigne
            l'état interne (``__dict__``) de cette instance.
            """
            self.feature_scaler = _SS()
            self.target_scalers = {col: _SS() for col in self.TARGET_COLS}
            self.is_fitted      = False

        @property
        def input_cols(self):
            """Liste ordonnée de toutes les colonnes d'entrée du modèle.

            :returns: Concatenation de :attr:`GEOMETRIC_COLS` et
                :attr:`AERO_COLS`, dans cet ordre exact.
            :rtype: list[str]
            """
            return self.GEOMETRIC_COLS + self.AERO_COLS

        def transform(self, df):
            """Standardise un DataFrame de features brutes.

            :param df: DataFrame contenant au moins les colonnes
                listées par :attr:`input_cols`.
            :type df: pandas.DataFrame
            :returns: Tableau ``float32`` standardisé, prêt à être
                fourni en entrée du réseau de neurones.
            :rtype: numpy.ndarray
            """
            X_raw = df[self.input_cols].values.astype(_np.float32)
            return self.feature_scaler.transform(X_raw).astype(_np.float32)

        def inverse_transform_target(self, y_scaled, col):
            """Dé-standardise une sortie prédite pour la ramener à son échelle physique.

            :param y_scaled: Valeurs standardisées prédites par le modèle.
            :type y_scaled: numpy.ndarray
            :param col: Nom de la cible concernée (``"CL"``, ``"CD"`` ou
                ``"CM"``).
            :type col: str
            :returns: Valeurs dans l'échelle physique d'origine.
            :rtype: numpy.ndarray
            """
            return self.target_scalers[col].inverse_transform(y_scaled)

        @property
        def input_dim(self):
            """Dimension du vecteur d'entrée du modèle.

            :returns: Nombre total de features d'entrée (géométriques +
                aérodynamiques).
            :rtype: int
            """
            return len(self.input_cols)

    # Injection de la classe dans __main__ pour permettre à pickle de
    # résoudre la référence enregistrée lors de la sérialisation.
    import __main__
    __main__.NACAAeroPreprocessor = NACAAeroPreprocessor

    with open(PREPROCESSOR_PATH, "rb") as f:
        pre = pickle.load(f)

    return model, pre


def _decoder_preds(preds_raw, ts: dict) -> dict:
    """Dé-normalise les sorties brutes (standardisées) du modèle ML.

    Le modèle Keras peut retourner ses trois têtes de sortie soit sous la
    forme d'une liste de tableaux (un par tâche), soit sous la forme d'un
    unique tableau 2D dont les colonnes correspondent respectivement à
    ``CL``, ``CD`` et ``CM`` ; les deux cas sont gérés ici.

    :param preds_raw: Sortie brute (standardisée) de ``model.predict``.
    :type preds_raw: list[numpy.ndarray] or numpy.ndarray
    :param ts: Dictionnaire des ``StandardScaler`` cibles, indexé par nom
        de coefficient (``"CL"``, ``"CD"``, ``"CM"``), tel qu'attendu par
        :meth:`NACAAeroPreprocessor.inverse_transform_target`.
    :type ts: dict
    :returns: Dictionnaire ``{"CL": array, "CD": array, "CM": array}``
        contenant les coefficients dans leur échelle physique d'origine.
    :rtype: dict[str, numpy.ndarray]
    """
    coefs = ["CL", "CD", "CM"]
    results = {}
    if isinstance(preds_raw, list):
        for i, coef in enumerate(coefs):
            col = preds_raw[i].flatten()
            results[coef] = ts[coef].inverse_transform(col.reshape(-1, 1)).flatten()
    else:
        for i, coef in enumerate(coefs):
            col = preds_raw[:, i]
            results[coef] = ts[coef].inverse_transform(col.reshape(-1, 1)).flatten()
    return results


def predire_polaires(
    model,
    pre,
    geo: dict,
    alphas: np.ndarray,
    re_val: float,
    naca_name: str = "",
    source_name: str = "naca_grid",
) -> pd.DataFrame:
    """Prédit la polaire complète (CL, CD, CM, finesse) d'un profil donné.

    Pour chaque angle d'attaque fourni, un vecteur d'entrée est construit
    en concaténant les 8 features géométriques (dans l'ordre imposé par
    :data:`FEATURES_ORDRE`) avec l'angle d'attaque courant et le nombre de
    Reynolds, puis standardisé et soumis au modèle.

    :param model: Modèle Keras chargé via :func:`charger_modele`.
    :param pre: Préprocesseur associé au modèle (instance de
        ``NACAAeroPreprocessor``), porteur des scalers ``feature_scaler``
        et ``target_scalers``.
    :param geo: Dictionnaire des 8 paramètres géométriques du profil ;
        doit contenir au minimum toutes les clés de
        :data:`FEATURES_ORDRE`.
    :type geo: dict
    :param alphas: Vecteur des angles d'attaque (en degrés) pour lesquels
        générer la prédiction.
    :type alphas: numpy.ndarray
    :param re_val: Nombre de Reynolds des conditions de vol.
    :type re_val: float
    :param naca_name: Nom du profil (purement informatif, non utilisé
        dans le calcul).
    :type naca_name: str
    :param source_name: Étiquette de la source de génération (purement
        informative, non utilisée dans le calcul).
    :type source_name: str
    :returns: DataFrame indexé par ``alpha``, contenant les colonnes
        ``CL``, ``CD``, ``CM`` et ``finesse`` (= ``CL / CD``, ``NaN``
        lorsque ``CD`` est nul).
    :rtype: pandas.DataFrame
    """
    fs = pre.__dict__["feature_scaler"]
    ts = pre.__dict__["target_scalers"]

    rows = [[geo[f] for f in FEATURES_ORDRE] + [alpha, re_val] for alpha in alphas]
    X_raw    = np.array(rows, dtype=np.float32)
    X_scaled = fs.transform(X_raw).astype(np.float32)

    preds_raw = model.predict(X_scaled, verbose=0)
    results   = _decoder_preds(preds_raw, ts)

    df = pd.DataFrame({"alpha": alphas, **results})
    df["finesse"] = df["CL"] / df["CD"].replace(0, np.nan)
    return df


def calculer_metriques(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule les métriques de qualité de régression R², MAE et RMSE.

    :param y_vrai: Valeurs de référence (vérité terrain), typiquement
        issues d'XFoil.
    :type y_vrai: numpy.ndarray
    :param y_pred: Valeurs prédites par le modèle ML, de même forme que
        ``y_vrai``.
    :type y_pred: numpy.ndarray
    :returns: Dictionnaire à trois clés :

        * ``"R2"`` — coefficient de détermination (``NaN`` si la variance
          de ``y_vrai`` est nulle) ;
        * ``"MAE"`` — erreur absolue moyenne ;
        * ``"RMSE"`` — racine de l'erreur quadratique moyenne.
    :rtype: dict[str, float]
    """
    residus = y_vrai - y_pred
    ss_res  = float(np.sum(residus ** 2))
    ss_tot  = float(np.sum((y_vrai - y_vrai.mean()) ** 2))
    return {
        "R2"  : 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "MAE" : float(np.mean(np.abs(residus))),
        "RMSE": float(np.sqrt(np.mean(residus ** 2))),
    }


# ── En-tête et Sidebar ───────────────────────────────────────────

def render_enhanced_header(df_xfoil: pd.DataFrame = None, df_ml: pd.DataFrame = None) -> None:
    """Affiche le bandeau d'en-tête du dashboard et ses indicateurs clés.

    Le bandeau principal (titre, sous-titre, mention du cours et des
    auteurs) est toujours affiché. Si des DataFrames sont fournis, quatre
    cartes statistiques supplémentaires sont rendues : nombre de profils
    analysés, nombre de points de données, nombre de Reynolds disponibles
    et taux de convergence XFoil.

    .. note::
        Fonction de rendu pur côté Streamlit : elle ne retourne aucune
        valeur, tout l'affichage se fait par effet de bord via
        ``st.markdown``.

    :param df_xfoil: DataFrame XFoil brut, utilisé pour calculer les
        indicateurs clés. Si ``None``, seul le bandeau principal est
        affiché (utilisé par exemple pour la page « Prédiction ML », qui
        ne dépend pas du dataset XFoil).
    :type df_xfoil: pandas.DataFrame, optional
    :param df_ml: DataFrame ML brut (actuellement non exploité pour le
        calcul des indicateurs, réservé pour extension future).
    :type df_ml: pandas.DataFrame, optional
    :returns: Rien (rendu Streamlit par effet de bord).
    :rtype: None
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1;">
                <h1 style="color: white; margin: 0; font-size: 2rem; font-weight: 600;">
                    AeroPredict
                </h1>
                <p style="color: rgba(255,255,255,0.9); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
                    Optimisation de profils aérodynamiques assistée par Machine Learning
                </p>
            </div>
            <div style="text-align: right;">
                <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.75rem;">
                    MGA 802
                </p>
                <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.7rem;">
                    Blanchard / Mechref / Condette
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_xfoil is not None and df_ml is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style="background:{COLORS['card_bg']};border-radius:12px;padding:1rem;border-left:4px solid {COLORS['accent']};box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size:0.75rem;color:{COLORS['text_light']};text-transform:uppercase;">Profils analysés</div>
                <div style="font-size:1.5rem;font-weight:600;color:{COLORS['primary']};">{df_xfoil['naca'].nunique():,}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background:{COLORS['card_bg']};border-radius:12px;padding:1rem;border-left:4px solid {COLORS['success']};box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size:0.75rem;color:{COLORS['text_light']};text-transform:uppercase;">Points de données</div>
                <div style="font-size:1.5rem;font-weight:600;color:{COLORS['primary']};">{len(df_xfoil):,}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            re_vals = sorted(df_xfoil["Re"].unique())
            st.markdown(f"""
            <div style="background:{COLORS['card_bg']};border-radius:12px;padding:1rem;border-left:4px solid {COLORS['ml']};box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size:0.75rem;color:{COLORS['text_light']};text-transform:uppercase;">Reynolds disponibles</div>
                <div style="font-size:1.5rem;font-weight:600;color:{COLORS['primary']};">{len(re_vals)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            conv_rate = df_xfoil["converged"].mean() * 100
            st.markdown(f"""
            <div style="background:{COLORS['card_bg']};border-radius:12px;padding:1rem;border-left:4px solid {COLORS['warning']};box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-size:0.75rem;color:{COLORS['text_light']};text-transform:uppercase;">Convergence XFoil</div>
                <div style="font-size:1.5rem;font-weight:600;color:{COLORS['primary']};">{conv_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)


def render_sidebar():
    """Affiche la barre latérale stylisée et collecte les choix de navigation.

    La sidebar contient, dans l'ordre : le logo et le titre de
    l'application, le menu de navigation entre les six pages du
    dashboard, le réglage de disposition des graphiques (1 ou 2
    colonnes), et un encart de pied de page rappelant le contexte du
    projet (cours, auteurs, outils utilisés).

    :returns: Un couple ``(page, disposition)`` où :

        * ``page`` est le nom de la page sélectionnée (parmi
          ``"Polaires"``, ``"Performance ML"``, ``"Validation ML"``,
          ``"Optimisation"``, ``"Dataset"``, ``"Prédiction ML"``) ;
        * ``disposition`` vaut ``"2 colonnes"`` ou ``"1 colonne"`` selon
          le choix de l'utilisateur.
    :rtype: tuple(str, str)
    """
    with st.sidebar:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        ">
            <img src="data:image/png;base64,{_logo_base64()}"
                 style="height: 90px; object-fit: contain; margin-bottom: 0.4rem;"
                 onerror="this.style.display='none'" />
            <h3 style="color: white; margin: 0.5rem 0 0 0;">AeroPredict</h3>
            <p style="color: #94A3B8; font-size: 0.7rem; margin: 0.25rem 0 0 0;">
                Aérodynamique par ML
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "**Navigation**",
            ["Polaires", "Performance ML", "Validation ML", "Optimisation", "Dataset", "Prédiction ML"],
            format_func=lambda x: x,
        )

        st.markdown("---")

        st.markdown("**Paramètres**")
        disposition = st.radio(
            "Disposition des graphiques",
            ["2 colonnes", "1 colonne"],
            horizontal=True,
        )

        st.markdown("---")

        st.markdown(f"""
        <div style="background:rgba(59,130,246,0.1);border-radius:8px;padding:0.75rem;margin-top:1rem;border:1px solid rgba(59,130,246,0.2);">
            <p style="font-size:0.7rem;color:#94A3B8;margin:0;text-align:center;">
                <strong style="color:#60A5FA;">MGA 802</strong><br>
                Blanchard / Mechref / Condette<br>
                XFoil • Réseau de neurones • Fluent
            </p>
        </div>
        """, unsafe_allow_html=True)

    return page, disposition


# ── Pages du dashboard ──

def page_polaires(df_merge: pd.DataFrame, fluent: dict, disposition: str) -> None:
    """Affiche la page « Polaires » : comparaison XFoil / ML / Fluent.

    L'utilisateur choisit un profil et un nombre de Reynolds parmi les
    valeurs disponibles dans ``df_merge`` ; quatre graphiques sont alors
    tracés (CL, CD, CM en fonction de l'angle d'attaque, ainsi que la
    finesse CL/CD), chacun superposant les courbes XFoil et ML, et les
    points Fluent lorsqu'ils sont disponibles pour ce couple
    ``(profil, Re)``. Un panneau dépliable permet de consulter les
    données tabulaires sous-jacentes.

    :param df_merge: DataFrame fusionné XFoil/ML tel que retourné par
        :func:`charger_donnees` (colonnes ``CL_xfoil``, ``CL_ml``, etc.).
    :type df_merge: pandas.DataFrame
    :param fluent: Dictionnaire des données Fluent indexées par
        ``(profil, Re)``, tel que retourné par :func:`charger_fluent`.
    :type fluent: dict
    :param disposition: ``"2 colonnes"`` ou ``"1 colonne"`` ; contrôle la
        mise en page des quatre graphiques principaux.
    :type disposition: str
    :returns: Rien (rendu Streamlit par effet de bord).
    :rtype: None
    """
    st.header("Polaires aérodynamiques")
    st.markdown("Comparaison des coefficients aérodynamiques entre XFoil, le modèle ML et Ansys Fluent.")

    col_a, col_b = st.columns(2)
    profils = sorted(df_merge["naca"].unique())
    profils_fluent = sorted({p for p, _ in fluent})

    with col_a:
        defaut = profils.index("naca0106") if "naca0106" in profils else 0
        profil = st.selectbox("Sélectionner un profil", profils, index=defaut)
    with col_b:
        re_dispo = sorted(df_merge[df_merge["naca"] == profil]["Re"].unique())
        re_val = st.select_slider("Nombre de Reynolds", re_dispo, format_func=re_label)

    sous = (df_merge[(df_merge["naca"] == profil) & (df_merge["Re"] == re_val)]
            .sort_values("alpha"))
    df_flu = fluent.get((profil, re_val))

    if df_flu is not None:
        st.success(f"Données Ansys Fluent disponibles pour **{profil}** à {re_label(re_val)}")

    conv = sous["converged"].mean() * 100
    st.caption(f"{len(sous)} points · convergence XFoil : {conv:.0f} %")

    st.markdown("---")

    # Spécification des trois courbes de coefficients : (clé colonne,
    # titre complet, libellé compact pour l'axe Y en notation LaTeX-like HTML).
    courbes = [
        ("CL", "Coefficient de portance", "c<sub>l</sub>"),
        ("CD", "Coefficient de traînée", "c<sub>d</sub>"),
        ("CM", "Coefficient de moment", "c<sub>m</sub>"),
    ]

    # Position de la légende adaptée à chaque type de courbe pour éviter
    # qu'elle ne masque les données (les coefficients ont des formes
    # caractéristiques différentes : CL croissant, CD en U, CM variable).
    positions_legendes = {
        "CL": dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        "CD": dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        "CM": dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02),
    }

    figs = []
    for coef, titre_long, titre_court in courbes:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sous["alpha"], y=sous[f"{coef}_xfoil"],
            name="XFoil", mode="lines+markers",
            line=dict(color=COLORS["xfoil"], width=2.5),
            marker=dict(size=6, symbol="circle", opacity=0.8),
            legendgroup="XFoil",
        ))
        fig.add_trace(go.Scatter(
            x=sous["alpha"], y=sous[f"{coef}_ml"],
            name="Modèle ML", mode="lines+markers",
            line=dict(color=COLORS["ml"], width=2.5, dash="dash"),
            marker=dict(size=6, symbol="square", opacity=0.8),
            legendgroup="ML",
        ))
        if df_flu is not None and coef in df_flu.columns:
            fig.add_trace(go.Scatter(
                x=df_flu["alpha_deg"], y=df_flu[coef],
                name="Ansys Fluent", mode="markers",
                marker=dict(color=COLORS["fluent"], size=10, symbol="diamond",
                            line=dict(width=1, color="white")),
                legendgroup="Fluent",
            ))

        legende_config = dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
            **positions_legendes[coef]
        )

        fig.update_layout(
            title=dict(text=f"<b>{titre_long}</b>", x=0.5, y=0.95,
                       xanchor="center", yanchor="top", xref="paper",
                       font=dict(size=16, color=COLORS["primary"])),
            xaxis=dict(title=dict(text="<b>Angle d'attaque α (°)</b>", font=dict(size=12)),
                       showgrid=True, gridwidth=1, gridcolor="lightgray",
                       showline=True, linewidth=2, linecolor="black", mirror=True,
                       ticks="outside", tickwidth=2, ticklen=8),
            yaxis=dict(title=dict(text=f"<b>{titre_court}</b>", font=dict(size=12)),
                       showgrid=True, gridwidth=1, gridcolor="lightgray",
                       showline=True, linewidth=2, linecolor="black", mirror=True,
                       ticks="outside", tickwidth=2, ticklen=8),
            height=450, margin=dict(t=80, b=50, l=60, r=30),
            legend=legende_config, plot_bgcolor="white", hovermode="x unified",
            paper_bgcolor=COLORS["background"],
        )
        figs.append(fig)

    # Quatrième graphique : finesse CL/CD, calculée à la volée (non
    # stockée dans df_merge) à partir des colonnes XFoil et ML.
    fig_ld = go.Figure()
    fig_ld.add_trace(go.Scatter(
        x=sous["alpha"], y=sous["CL_xfoil"] / sous["CD_xfoil"],
        name="XFoil", mode="lines+markers",
        line=dict(color=COLORS["xfoil"], width=2.5),
        marker=dict(size=6, symbol="circle", opacity=0.8),
    ))
    fig_ld.add_trace(go.Scatter(
        x=sous["alpha"], y=sous["CL_ml"] / sous["CD_ml"],
        name="Modèle ML", mode="lines+markers",
        line=dict(color=COLORS["ml"], width=2.5, dash="dash"),
        marker=dict(size=6, symbol="square", opacity=0.8),
    ))
    if df_flu is not None and "LD" in df_flu.columns:
        fig_ld.add_trace(go.Scatter(
            x=df_flu["alpha_deg"], y=df_flu["LD"],
            name="Ansys Fluent", mode="markers",
            marker=dict(color=COLORS["fluent"], size=10, symbol="diamond",
                        line=dict(width=1, color="white")),
        ))

    fig_ld.update_layout(
        title=dict(text="<b>Coefficient de finesse</b>", x=0.5, y=0.95,
                   xanchor="center", yanchor="top", xref="paper",
                   font=dict(size=16, color=COLORS["primary"])),
        xaxis=dict(title=dict(text="<b>Angle d'attaque α (°)</b>", font=dict(size=12)),
                   showgrid=True, gridwidth=1, gridcolor="lightgray",
                   showline=True, linewidth=2, linecolor="black", mirror=True),
        yaxis=dict(title=dict(text="<b>c<sub>l</sub> / c<sub>d</sub></b>", font=dict(size=12)),
                   showgrid=True, gridwidth=1, gridcolor="lightgray",
                   showline=True, linewidth=2, linecolor="black", mirror=True),
        height=450, margin=dict(t=80, b=50, l=60, r=30),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1,
                    font=dict(size=11), yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        plot_bgcolor="white", paper_bgcolor=COLORS["background"], hovermode="x unified",
    )
    figs.append(fig_ld)

    if disposition == "2 colonnes":
        st.subheader("Coefficients aérodynamiques")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figs[0], use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(figs[2], use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.plotly_chart(figs[1], use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(figs[3], use_container_width=True, config={"displayModeBar": False})
    else:
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Voir les données détaillées"):
        tab1, tab2, tab3 = st.tabs(["XFoil vs ML", "Ansys Fluent", "Comparaison"])
        with tab1:
            df_compare = sous[["alpha", "CL_xfoil", "CL_ml", "CD_xfoil", "CD_ml",
                               "CM_xfoil", "CM_ml"]].copy()
            df_compare["Finesse_XFoil"] = df_compare["CL_xfoil"] / df_compare["CD_xfoil"]
            df_compare["Finesse_ML"] = df_compare["CL_ml"] / df_compare["CD_ml"]
            st.dataframe(df_compare.round(4), use_container_width=True, hide_index=True)
        with tab2:
            if df_flu is not None:
                st.dataframe(df_flu.round(4), use_container_width=True, hide_index=True)
            else:
                st.info("Aucune donnée Fluent disponible")
        with tab3:
            if df_flu is not None:
                df_merged = sous[["alpha", "CL_xfoil", "CL_ml", "CD_xfoil", "CD_ml"]].copy()
                df_flu_subset = df_flu[["alpha_deg", "CL", "CD", "LD"]].copy()
                df_flu_subset.columns = ["alpha", "CL_Fluent", "CD_Fluent", "Finesse_Fluent"]
                df_comparison = pd.merge(df_merged, df_flu_subset, on="alpha", how="inner")
                st.dataframe(df_comparison.round(4), use_container_width=True, hide_index=True)
            else:
                st.info("Données Fluent non disponibles")


def page_performance(df_merge: pd.DataFrame) -> None:
    """Affiche la page « Performance ML » : métriques globales du modèle.

    Calcule et affiche, pour chacun des trois coefficients (``CL``,
    ``CD``, ``CM``) : le R², la MAE et la RMSE entre les valeurs XFoil
    (référence) et les valeurs prédites par le modèle ML ; un nuage de
    points « Réel vs Prédit » avec la droite ``y = x`` idéale ; et un
    histogramme de la distribution des erreurs absolues, avec la MAE
    superposée en ligne verticale.

    :param df_merge: DataFrame fusionné XFoil/ML tel que retourné par
        :func:`charger_donnees`.
    :type df_merge: pandas.DataFrame
    :returns: Rien (rendu Streamlit par effet de bord).
    :rtype: None
    """
    st.header("Performance du modèle ML")
    st.markdown("Évaluation quantitative des prédictions du réseau de neurones multi-tâches.")

    seulement_convergees = st.checkbox("Évaluer uniquement sur les points XFoil convergés", value=True)
    df_eval = df_merge[df_merge["converged"]] if seulement_convergees else df_merge
    st.caption(f"{len(df_eval):,} points d'évaluation")

    couleurs_coefs = {
        "CL": {"principal": COLORS["xfoil"], "fonce": "#0B79D0", "nom": "Coefficient de portance"},
        "CD": {"principal": COLORS["fluent"], "fonce": "#388E3C", "nom": "Coefficient de traînée"},
        "CM": {"principal": COLORS["ml"], "fonce": "#C2185B", "nom": "Coefficient de moment"},
    }
    labels_coef = {"CL": "c<sub>l</sub>", "CD": "c<sub>d</sub>", "CM": "c<sub>m</sub>"}

    colonnes_metriques = st.columns(3)
    toutes_metriques = {}
    for col_st, coef in zip(colonnes_metriques, ["CL", "CD", "CM"]):
        m = calculer_metriques(df_eval[f"{coef}_xfoil"].values, df_eval[f"{coef}_ml"].values)
        toutes_metriques[coef] = m
        with col_st:
            st.markdown(f"<h3 style='margin-bottom:0;color:{COLORS['primary']};'>{labels_coef[coef]}</h3>", unsafe_allow_html=True)
            st.metric("R²", f"{m['R2']:.4f}")
            st.metric("MAE", f"{m['MAE']:.5f}")
            st.metric("RMSE", f"{m['RMSE']:.5f}")

    st.divider()

    n_points = st.slider("Points affichés pour la dispersion", 1_000, 50_000, 15_000, step=1_000)
    echantillon = df_eval.sample(min(n_points, len(df_eval)), random_state=42)

    st.subheader("Dispersion : Prédictions vs Références (XFoil)")
    colonnes_dispersion = st.columns(3)
    for col_st, coef in zip(colonnes_dispersion, ["CL", "CD", "CM"]):
        x = echantillon[f"{coef}_xfoil"]
        y = echantillon[f"{coef}_ml"]
        borne = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
        c_p = couleurs_coefs[coef]["principal"]
        nom_complet = couleurs_coefs[coef]["nom"]

        fig_disp = go.Figure()
        fig_disp.add_trace(go.Scattergl(x=x, y=y, mode="markers",
                                        marker=dict(size=4, color=c_p, opacity=0.5), showlegend=False))
        fig_disp.add_trace(go.Scatter(x=borne, y=borne, mode="lines",
                                      line=dict(color="#000000", dash="dash", width=2), name="Idéal (y=x)"))
        fig_disp.update_layout(
            title=dict(text=f"<b>{nom_complet}</b>", x=0.5, y=0.95, xanchor="center", yanchor="top",
                       xref="paper", font=dict(size=14, color=COLORS["primary"])),
            xaxis=dict(title=dict(text=f"<b>{nom_complet} Réel</b>", font=dict(size=11)),
                       showgrid=True, gridwidth=1, gridcolor="lightgray",
                       showline=True, linewidth=1.5, linecolor="black", mirror=True, zeroline=False),
            yaxis=dict(title=dict(text=f"<b>{nom_complet} Prédit</b>", font=dict(size=11)),
                       showgrid=True, gridwidth=1, gridcolor="lightgray",
                       showline=True, linewidth=1.5, linecolor="black", mirror=True, zeroline=False),
            height=380, margin=dict(t=60, b=50, l=50, r=20),
            plot_bgcolor="white", paper_bgcolor=COLORS["background"],
            legend=dict(x=0.05, y=0.95, yanchor="top", bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="black", borderwidth=1),
        )
        col_st.plotly_chart(fig_disp, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Distribution de la valeur absolue de l'erreur")
    colonnes_erreur = st.columns(3)
    for col_st, coef in zip(colonnes_erreur, ["CL", "CD", "CM"]):
        erreurs_abs = np.abs(df_eval[f"{coef}_xfoil"].values - df_eval[f"{coef}_ml"].values)
        mae_val = toutes_metriques[coef]["MAE"]
        c_p = couleurs_coefs[coef]["principal"]
        c_f = couleurs_coefs[coef]["fonce"]
        nom_complet = couleurs_coefs[coef]["nom"]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=erreurs_abs, autobinx=True,
                                        marker=dict(color=c_p, opacity=0.75,
                                                    line=dict(color=c_f, width=0.5)),
                                        showlegend=False))
        fig_hist.add_vline(x=mae_val, line_dash="dash", line_color=c_f, line_width=2)
        fig_hist.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                      line=dict(color=c_f, dash="dash", width=2),
                                      name=f"MAE = {mae_val:.6f}"))
        fig_hist.update_layout(
            title=dict(text=f"<b>Distribution |Erreur| — {coef}</b>", x=0.5, y=0.95,
                       xanchor="center", yanchor="top", xref="paper", font=dict(size=14, color=COLORS["primary"])),
            xaxis=dict(title=dict(text=f"<b>|{nom_complet} Réel − Prédit|</b>", font=dict(size=11)),
                       showgrid=True, gridwidth=1, gridcolor="lightgray",
                       showline=True, linewidth=1.5, linecolor="black", mirror=True),
            yaxis=dict(title=dict(text="<b>Fréquence</b>", font=dict(size=11)),
                       showgrid=True, gridwidth=1, gridcolor="lightgray",
                       showline=True, linewidth=1.5, linecolor="black", mirror=True),
            height=380, margin=dict(t=60, b=50, l=50, r=20),
            plot_bgcolor="white", paper_bgcolor=COLORS["background"],
            legend=dict(x=0.95, y=0.95, xanchor="right", yanchor="top",
                        bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1),
        )
        col_st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


def page_validation_model(df_merge: pd.DataFrame) -> None:
    """Affiche la page « Validation ML » : analyse fine du comportement du modèle.

    Cette page permet de sélectionner un ou plusieurs profils ainsi
    qu'un nombre de Reynolds, puis présente :

    #. la dérivée numérique ``dCL/dα`` (calculée par différences finies
       via ``numpy.gradient``), comparant XFoil et ML, profil par
       profil ;
    #. le diagramme polaire ``CD(CL)`` restreint à la portion *avant
       décrochage* (c'est-à-dire jusqu'au ``CL`` maximal observé), avec
       une régression polynomiale d'ordre 4 optionnelle sur les points
       XFoil et son R² associé ;
    #. un tableau récapitulatif des coefficients de régression et de
       l'angle de décrochage par profil ;
    #. un tableau comparatif des métriques (R², MAE) par profil.

    :param df_merge: DataFrame fusionné XFoil/ML tel que retourné par
        :func:`charger_donnees`.
    :type df_merge: pandas.DataFrame
    :returns: Rien (rendu Streamlit par effet de bord). La fonction peut
        retourner précocement (``return`` simple) si aucun profil n'est
        sélectionné ou si la sélection ne correspond à aucune donnée.
    :rtype: None
    """
    st.header("Validation du modèle ML")
    st.markdown("Analyse approfondie du comportement du modèle.")

    col_filtre1, col_filtre2, col_filtre3 = st.columns(3)

    with col_filtre1:
        seulement_convergees = st.checkbox("Points XFoil convergés uniquement", value=True, key="validation_converged")

    df_base = df_merge[df_merge["converged"]] if seulement_convergees else df_merge

    with col_filtre2:
        profils_dispo = sorted(df_base["naca"].unique())
        profil_selection = st.multiselect("Profils à analyser", profils_dispo, default=[])

    with col_filtre3:
        re_dispo = sorted(df_base["Re"].unique())
        re_selection = st.select_slider("Nombre de Reynolds", options=re_dispo, format_func=re_label)

    if not profil_selection:
        st.info("Veuillez sélectionner au moins un profil.")
        return

    df_filtre = df_base[(df_base["naca"].isin(profil_selection)) & (df_base["Re"] == re_selection)].copy()

    if df_filtre.empty:
        st.warning("Aucune donnée pour la sélection choisie.")
        return

    df_filtre = df_filtre.sort_values(["naca", "alpha"])
    st.markdown("---")

    # Graphique 1 : dCl/dalpha
    st.subheader("Dérivée de portance dCL/dα")

    fig_dclda = go.Figure()
    for profil in profil_selection:
        df_profil = df_filtre[df_filtre["naca"] == profil].sort_values("alpha").copy()
        if len(df_profil) < 3:
            # La dérivée numérique par np.gradient nécessite au moins 2
            # points (et 3 pour une estimation raisonnable de la tendance).
            continue

        alpha_vals = df_profil["alpha"].values
        cl_xfoil = df_profil["CL_xfoil"].values
        cl_ml = df_profil["CL_ml"].values

        dcl_dalpha_xfoil = np.gradient(cl_xfoil, alpha_vals)
        dcl_dalpha_ml = np.gradient(cl_ml, alpha_vals)

        fig_dclda.add_trace(go.Scatter(
            x=alpha_vals, y=dcl_dalpha_xfoil,
            mode="lines+markers",
            name=f"{profil} (XFoil)",
            line=dict(color=COLORS["xfoil"], width=2.5),
            marker=dict(size=6, symbol="circle", color=COLORS["xfoil"]),
        ))
        fig_dclda.add_trace(go.Scatter(
            x=alpha_vals, y=dcl_dalpha_ml,
            mode="lines+markers",
            name=f"{profil} (ML)",
            line=dict(color=COLORS["ml"], width=2.5, dash="dash"),
            marker=dict(size=6, symbol="square", color=COLORS["ml"]),
        ))

    fig_dclda.update_layout(
        title=dict(text=f"<b>dCL/dα — {re_label(re_selection)}</b>", x=0.5, xanchor="center", y=0.95, yanchor="top", xref="paper", font=dict(size=16, color=COLORS["primary"])),
        xaxis=dict(title=dict(text="<b>Angle d'attaque α (°)</b>", font=dict(size=12)), showgrid=True, gridwidth=1, gridcolor="lightgray", showline=True, linewidth=2, linecolor="black", mirror=True),
        yaxis=dict(title=dict(text="<b>dCL/dα (par degré)</b>", font=dict(size=12)), showgrid=True, gridwidth=1, gridcolor="lightgray", showline=True, linewidth=2, linecolor="black", mirror=True),
        height=500, margin=dict(t=80, b=50, l=60, r=30),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="black", borderwidth=1, font=dict(size=10), yanchor="bottom", y=0.02, xanchor="left", x=0.02),
        plot_bgcolor="white", paper_bgcolor=COLORS["background"], hovermode="x unified",
    )
    st.plotly_chart(fig_dclda, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # Graphique 2 : CD vs CL
    st.subheader("Diagramme polaire CD(CL) — Avant décrochage")

    col_config1, col_config2 = st.columns(2)
    with col_config1:
        afficher_points_xfoil = st.checkbox("Afficher les points XFoil", value=True)
    with col_config2:
        afficher_regression = st.checkbox("Afficher la régression polynomiale (ordre 4)", value=True)

    fig_cd_cl = go.Figure()

    for profil in profil_selection:
        df_profil = df_filtre[df_filtre["naca"] == profil].sort_values("alpha").copy()
        if len(df_profil) < 5:
            continue

        cl_xfoil = df_profil["CL_xfoil"].values
        cd_xfoil = df_profil["CD_xfoil"].values
        cl_ml = df_profil["CL_ml"].values
        cd_ml = df_profil["CD_ml"].values

        # Le décrochage correspond au CL maximal : on ne conserve que la
        # portion de la polaire avant ce point pour la régression, le
        # comportement post-décrochage étant fortement non monotone.
        idx_max = np.argmax(cl_xfoil)
        mask_avant_stall = np.arange(len(cl_xfoil)) <= idx_max

        cl_avant = cl_xfoil[mask_avant_stall]
        cd_avant = cd_xfoil[mask_avant_stall]
        cl_ml_avant = cl_ml[mask_avant_stall]
        cd_ml_avant = cd_ml[mask_avant_stall]

        if afficher_points_xfoil:
            fig_cd_cl.add_trace(go.Scatter(
                x=cl_avant, y=cd_avant,
                mode="markers",
                name=f"{profil} (XFoil points)",
                marker=dict(color=COLORS["xfoil"], size=7, symbol="circle", opacity=0.7),
            ))

        if afficher_regression and len(cl_avant) >= 5:
            coeffs = np.polyfit(cl_avant, cd_avant, 4)
            a, b, c, d, e = coeffs
            cl_fit = np.linspace(cl_avant.min(), cl_avant.max(), 200)
            cd_fit = a * cl_fit ** 4 + b * cl_fit ** 3 + c * cl_fit ** 2 + d * cl_fit + e
            cd_pred = a * cl_avant ** 4 + b * cl_avant ** 3 + c * cl_avant ** 2 + d * cl_avant + e
            ss_res = np.sum((cd_avant - cd_pred) ** 2)
            ss_tot = np.sum((cd_avant - np.mean(cd_avant)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            fig_cd_cl.add_trace(go.Scatter(
                x=cl_fit, y=cd_fit,
                mode="lines",
                name=f"{profil} (poly deg4, R²={r2:.4f})",
                line=dict(color=COLORS["accent"], width=2.5),
            ))

        fig_cd_cl.add_trace(go.Scatter(
            x=cl_ml_avant, y=cd_ml_avant,
            mode="lines+markers",
            name=f"{profil} (ML)",
            line=dict(color=COLORS["ml"], width=2.5, dash="dash"),
            marker=dict(size=5, symbol="square", color=COLORS["ml"], opacity=0.8),
        ))

    fig_cd_cl.update_layout(
        title=dict(text=f"<b>Diagramme polaire CD(CL) — {re_label(re_selection)}</b>", x=0.5, xanchor="center", y=0.95, yanchor="top", xref="paper", font=dict(size=16, color=COLORS["primary"])),
        xaxis=dict(title=dict(text="<b>Coefficient de portance C<sub>L</sub></b>", font=dict(size=12)), showgrid=True, gridwidth=1, gridcolor="lightgray", showline=True, linewidth=2, linecolor="black", mirror=True),
        yaxis=dict(title=dict(text="<b>Coefficient de traînée C<sub>D</sub></b>", font=dict(size=12)), showgrid=True, gridwidth=1, gridcolor="lightgray", showline=True, linewidth=2, linecolor="black", mirror=True),
        height=550, margin=dict(t=80, b=50, l=60, r=30),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="black", borderwidth=1, font=dict(size=9), yanchor="top", y=0.98, xanchor="left", x=0.02),
        plot_bgcolor="white", paper_bgcolor=COLORS["background"], hovermode="closest",
    )
    st.plotly_chart(fig_cd_cl, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Coefficients des régressions polynomiales"):
        data_reg = []
        for profil in profil_selection:
            df_profil = df_filtre[df_filtre["naca"] == profil].sort_values("alpha").copy()
            if len(df_profil) >= 5:
                cl_xfoil = df_profil["CL_xfoil"].values
                cd_xfoil = df_profil["CD_xfoil"].values
                idx_max = np.argmax(cl_xfoil)
                cl_avant = cl_xfoil[:idx_max + 1]
                cd_avant = cd_xfoil[:idx_max + 1]
                if len(cl_avant) >= 5:
                    coeffs = np.polyfit(cl_avant, cd_avant, 4)
                    a, b, c, d, e = coeffs
                    cd_pred = a * cl_avant ** 4 + b * cl_avant ** 3 + c * cl_avant ** 2 + d * cl_avant + e
                    ss_res = np.sum((cd_avant - cd_pred) ** 2)
                    ss_tot = np.sum((cd_avant - np.mean(cd_avant)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    data_reg.append({
                        "Profil": profil,
                        "a (CL⁴)": f"{a:.6e}",
                        "b (CL³)": f"{b:.6e}",
                        "c (CL²)": f"{c:.6e}",
                        "d (CL)": f"{d:.6e}",
                        "e (constante)": f"{e:.6e}",
                        "R²": f"{r2:.5f}",
                        "α décrochage": f"{df_profil['alpha'].values[idx_max]:.1f}°",
                        "CL max": f"{cl_avant[-1]:.3f}",
                    })
        if data_reg:
            st.dataframe(pd.DataFrame(data_reg), use_container_width=True, hide_index=True)
        else:
            st.info("Aucune régression valide calculée.")

    st.markdown("---")
    st.subheader("Comparaison des métriques par profil")

    metriques_profil = []
    for profil in profil_selection:
        df_profil = df_filtre[df_filtre["naca"] == profil]
        if len(df_profil) > 0:
            m_cl = calculer_metriques(df_profil["CL_xfoil"].values, df_profil["CL_ml"].values)
            m_cd = calculer_metriques(df_profil["CD_xfoil"].values, df_profil["CD_ml"].values)
            idx_max = np.argmax(df_profil["CL_xfoil"].values)
            alpha_stall = df_profil["alpha"].values[idx_max]
            metriques_profil.append({
                "Profil": profil,
                "R² CL": f"{m_cl['R2']:.4f}",
                "MAE CL": f"{m_cl['MAE']:.5f}",
                "R² CD": f"{m_cd['R2']:.4f}",
                "MAE CD": f"{m_cd['MAE']:.6f}",
                "α décrochage": f"{alpha_stall:.1f}°",
                "N points": len(df_profil),
            })
    if metriques_profil:
        st.dataframe(pd.DataFrame(metriques_profil), use_container_width=True, hide_index=True)


def page_optimisation(df_ml: pd.DataFrame) -> None:
    """Affiche la page « Optimisation » : recherche du meilleur profil par finesse.

    Pour un nombre de Reynolds et une plage d'angles d'attaque donnés,
    cette page calcule, pour chaque profil du dataset ML, la finesse
    maximale atteinte (``CL / CD``), puis classe et affiche les ``top_n``
    profils les plus performants sous forme de diagramme en barres
    horizontales, accompagné d'un tableau de détails (angle optimal,
    coefficients, épaisseur, cambrure).

    .. note::
        Les points dont ``CD <= 1e-5`` sont écartés en amont du calcul
        afin d'éviter les valeurs de finesse non physiques (division par
        une traînée quasi nulle).

    :param df_ml: DataFrame des prédictions ML brutes (issu de
        :data:`CSV_ML`), contenant au minimum les colonnes ``naca``,
        ``source``, ``alpha``, ``Re``, ``CL``, ``CD``, ``CM``, ``t`` et
        ``camber``.
    :type df_ml: pandas.DataFrame
    :returns: Rien (rendu Streamlit par effet de bord).
    :rtype: None
    """
    st.header("Optimisation — Meilleur profil pour vos conditions de vol")
    st.markdown("Objectif : identifier le profil **maximisant le ratio CL/CD**.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        re_val = st.select_slider("Nombre de Reynolds", sorted(df_ml["Re"].unique()), format_func=re_label)
    with col_b:
        alpha_min, alpha_max = st.slider("Plage d'angles d'attaque (°)", -6.0, 23.5, (0.0, 10.0), step=0.5)
    with col_c:
        top_n = st.number_input("Nombre de profils affichés", 5, 50, 10)

    sous = df_ml[(df_ml["Re"] == re_val) & (df_ml["alpha"] >= alpha_min) & (df_ml["alpha"] <= alpha_max) & (df_ml["CD"] > 1e-5)].copy()
    sous["finesse"] = sous["CL"] / sous["CD"]

    # Pour chaque profil, on ne retient que l'angle d'attaque offrant la
    # meilleure finesse dans la plage choisie (recherche du maximum local
    # par groupe).
    idx_max = sous.groupby("naca")["finesse"].idxmax()
    meilleurs = sous.loc[idx_max].sort_values("finesse", ascending=False).head(int(top_n))

    fig = go.Figure(go.Bar(
        x=meilleurs["finesse"][::-1],
        y=meilleurs["naca"][::-1],
        orientation="h",
        marker_color=COLORS["accent"],
        text=[f"α = {a:+.1f}°" for a in meilleurs["alpha"][::-1]],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Top {int(top_n)} des finesses maximales — {re_label(re_val)}, α ∈ [{alpha_min}°, {alpha_max}°]",
        xaxis_title="Finesse maximale c<sub>l</sub>/c<sub>d</sub>",
        yaxis_title="Profil",
        height=max(400, 35 * int(top_n)),
        margin=dict(t=50, b=40),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Détails des meilleurs profils")
    tableau = meilleurs[["naca", "source", "alpha", "CL", "CD", "CM", "finesse", "t", "camber"]].rename(columns={
        "naca": "Profil", "source": "Source", "alpha": "α optimal (°)",
        "finesse": "CL/CD", "t": "Épaisseur", "camber": "Cambrure",
    })
    st.dataframe(tableau.style.format({
        "CL": "{:.4f}", "CD": "{:.5f}", "CM": "{:.4f}",
        "CL/CD": "{:.1f}", "α optimal (°)": "{:+.1f}",
        "Épaisseur": "{:.3f}", "Cambrure": "{:.3f}",
    }), use_container_width=True, hide_index=True)


def page_dataset(df_xfoil: pd.DataFrame, disposition: str) -> None:
    """Affiche la page « Dataset » : exploration statistique du corpus XFoil.

    Présente des indicateurs globaux (nombre de lignes, de profils
    uniques, de familles, taux de convergence), la répartition des
    profils par famille (préfixe alphabétique du nom NACA), le taux de
    convergence XFoil par nombre de Reynolds (avec un gradient de couleur
    reflétant la qualité de convergence), ainsi qu'un histogramme avec
    estimation de densité par noyau (KDE) pour chacune des 8 features
    géométriques du dataset.

    :param df_xfoil: DataFrame XFoil brut tel que retourné par
        :func:`charger_donnees`.
    :type df_xfoil: pandas.DataFrame
    :param disposition: ``"2 colonnes"`` ou ``"1 colonne"`` ; contrôle la
        mise en page des deux premiers graphiques (familles et
        convergence).
    :type disposition: str
    :returns: Rien (rendu Streamlit par effet de bord).
    :rtype: None
    """
    st.header("Exploration du dataset")
    st.markdown("Analyse statistique du corpus de profils généré par XFoil.")

    df_profils = df_xfoil.drop_duplicates("naca").copy()
    df_profils["famille"] = df_profils["naca"].str.extract(r"^([a-zA-Z]+)")[0].str.lower()

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Lignes", f"{len(df_xfoil):,}")
    with col_b:
        st.metric("Profils uniques", f"{df_xfoil['naca'].nunique():,}")
    with col_c:
        st.metric("Familles", f"{df_profils['famille'].nunique()}")
    with col_d:
        st.metric("Convergence XFoil", f"{df_xfoil['converged'].mean() * 100:.1f} %")

    familles = df_profils.groupby("famille")["naca"].count().sort_values(ascending=False)
    fig_fam = go.Figure(go.Bar(
        x=familles.index, y=familles.values,
        marker_color=COLORS["accent"], text=familles.values, textposition="outside",
    ))
    fig_fam.update_layout(title="Nombre de profils par famille", xaxis_title="Famille",
                          yaxis_title="Profils", height=420, margin=dict(t=50, b=40),
                          plot_bgcolor=COLORS["background"], paper_bgcolor=COLORS["background"])

    conv_re = df_xfoil.groupby("Re")["converged"].mean() * 100
    re_labels = [re_tick(r) for r in conv_re.index]

    def _couleur_conv_gradient(v: float) -> str:
        """Calcule une couleur hexadécimale interpolée selon le taux de convergence.

        Le gradient parcourt quatre teintes de référence (sombre →
        gris-bleu → turquoise → vert) sur la plage ``[70, 100]`` %, afin
        de mettre visuellement en évidence les nombres de Reynolds les
        moins bien convergés.

        :param v: Taux de convergence en pourcentage, sera tronqué à
            l'intervalle ``[70, 100]``.
        :type v: float
        :returns: Couleur au format hexadécimal ``"#rrggbb"``.
        :rtype: str
        """
        v = max(70, min(100, v))
        t = (v - 70) / 30.0
        colors = [(30, 41, 59), (71, 85, 105), (45, 212, 191), (5, 150, 105)]
        positions = [0.0, 0.4, 0.75, 1.0]
        for i in range(len(positions) - 1):
            if positions[i] <= t <= positions[i + 1]:
                local_t = (t - positions[i]) / (positions[i + 1] - positions[i])
                r = int(colors[i][0] + local_t * (colors[i + 1][0] - colors[i][0]))
                g = int(colors[i][1] + local_t * (colors[i + 1][1] - colors[i][1]))
                b = int(colors[i][2] + local_t * (colors[i + 1][2] - colors[i][2]))
                return f"#{r:02x}{g:02x}{b:02x}"
        return "#059669"

    couleurs_conv = [_couleur_conv_gradient(v) for v in conv_re.values]
    fig_conv = go.Figure(go.Bar(
        x=re_labels, y=conv_re.values, marker_color=couleurs_conv,
        text=[f"{v:.1f} %" for v in conv_re.values], textposition="outside",
    ))
    fig_conv.update_layout(
        title="Taux de convergence XFoil par nombre de Reynolds",
        xaxis=dict(title="Re", type="category", tickangle=-30),
        yaxis=dict(title="Convergence (%)", range=[0, 105]),
        height=420, margin=dict(t=50, b=60),
        plot_bgcolor=COLORS["background"], paper_bgcolor=COLORS["background"],
    )

    if disposition == "2 colonnes":
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_fam, use_container_width=True, config={"displayModeBar": False})
        col2.plotly_chart(fig_conv, use_container_width=True, config={"displayModeBar": False})
    else:
        st.plotly_chart(fig_fam, use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(fig_conv, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Distribution des features géométriques")
    features = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area"]
    feature = st.selectbox("Caractéristique", features)

    data = df_profils[feature].dropna().values
    kde = stats.gaussian_kde(data)
    nbins = 50
    hist_counts, bin_edges = np.histogram(data, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    x_dense = np.linspace(data.min(), data.max(), 200)
    y_dense = kde(x_dense)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(x=bin_centers, y=hist_counts, name="Histogramme",
                              marker_color=COLORS["accent"], opacity=0.8,
                              width=bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.01))
    fig_hist.add_trace(go.Scatter(x=x_dense, y=y_dense, name="Densité", mode="lines",
                                  line=dict(color=COLORS["ml"], width=3)))
    mediane = float(np.median(data))
    fig_hist.add_vline(x=mediane, line_dash="dash", line_color=COLORS["accent_light"], line_width=2,
                       annotation_text=f"médiane = {mediane:.3f}", annotation_position="top")
    fig_hist.update_layout(
        title=f"Distribution de {feature}", xaxis_title=feature, yaxis_title="Densité",
        height=500, margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), bargap=0.05,
        plot_bgcolor=COLORS["background"], paper_bgcolor=COLORS["background"],
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


# ── Page 6 : Prédiction ML avec Fluent (sans XFoil) ──────────────────

#: Configuration des sliders de saisie géométrique pour la page
#: « Prédiction ML ». Chaque entrée associe le nom interne d'une feature
#: (clé) à un tuple ``(label, min, max, valeur_par_défaut, pas, aide)``
#: utilisé pour construire dynamiquement les widgets ``st.slider``.
_GEO_CONFIG = {
    "t": ("Épaisseur relative  t/c", 0.02, 0.40, 0.12, 0.005, "Ratio épaisseur max / corde"),
    "camber": ("Cambrure max  m/c", 0.00, 0.12, 0.02, 0.002, "Ratio cambrure max / corde"),
    "x_t": ("Position épaisseur max  x_t/c", 0.10, 0.70, 0.30, 0.01, "Position longitudinale de l'épaisseur maximale"),
    "x_c": ("Position cambrure max  x_c/c", 0.00, 0.70, 0.40, 0.01, "Position longitudinale de la cambrure maximale"),
    "LE_radius": ("Rayon de bord d'attaque  r_LE/c", 0.001, 0.10, 0.016, 0.001, "Rayon de courbure au bord d'attaque normalisé"),
    "TE_angle": ("Angle de bord de fuite  β (°)", 0.0, 40.0, 12.0, 0.5, "Demi-angle d'ouverture au bord de fuite"),
    "t_over_xt": ("Ratio t / x_t", 0.05, 1.20, 0.40, 0.01, "Épaisseur relative divisée par sa position"),
    "area": ("Aire de section  A/c²", 0.01, 0.20, 0.077, 0.002, "Surface du profil normalisée"),
}


def _tracer_polaire_ml_fluent(
    df_pred: pd.DataFrame,
    disposition: str,
    df_flu: pd.DataFrame = None
) -> None:
    """Trace les quatre courbes de polaire (CL, CD, CM, finesse) pour la page Prédiction ML.

    Contrairement à :func:`page_polaires`, cette fonction ne dispose pas
    de référence XFoil (le profil saisi est arbitraire et n'existe pas
    nécessairement dans le dataset XFoil) : seules les prédictions ML et,
    le cas échéant, les points de référence Ansys Fluent sont superposés.

    :param df_pred: DataFrame de prédictions tel que retourné par
        :func:`predire_polaires` (colonnes ``alpha``, ``CL``, ``CD``,
        ``CM``, ``finesse``).
    :type df_pred: pandas.DataFrame
    :param disposition: ``"2 colonnes"`` ou ``"1 colonne"`` ; contrôle la
        mise en page des quatre graphiques.
    :type disposition: str
    :param df_flu: DataFrame Fluent optionnel pour le même profil et le
        même nombre de Reynolds, contenant les colonnes ``alpha_deg`` et
        au moins l'une de ``CL``/``CD``/``CM``/``LD``. Si ``None``, seules
        les courbes ML sont tracées.
    :type df_flu: pandas.DataFrame, optional
    :returns: Rien (rendu Streamlit par effet de bord).
    :rtype: None
    """
    spec_courbes = [
        ("CL", "Coefficient de portance", "c<sub>l</sub>"),
        ("CD", "Coefficient de traînée", "c<sub>d</sub>"),
        ("CM", "Coefficient de moment", "c<sub>m</sub>"),
        ("finesse", "Coefficient de finesse", "c<sub>l</sub> / c<sub>d</sub>"),
    ]

    # Correspondance entre le nom de colonne côté ML (df_pred) et le nom
    # de colonne équivalent côté Fluent (df_flu) ; la finesse est nommée
    # "LD" (lift-to-drag) dans les exports Fluent.
    flu_col_map = {"CL": "CL", "CD": "CD", "CM": "CM", "finesse": "LD"}

    figs = []
    for col_data, titre_long, titre_court in spec_courbes:
        fig = go.Figure()

        # ML (toujours présent)
        fig.add_trace(go.Scatter(
            x=df_pred["alpha"], y=df_pred[col_data],
            mode="lines+markers",
            line=dict(color=COLORS["ml"], width=2.5),
            marker=dict(size=6, symbol="circle", opacity=0.85),
            name="ML (prédiction)",
            hovertemplate="α = %{x:.1f}°<br>" + titre_court + " (ML) = %{y:.4f}<extra></extra>",
        ))

        # Fluent (si disponible)
        if df_flu is not None:
            flu_col = flu_col_map[col_data]
            if flu_col in df_flu.columns:
                fig.add_trace(go.Scatter(
                    x=df_flu["alpha_deg"], y=df_flu[flu_col],
                    mode="markers",
                    name="Ansys Fluent",
                    marker=dict(color=COLORS["fluent"], size=10, symbol="diamond",
                                line=dict(width=1, color="white")),
                    hovertemplate="α = %{x:.1f}°<br>" + titre_court + " (Fluent) = %{y:.4f}<extra></extra>",
                ))

        fig.update_layout(
            title=dict(text=f"<b>{titre_long}</b>", x=0.5, y=0.95,
                       xanchor="center", yanchor="top", xref="paper",
                       font=dict(size=16, color=COLORS["primary"])),
            xaxis=dict(title=dict(text="<b>Angle d'attaque α (°)</b>", font=dict(size=12)),
                       showgrid=True, gridwidth=1, gridcolor="lightgray",
                       showline=True, linewidth=2, linecolor="black", mirror=True,
                       ticks="outside", tickwidth=2, ticklen=8),
            yaxis=dict(title=dict(text=f"<b>{titre_court}</b>", font=dict(size=12)),
                       showgrid=True, gridwidth=1, gridcolor="lightgray",
                       showline=True, linewidth=2, linecolor="black", mirror=True,
                       ticks="outside", tickwidth=2, ticklen=8),
            height=430, margin=dict(t=80, b=50, l=65, r=30),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="black",
                        borderwidth=1, font=dict(size=11),
                        yanchor="top", y=0.98, xanchor="left", x=0.02),
            plot_bgcolor="white", paper_bgcolor=COLORS["background"],
            hovermode="x unified",
        )
        figs.append(fig)

    if disposition == "2 colonnes":
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figs[0], use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(figs[2], use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.plotly_chart(figs[1], use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(figs[3], use_container_width=True, config={"displayModeBar": False})
    else:
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def page_prediction_ml(disposition: str, fluent: dict) -> None:
    """Affiche la page « Prédiction ML » : génération de polaires pour un profil arbitraire.

    Cette page permet à l'utilisateur de définir librement les 8
    paramètres géométriques d'un profil (éventuellement pré-remplis à
    partir d'un profil connu disposant de données Fluent), ainsi que les
    conditions de vol (nombre de Reynolds, plage et résolution d'angle
    d'attaque), puis de déclencher une inférence du modèle ML pour
    générer la polaire complète correspondante. Si le profil choisi
    correspond à un préréglage Fluent disponible au même Reynolds, les
    points Fluent sont superposés à titre de comparaison.

    Le flux complet est : sélection d'un préréglage optionnel → réglage
    des sliders géométriques → vérifications de cohérence → réglage des
    conditions de vol → clic sur le bouton de génération → inférence →
    affichage des métriques clés (CL max, finesse max, CD min) et des
    courbes → export CSV optionnel des données générées.

    :param disposition: ``"2 colonnes"`` ou ``"1 colonne"`` ; transmis à
        :func:`_tracer_polaire_ml_fluent` pour la mise en page des
        graphiques.
    :type disposition: str
    :param fluent: Dictionnaire des données Fluent indexées par
        ``(profil, Re)``, tel que retourné par :func:`charger_fluent` ;
        utilisé à la fois pour proposer des préréglages et pour
        superposer les points de référence après inférence.
    :type fluent: dict
    :returns: Rien (rendu Streamlit par effet de bord). La fonction peut
        retourner précocement si les fichiers du modèle sont introuvables,
        si le chargement du modèle échoue, si l'utilisateur n'a pas
        encore cliqué sur le bouton de génération, ou si l'inférence lève
        une exception.
    :rtype: None
    """
    st.header("Prédiction ML — Polaires pour un profil quelconque")
    st.markdown("Entrez les **caractéristiques géométriques** de votre profil et les **conditions de vol**.")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        st.error("Fichiers modèle introuvables.")
        return

    model, pre = charger_modele()
    if model is None:
        st.error("Le modèle n'a pas pu être chargé.")
        return

    st.markdown("---")

    # ── Construction des presets depuis dataset_fluent.csv ──────────
    # Récupérer tous les profils disponibles dans fluent
    profils_disponibles = sorted(set(p for p, _ in fluent.keys()))

    if not profils_disponibles:
        st.warning("Aucun profil trouvé dans dataset_fluent.csv. Utilisez les paramètres personnalisés.")

    # Construire la liste des options
    options_presets = ["— Personnalisé —"] + [f"{p} (Fluent)" for p in profils_disponibles]

    st.subheader("Préréglage rapide (optionnel)")
    preset_choisi = st.selectbox("Partir d'un profil connu", options_presets)

    # Extraire les valeurs du preset : on prend la première ligne
    # disponible (toutes les lignes d'un même profil partagent la même
    # géométrie, seul alpha varie) du premier Reynolds rencontré.
    valeurs_preset = None
    if preset_choisi != "— Personnalisé —":
        profil_nom = preset_choisi.replace(" (Fluent)", "")
        # Récupérer les données du profil
        for (p, re), df in fluent.items():
            if p == profil_nom:
                row = df.iloc[0]
                valeurs_preset = {
                    "t": float(row["t"]),
                    "camber": float(row["camber"]),
                    "x_t": float(row["x_t"]),
                    "x_c": float(row["x_c"]),
                    "LE_radius": float(row["LE_radius"]),
                    "TE_angle": float(row["TE_angle"]),
                    "t_over_xt": float(row["t_over_xt"]),
                    "area": float(row["area"]),
                    "default_re": int(re),
                }
                break

    st.markdown("---")

    # Features géométriques
    st.subheader("Géométrie du profil")
    st.caption("Ces 8 paramètres définissent complètement la forme du profil.")

    geo = {}
    col_gauche, col_droite = st.columns(2)
    items = list(_GEO_CONFIG.items())
    moitie = (len(items) + 1) // 2

    for idx, (feat, (label, f_min, f_max, f_def, f_step, aide)) in enumerate(items):
        valeur_init = valeurs_preset[feat] if valeurs_preset else f_def
        decimales = max(0, -int(np.floor(np.log10(f_step))))
        valeur_init = round(valeur_init, decimales)
        col = col_gauche if idx < moitie else col_droite
        with col:
            geo[feat] = st.slider(label, min_value=f_min, max_value=f_max, value=float(valeur_init),
                                  step=f_step, format=f"%.{decimales}f", help=aide, key=f"slider_{feat}")

    # Vérifications de cohérence géométrique (purement informatives : ne
    # bloquent pas l'inférence, mais alertent l'utilisateur sur une
    # combinaison de paramètres potentiellement irréaliste).
    avertissements = []
    if geo["t_over_xt"] > 0 and abs(geo["t_over_xt"] - geo["t"] / max(geo["x_t"], 1e-6)) > 0.15:
        avertissements.append(f"⚠️ `t/x_t` ({geo['t_over_xt']:.3f}) incohérent avec `t`/{geo['x_t']:.3f} = {geo['t']/max(geo['x_t'],1e-6):.3f}.")
    if geo["camber"] > 0 and geo["x_c"] < 1e-4:
        avertissements.append("⚠️ Cambrure non nulle mais `x_c` ≈ 0.")
    for msg in avertissements:
        st.warning(msg)

    st.markdown("---")

    # Conditions de vol
    st.subheader("Conditions de vol")

    col_re, col_alpha = st.columns(2)
    with col_re:
        RE_VALEURS_MAJ = [50_000, 100_000, 200_000, 400_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
        re_defaut = valeurs_preset.get("default_re", 1_000_000) if valeurs_preset else 1_000_000
        re_val = st.select_slider("Nombre de Reynolds", options=RE_VALEURS_MAJ, value=re_defaut, format_func=re_label)
    with col_alpha:
        alpha_range = st.slider("Plage d'angles d'attaque α (°)", min_value=-10.0, max_value=25.0, value=(-5.0, 15.0), step=0.5)

    col_step, _ = st.columns([1, 3])
    with col_step:
        alpha_step = st.select_slider("Résolution Δα (°)", options=[0.25, 0.5, 1.0, 2.0], value=0.5)

    alphas = np.arange(alpha_range[0], alpha_range[1] + alpha_step / 2, alpha_step)
    st.caption(f"→ {len(alphas)} points de calcul : α ∈ [{alpha_range[0]:.1f}°, {alpha_range[1]:.1f}°]")

    st.markdown("---")

    # Lancement
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        lancer = st.button("Générer les polaires", type="primary", use_container_width=True)
    with col_info:
        st.markdown(f"<div style='padding-top:0.55rem; color:#666;'>Profil : <b>t={geo['t']:.3f}</b> | <b>m={geo['camber']:.3f}</b> | <b>r_LE={geo['LE_radius']:.4f}</b> — Re = <b>{re_label(re_val)}</b></div>", unsafe_allow_html=True)

    if not lancer:
        st.info("Réglez les paramètres puis cliquez sur **Générer les polaires**.")
        return

    with st.spinner("Inférence en cours..."):
        try:
            df_pred = predire_polaires(model, pre, geo, alphas, float(re_val),
                                       naca_name="", source_name="naca_grid")
        except Exception as exc:
            st.error(f"Erreur lors de la prédiction : {exc}")
            return

    st.success("Polaires générées avec succès !")

    # ── Récupération du nom du profil pour Fluent ──────────────────
    profil_nom = None
    if preset_choisi != "— Personnalisé —":
        profil_nom = preset_choisi.replace(" (Fluent)", "")

    # ── Récupération des données Fluent ────────────────────────────
    df_flu_plot = None
    if profil_nom is not None:
        df_flu_plot = fluent.get((profil_nom, float(re_val)))
        if df_flu_plot is not None:
            st.info(f"✨ Données Ansys Fluent disponibles pour **{profil_nom}** à {re_label(re_val)}.")

    # Métriques
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    cl_max = df_pred["CL"].max()
    alpha_cl = df_pred.loc[df_pred["CL"].idxmax(), "alpha"]
    fin_max = df_pred["finesse"].max()
    alpha_fin = df_pred.loc[df_pred["finesse"].idxmax(), "alpha"]
    cd_min = df_pred["CD"].min()

    m_col1.metric("CL max", f"{cl_max:.3f}", help=f"α = {alpha_cl:.1f}°")
    m_col2.metric("α @ CL max", f"{alpha_cl:.1f}°")
    m_col3.metric("(CL/CD) max", f"{fin_max:.1f}", help=f"α = {alpha_fin:.1f}°")
    m_col4.metric("α @ finesse max", f"{alpha_fin:.1f}°")
    m_col5.metric("CD min", f"{cd_min:.5f}")

    st.markdown("---")
    st.subheader("Polaires aérodynamiques")

    # Tracer avec ML + Fluent (sans XFoil)
    _tracer_polaire_ml_fluent(df_pred, disposition, df_flu_plot)

    with st.expander("Voir les données tabulaires"):
        df_affiche = df_pred.copy()
        df_affiche.columns = ["α (°)", "CL_ML", "CD_ML", "CM_ML", "CL/CD_ML"]

        if df_flu_plot is not None:
            # Fusionner avec Fluent
            df_flu_subset = df_flu_plot[["alpha_deg", "CL", "CD", "CM", "LD"]].copy()
            df_flu_subset.columns = ["α (°)", "CL_Fluent", "CD_Fluent", "CM_Fluent", "CL/CD_Fluent"]
            df_merge_display = pd.merge(df_affiche, df_flu_subset, on="α (°)", how="inner")
            st.dataframe(df_merge_display.style.format({
                "α (°)": "{:.2f}", "CL_ML": "{:.5f}", "CD_ML": "{:.6f}", "CM_ML": "{:.5f}",
                "CL/CD_ML": "{:.2f}", "CL_Fluent": "{:.5f}", "CD_Fluent": "{:.6f}",
                "CM_Fluent": "{:.5f}", "CL/CD_Fluent": "{:.2f}",
            }), use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_affiche.style.format({
                "α (°)": "{:.2f}", "CL_ML": "{:.5f}", "CD_ML": "{:.6f}",
                "CM_ML": "{:.5f}", "CL/CD_ML": "{:.2f}",
            }), use_container_width=True, hide_index=True)

        csv_bytes = df_affiche.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger les données (CSV)", data=csv_bytes,
                           file_name=f"polaire_ML_Re{int(re_val)}_t{geo['t']:.3f}_m{geo['camber']:.3f}.csv",
                           mime="text/csv")


# ── Point d'entrée ───────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal : construit la structure du dashboard et route vers la page choisie.

    Cette fonction orchestre l'application Streamlit dans son ensemble :

    #. injection du CSS global (thème de la sidebar, masquage de l'en-tête
       natif Streamlit) ;
    #. rendu de la sidebar et récupération de la page/disposition
       choisies via :func:`render_sidebar` ;
    #. cas particulier de la page « Prédiction ML », qui ne nécessite pas
       les fichiers CSV XFoil/ML et est donc traitée en court-circuit ;
    #. pour les autres pages, vérification de l'existence des fichiers
       CSV requis (arrêt de l'exécution via ``st.stop()`` si manquants),
       puis chargement des données et routage explicite vers la fonction
       de page correspondante.

    :returns: Rien ; ``main`` ne fait que déclencher des effets de bord
        Streamlit (rendu de widgets et de graphiques) et peut interrompre
        l'exécution du script via ``st.stop()`` en cas de fichier
        manquant.
    :rtype: None
    """

    st.markdown(f"""
    <style>
        [data-testid="stSidebar"] {{ background-color: {COLORS['sidebar_bg']}; }}
        [data-testid="stSidebar"] * {{ color: {COLORS['sidebar_text']}; }}
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stCheckbox label {{ color: {COLORS['sidebar_text']} !important; }}
        hr {{ margin: 1rem 0; border-color: rgba(255,255,255,0.1); }}
        .stMetric {{ background-color: {COLORS['card_bg']}; border-radius: 8px; padding: 0.5rem; }}
        .stApp header {{ display: none; }}
    </style>
    """, unsafe_allow_html=True)

    page, disposition = render_sidebar()

    # Chargement des données
    df_xfoil = None
    df_ml = None
    df_merge = None
    fluent = charger_fluent()

    # Pour la page Prédiction ML, on n'a pas besoin de XFoil
    if page == "Prédiction ML":
        render_enhanced_header()
        page_prediction_ml(disposition, fluent)
        return

    # Pages nécessitant les CSV
    for fichier in (CSV_XFOIL, CSV_ML):
        if not os.path.exists(fichier):
            st.error(f"Fichier introuvable : `{fichier}`. Exécutez d'abord le pipeline.")
            st.stop()

    df_xfoil, df_ml, df_merge = charger_donnees()
    fluent = charger_fluent()

    render_enhanced_header(df_xfoil, df_ml)

    if page == "Polaires":
        page_polaires(df_merge, fluent, disposition)
    elif page == "Performance ML":
        page_performance(df_merge)
    elif page == "Validation ML":
        page_validation_model(df_merge)
    elif page == "Optimisation":
        page_optimisation(df_ml)
    elif page == "Dataset":
        page_dataset(df_xfoil, disposition)


if __name__ == "__main__":
    main()
