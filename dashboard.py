"""Dashboard interactif AeroPredict — MGA 802.

Visualisation des résultats du pipeline aérodynamique + Machine Learning :
  - Polaires par profil (XFoil vs modèle ML vs Ansys Fluent)
  - Performance globale du modèle ML (R², MAE, dispersion)
  - Validation modèle ML (dCl/dalpha, régression parabolique Cl vs alpha)
  - Optimisation : recherche du profil maximisant la finesse CL/CD
  - Exploration du dataset (familles, convergence, features géométriques)
  - Prédiction ML : génération des polaires pour un profil quelconque

Usage:
    streamlit run dashboard.py
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
    """Retourne le logo ETS encodé en base64 pour injection HTML inline."""
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

#: Chemins des fichiers de données
CSV_XFOIL = "dataset_aeroXfoil.csv"
CSV_ML    = "dataset_aeroXfoil_avec_predictions.csv"

#: Chemins des fichiers modèle ML
MODEL_PATH       = "naca_multitask_model.keras"
PREPROCESSOR_PATH = "preprocessor.pkl"

#: Palette de couleurs scientifique moderne
COLORS = {
    "primary": "#0F172A",      # Bleu nuit profond
    "secondary": "#1E293B",    # Bleu ardoise
    "accent": "#3B82F6",       # Bleu électrique
    "accent_light": "#60A5FA", # Bleu ciel
    "xfoil": "#2563EB",        # Bleu royal (XFoil)
    "ml": "#DC2626",           # Rouge vif (ML)
    "fluent": "#059669",       # Vert émeraude (Fluent)
    "background": "#F8FAFC",   # Fond très clair
    "card_bg": "#FFFFFF",      # Fond des cartes
    "text": "#1E293B",         # Texte principal
    "text_light": "#64748B",   # Texte secondaire
    "success": "#10B981",      # Vert succès
    "warning": "#F59E0B",      # Orange avertissement
    "info": "#3B82F6",         # Bleu info
    "sidebar_bg": "#0F172A",   # Fond sidebar sombre
    "sidebar_text": "#E2E8F0", # Texte sidebar clair
}

#: Ordre exact des features attendues par le StandardScaler
FEATURES_ORDRE = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area"]


# ── Helpers ──────────────────────────────────────────────────────

def re_label(r: float) -> str:
    """Formate un nombre de Reynolds en notation lisible pour les widgets."""
    exp      = int(np.floor(np.log10(r)))
    coef     = r / 10**exp
    exposant = str(exp).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    if abs(coef - 1.0) < 0.05:
        return f"Re = 10{exposant}"
    return f"Re = {coef:.1f}×10{exposant}"


def re_tick(r: float) -> str:
    """Formate un nombre de Reynolds pour les étiquettes d'axe Plotly (texte pur)."""
    exp      = int(np.floor(np.log10(r)))
    coef     = r / 10**exp
    exposant = str(exp).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{coef:.1f}×10{exposant}"


# ── Chargement des données (mis en cache) ────────────────────────

@st.cache_data(show_spinner="Chargement des datasets...")
def charger_donnees():
    """Charge les datasets XFoil et ML, et fusionne sur (profil, alpha, Re)."""
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
    """Découvre et charge les fichiers de validation Ansys Fluent."""
    donnees = {}
    for chemin in glob.glob("*_FLUENT.csv"):
        m = regex.match(r"(.+)_Re(\d+)_FLUENT\.csv", os.path.basename(chemin))
        if m:
            profil, re_val = m.group(1), float(m.group(2))
            donnees[(profil, re_val)] = pd.read_csv(chemin)
    return donnees


@st.cache_resource(show_spinner="Chargement du modèle ML…")
def charger_modele():
    """Charge le modèle Keras et le préprocesseur. Retourne (model, preprocessor) ou (None, None)."""
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

    class NACAAeroPreprocessor:
        pass

    import __main__
    __main__.NACAAeroPreprocessor = NACAAeroPreprocessor

    with open(PREPROCESSOR_PATH, "rb") as f:
        pre = pickle.load(f)

    return model, pre


def _construire_X(X_scaled: np.ndarray, naca_norm: float, source_norm: float, ordre: int) -> np.ndarray:
    """Assemble la matrice 12-features selon l'ordre testé."""
    n = len(X_scaled)
    nc = np.full((n, 1), naca_norm,   dtype=np.float32)
    sc = np.full((n, 1), source_norm, dtype=np.float32)
    if ordre == 0:   # [scaled_10 | naca_norm | source_norm]
        return np.concatenate([X_scaled, nc, sc], axis=1).astype(np.float32)
    elif ordre == 1: # [naca_norm | source_norm | scaled_10]
        return np.concatenate([nc, sc, X_scaled], axis=1).astype(np.float32)
    elif ordre == 2: # [naca_norm | scaled_10 | source_norm]
        return np.concatenate([nc, X_scaled, sc], axis=1).astype(np.float32)
    else:            # [source_norm | scaled_10 | naca_norm]
        return np.concatenate([sc, X_scaled, nc], axis=1).astype(np.float32)


def _decoder_preds(preds_raw, ts: dict) -> dict:
    """Dé-normalise les sorties CL, CD, CM du modèle."""
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


def _valider_physique(results: dict, alpha_test: float) -> bool:
    """Vérifie que les prédictions sont physiquement plausibles."""
    cl = float(np.nanmean(results["CL"]))
    cd = float(np.nanmean(results["CD"]))
    cm = float(np.nanmean(results["CM"]))

    return (
            -0.5 < cl < 2.0
            and 0.001 < cd < 0.04
            and -0.5 < cm < 0.5
            and cl > cd
    )


def predire_polaires(
    model,
    pre,
    geo: dict,
    alphas: np.ndarray,
    re_val: float,
    naca_name: str = "",
    source_name: str = "naca_grid",
) -> pd.DataFrame:
    """Prédit CL, CD, CM pour un vecteur d'angles d'attaque."""
    le = pre.__dict__["label_encoders"]
    fs = pre.__dict__["feature_scaler"]
    ts = pre.__dict__["target_scalers"]

    n_naca = float(len(le["naca"].classes_))
    try:
        naca_idx = float(le["naca"].transform([naca_name])[0])
    except Exception:
        naca_idx = n_naca / 2.0
    naca_norm = naca_idx / n_naca

    try:
        source_idx = float(le["source"].transform([source_name])[0])
    except Exception:
        source_idx = 0.0
    source_norm = source_idx

    rows = [[geo[f] for f in FEATURES_ORDRE] + [alpha, re_val] for alpha in alphas]
    X_raw    = np.array(rows, dtype=np.float64)
    X_scaled = fs.transform(X_raw)

    CACHE_KEY = "_predire_ordre_features"

    if CACHE_KEY not in st.session_state:
        idx_test   = np.argmin(np.abs(alphas - 5.0))
        X_test_sc  = X_scaled[idx_test : idx_test + 1]
        ordre_retenu = None
        for ordre in range(4):
            try:
                X_test = _construire_X(X_test_sc, naca_norm, source_norm, ordre)
                pred_test = model.predict(X_test, verbose=0)
                res_test  = _decoder_preds(pred_test, ts)
                if _valider_physique(res_test, alphas[idx_test]):
                    ordre_retenu = ordre
                    break
            except Exception:
                continue
        if ordre_retenu is None:
            ordre_retenu = 0
        st.session_state[CACHE_KEY] = ordre_retenu

    ordre_final = st.session_state[CACHE_KEY]

    X_full    = _construire_X(X_scaled, naca_norm, source_norm, ordre_final)
    preds_raw = model.predict(X_full, verbose=0)
    results   = _decoder_preds(preds_raw, ts)

    df = pd.DataFrame({"alpha": alphas, **results})
    df["finesse"] = df["CL"] / df["CD"].replace(0, np.nan)
    return df


def calculer_metriques(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule R², MAE et RMSE entre valeurs de référence et prédictions."""
    residus = y_vrai - y_pred
    ss_res  = float(np.sum(residus ** 2))
    ss_tot  = float(np.sum((y_vrai - y_vrai.mean()) ** 2))
    return {
        "R2"  : 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "MAE" : float(np.mean(np.abs(residus))),
        "RMSE": float(np.sqrt(np.mean(residus ** 2))),
    }


# ── En-tête simplifié du dashboard (sans logos) ───────────────────────────

def render_enhanced_header(df_xfoil: pd.DataFrame = None, df_ml: pd.DataFrame = None) -> None:


    # Bannière supérieure avec gradient - sans logo
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

    # Cartes de métriques (si données disponibles)
    if df_xfoil is not None and df_ml is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style="
                background: {COLORS['card_bg']};
                border-radius: 12px;
                padding: 1rem;
                border-left: 4px solid {COLORS['accent']};
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.75rem; color: {COLORS['text_light']}; text-transform: uppercase;">
                    Profils analysés
                </div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['primary']};">
                    {df_xfoil['naca'].nunique():,}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="
                background: {COLORS['card_bg']};
                border-radius: 12px;
                padding: 1rem;
                border-left: 4px solid {COLORS['success']};
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.75rem; color: {COLORS['text_light']}; text-transform: uppercase;">
                    Points de données
                </div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['primary']};">
                    {len(df_xfoil):,}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            re_vals = sorted(df_xfoil["Re"].unique())
            st.markdown(f"""
            <div style="
                background: {COLORS['card_bg']};
                border-radius: 12px;
                padding: 1rem;
                border-left: 4px solid {COLORS['ml']};
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.75rem; color: {COLORS['text_light']}; text-transform: uppercase;">
                    Reynolds disponibles
                </div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['primary']};">
                    {len(re_vals)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            conv_rate = df_xfoil["converged"].mean() * 100
            st.markdown(f"""
            <div style="
                background: {COLORS['card_bg']};
                border-radius: 12px;
                padding: 1rem;
                border-left: 4px solid {COLORS['warning'] if conv_rate > 80 else COLORS['warning']};
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 0.75rem; color: {COLORS['text_light']}; text-transform: uppercase;">
                    Convergence XFoil
                </div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {COLORS['primary']};">
                    {conv_rate:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_sidebar():
    """Affiche la sidebar stylisée avec fond sombre - seul endroit avec logo."""

    with st.sidebar:
        # Logo et titre dans la sidebar (seul logo du dashboard)
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

        # Navigation avec icônes
        page = st.radio(
            "**Navigation**",
            ["Polaires", "Performance ML", "Validation ML", "Optimisation", "Dataset", "Prédiction ML"],
            format_func=lambda x: x,
        )

        st.markdown("---")

        # Paramètres d'affichage
        st.markdown("**Paramètres**")
        disposition = st.radio(
            "Disposition des graphiques",
            ["2 colonnes", "1 colonne"],
            horizontal=True,
            help="Organisation des graphiques sur la page"
        )

        st.markdown("---")

        # Informations du projet
        st.markdown(f"""
        <div style="
            background: rgba(59,130,246,0.1);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 1rem;
            border: 1px solid rgba(59,130,246,0.2);
        ">
            <p style="font-size: 0.7rem; color: #94A3B8; margin: 0; text-align: center;">
                <strong style="color: #60A5FA;">MGA 802</strong><br>
                Blanchard / Mechref / Condette<br>
                XFoil • Réseau de neurones • Fluent
            </p>
        </div>
        """, unsafe_allow_html=True)

    return page, disposition


# ── Pages du dashboard ──

def page_polaires(df_merge: pd.DataFrame, fluent: dict, disposition: str) -> None:
    """Page 1 : polaires aérodynamiques d'un profil (XFoil vs ML vs Fluent)."""
    st.header("Polaires aérodynamiques")
    st.markdown("Comparaison des coefficients aérodynamiques entre XFoil, le modèle ML et Ansys Fluent.")

    col_a, col_b = st.columns(2)
    profils = sorted(df_merge["naca"].unique())
    profils_fluent = sorted({p for p, _ in fluent})

    with col_a:
        defaut = profils.index("naca0106") if "naca0106" in profils else 0
        profil = st.selectbox(
            "Sélectionner un profil", profils, index=defaut,
            help=f"Profils avec validation Fluent : {', '.join(profils_fluent)}",
        )
    with col_b:
        re_dispo = sorted(df_merge[df_merge["naca"] == profil]["Re"].unique())
        re_val = st.select_slider(
            "Nombre de Reynolds", re_dispo,
            format_func=re_label,
        )

    sous = (df_merge[(df_merge["naca"] == profil) & (df_merge["Re"] == re_val)]
            .sort_values("alpha"))
    df_flu = fluent.get((profil, re_val))

    if df_flu is not None:
        st.success(
            f"Données Ansys Fluent disponibles pour **{profil}** "
            f"à {re_label(re_val)}"
        )

    conv = sous["converged"].mean() * 100
    st.caption(
        f"{len(sous)} points · convergence XFoil : {conv:.0f} % "
        "(points non convergés interpolés)"
    )
    st.markdown("---")

    courbes = [
        ("CL", "Coefficient de portance", "c<sub>l</sub>"),
        ("CD", "Coefficient de traînée", "c<sub>d</sub>"),
        ("CM", "Coefficient de moment", "c<sub>m</sub>"),
    ]

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
                   showline=True, linewidth=2, linecolor="black", mirror=True,
                   ticks="outside", tickwidth=2, ticklen=8),
        yaxis=dict(title=dict(text="<b>c<sub>l</sub> / c<sub>d</sub></b>", font=dict(size=12)),
                   showgrid=True, gridwidth=1, gridcolor="lightgray",
                   showline=True, linewidth=2, linecolor="black", mirror=True,
                   ticks="outside", tickwidth=2, ticklen=8),
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
                st.info("Aucune donnée Fluent disponible pour ce profil/Re")
        with tab3:
            if df_flu is not None:
                df_merged = sous[["alpha", "CL_xfoil", "CL_ml", "CD_xfoil", "CD_ml"]].copy()
                df_flu_subset = df_flu[["alpha_deg", "CL", "CD", "LD"]].copy()
                df_flu_subset.columns = ["alpha", "CL_Fluent", "CD_Fluent", "Finesse_Fluent"]
                df_comparison = pd.merge(df_merged, df_flu_subset, on="alpha", how="inner")
                st.dataframe(df_comparison.round(4), use_container_width=True, hide_index=True)
            else:
                st.info("Données Fluent non disponibles pour la comparaison")


def page_performance(df_merge: pd.DataFrame) -> None:
    """Page 2 : performance globale du modèle ML face à XFoil."""
    st.header("Performance du modèle ML")
    st.markdown("Évaluation quantitative des prédictions du réseau de neurones multi-tâches.")

    seulement_convergees = st.checkbox(
        "Évaluer uniquement sur les points XFoil convergés", value=True,
        help="Les points non convergés sont des interpolations, pas des références physiques fiables.",
    )
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
            st.markdown(f"<h3 style='margin-bottom:0; color:{COLORS['primary']};'>{labels_coef[coef]}</h3>", unsafe_allow_html=True)
            st.metric("R²", f"{m['R2']:.4f}")
            st.metric("MAE", f"{m['MAE']:.5f}")
            st.metric("RMSE", f"{m['RMSE']:.5f}")

    st.divider()

    n_points = st.slider("Points affichés pour la dispersion (échantillon aléatoire)", 1_000, 50_000, 15_000, step=1_000)
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
    """Page 2bis : Validation modèle ML - sans profils par défaut."""
    st.header("Validation du modèle ML")
    st.markdown("Analyse approfondie du comportement du modèle : dérivée de portance et diagramme polaire CD(CL) avant décrochage.")

    col_filtre1, col_filtre2, col_filtre3 = st.columns(3)

    with col_filtre1:
        seulement_convergees = st.checkbox(
            "Points XFoil convergés uniquement", value=True,
            key="validation_converged",
            help="Filtre les points non convergés d'XFoil (interpolations)",
        )

    df_base = df_merge[df_merge["converged"]] if seulement_convergees else df_merge

    with col_filtre2:
        profils_dispo = sorted(df_base["naca"].unique())
        profil_selection = st.multiselect(
            "Profils à analyser",
            profils_dispo,
            default=[],
            help="Sélectionnez un ou plusieurs profils",
        )

    with col_filtre3:
        re_dispo = sorted(df_base["Re"].unique())
        re_selection = st.select_slider(
            "Nombre de Reynolds",
            options=re_dispo,
            format_func=re_label,
        )

    if not profil_selection:
        st.info("Veuillez sélectionner au moins un profil dans la liste ci-dessus pour visualiser les analyses.")
        return

    df_filtre = df_base[
        (df_base["naca"].isin(profil_selection)) &
        (df_base["Re"] == re_selection)
        ].copy()

    if df_filtre.empty:
        st.warning("Aucune donnée pour la sélection choisie.")
        return

    df_filtre = df_filtre.sort_values(["naca", "alpha"])

    st.markdown("---")

    # Graphique 1 : dCl/dalpha
    st.subheader("Dérivée de portance dCL/dα")
    st.markdown("Comparaison de la dérivée numérique de la courbe de portance entre XFoil et le modèle ML.")

    fig_dclda = go.Figure()

    for profil in profil_selection:
        df_profil = df_filtre[df_filtre["naca"] == profil].sort_values("alpha").copy()
        if len(df_profil) < 3:
            continue

        alpha_vals = df_profil["alpha"].values
        cl_xfoil = df_profil["CL_xfoil"].values
        cl_ml = df_profil["CL_ml"].values

        dcl_dalpha_xfoil = np.gradient(cl_xfoil, alpha_vals)
        dcl_dalpha_ml = np.gradient(cl_ml, alpha_vals)

        fig_dclda.add_trace(go.Scatter(
            x=alpha_vals,
            y=dcl_dalpha_xfoil,
            mode="lines+markers",
            name=f"{profil} (XFoil)",
            line=dict(color=COLORS["xfoil"], width=2.5),
            marker=dict(size=6, symbol="circle", color=COLORS["xfoil"]),
            legendgroup=f"{profil}_xfoil",
        ))

        fig_dclda.add_trace(go.Scatter(
            x=alpha_vals,
            y=dcl_dalpha_ml,
            mode="lines+markers",
            name=f"{profil} (ML)",
            line=dict(color=COLORS["ml"], width=2.5, dash="dash"),
            marker=dict(size=6, symbol="square", color=COLORS["ml"]),
            legendgroup=f"{profil}_ml",
        ))

    fig_dclda.update_layout(
        title=dict(
            text=f"<b>dCL/dα — {re_label(re_selection)}</b>",
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            xref="paper",
            font=dict(size=16, color=COLORS["primary"])
        ),
        xaxis=dict(
            title=dict(text="<b>Angle d'attaque α (°)</b>", font=dict(size=12)),
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            showline=True, linewidth=2, linecolor="black", mirror=True,
        ),
        yaxis=dict(
            title=dict(text="<b>dCL/dα (par degré)</b>", font=dict(size=12)),
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            showline=True, linewidth=2, linecolor="black", mirror=True,
        ),
        height=500,
        margin=dict(t=80, b=50, l=60, r=30),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
        ),
        plot_bgcolor="white", paper_bgcolor=COLORS["background"],
        hovermode="x unified",
    )
    st.plotly_chart(fig_dclda, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # Graphique 2 : CD vs CL
    st.subheader("Diagramme polaire CD(CL) — Avant décrochage")
    st.markdown("Seuls les points **avant le décrochage** (jusqu'au CL maximum) sont affichés. La régression polynomiale d'ordre 4 est calculée sur cette zone.")

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

        idx_max = np.argmax(cl_xfoil)
        mask_avant_stall = np.arange(len(cl_xfoil)) <= idx_max

        cl_avant = cl_xfoil[mask_avant_stall]
        cd_avant = cd_xfoil[mask_avant_stall]
        cl_ml_avant = cl_ml[mask_avant_stall]
        cd_ml_avant = cd_ml[mask_avant_stall]

        if afficher_points_xfoil:
            fig_cd_cl.add_trace(go.Scatter(
                x=cl_avant,
                y=cd_avant,
                mode="markers",
                name=f"{profil} (XFoil points)",
                marker=dict(color=COLORS["xfoil"], size=7, symbol="circle", opacity=0.7),
                legendgroup=f"{profil}_points",
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
                x=cl_fit,
                y=cd_fit,
                mode="lines",
                name=f"{profil} (poly deg4, R²={r2:.4f})",
                line=dict(color=COLORS["accent"], width=2.5),
                legendgroup=f"{profil}_reg",
            ))

        fig_cd_cl.add_trace(go.Scatter(
            x=cl_ml_avant,
            y=cd_ml_avant,
            mode="lines+markers",
            name=f"{profil} (ML)",
            line=dict(color=COLORS["ml"], width=2.5, dash="dash"),
            marker=dict(size=5, symbol="square", color=COLORS["ml"], opacity=0.8),
            legendgroup=f"{profil}_ml",
        ))

    fig_cd_cl.update_layout(
        title=dict(
            text=f"<b>Diagramme polaire CD(CL) — {re_label(re_selection)}</b>",
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            xref="paper",
            font=dict(size=16, color=COLORS["primary"])
        ),
        xaxis=dict(
            title=dict(text="<b>Coefficient de portance C<sub>L</sub></b>", font=dict(size=12)),
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            showline=True, linewidth=2, linecolor="black", mirror=True,
        ),
        yaxis=dict(
            title=dict(text="<b>Coefficient de traînée C<sub>D</sub></b>", font=dict(size=12)),
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            showline=True, linewidth=2, linecolor="black", mirror=True,
        ),
        height=550,
        margin=dict(t=80, b=50, l=60, r=30),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=9),
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
        ),
        plot_bgcolor="white", paper_bgcolor=COLORS["background"],
        hovermode="closest",
    )
    st.plotly_chart(fig_cd_cl, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Coefficients des régressions polynomiales (ordre 4) — avant décrochage"):
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
            st.info("Aucune régression valide calculée (minimum 5 points requis avant décrochage pour l'ordre 4).")

    st.markdown("---")
    st.subheader("Comparaison des métriques par profil")

    metriques_profil = []
    for profil in profil_selection:
        df_profil = df_filtre[df_filtre["naca"] == profil]
        if len(df_profil) > 0:
            m_cl = calculer_metriques(
                df_profil["CL_xfoil"].values,
                df_profil["CL_ml"].values
            )
            m_cd = calculer_metriques(
                df_profil["CD_xfoil"].values,
                df_profil["CD_ml"].values
            )
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
        df_metriques = pd.DataFrame(metriques_profil)
        st.dataframe(df_metriques, use_container_width=True, hide_index=True)


def page_optimisation(df_ml: pd.DataFrame) -> None:
    """Page 3 : recherche du profil maximisant la finesse CL/CD."""
    st.header("Optimisation — Meilleur profil pour vos conditions de vol")
    st.markdown("Objectif du projet : identifier le profil **maximisant le ratio CL/CD** pour des conditions de vol spécifiées.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        re_val = st.select_slider("Nombre de Reynolds", sorted(df_ml["Re"].unique()), format_func=re_label)
    with col_b:
        alpha_min, alpha_max = st.slider("Plage d'angles d'attaque (°)", -6.0, 23.5, (0.0, 10.0), step=0.5)
    with col_c:
        top_n = st.number_input("Nombre de profils affichés", 5, 50, 10)

    sous = df_ml[
        (df_ml["Re"]    == re_val)
        & (df_ml["alpha"] >= alpha_min)
        & (df_ml["alpha"] <= alpha_max)
        & (df_ml["CD"]    >  1e-5)
    ].copy()
    sous["finesse"] = sous["CL"] / sous["CD"]

    idx_max   = sous.groupby("naca")["finesse"].idxmax()
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
    tableau = meilleurs[
        ["naca", "source", "alpha", "CL", "CD", "CM", "finesse", "t", "camber"]
    ].rename(columns={
        "naca"   : "Profil",
        "source" : "Source",
        "alpha"  : "α optimal (°)",
        "finesse": "CL/CD",
        "t"      : "Épaisseur",
        "camber" : "Cambrure",
    })
    st.dataframe(
        tableau.style.format({
            "CL"           : "{:.4f}",
            "CD"           : "{:.5f}",
            "CM"           : "{:.4f}",
            "CL/CD"        : "{:.1f}",
            "α optimal (°)": "{:+.1f}",
            "Épaisseur"    : "{:.3f}",
            "Cambrure"     : "{:.3f}",
        }),
        use_container_width=True, hide_index=True,
    )


def page_dataset(df_xfoil: pd.DataFrame, disposition: str) -> None:
    """Page 4 : exploration du dataset (familles, convergence, features)."""
    st.header("Exploration du dataset")
    st.markdown("Analyse statistique du corpus de profils NACA généré par XFoil.")

    df_profils = df_xfoil.drop_duplicates("naca").copy()
    df_profils["famille"] = df_profils["naca"].str.extract(r"^([a-zA-Z]+)")[0].str.lower()

    # Métriques en colonnes
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
        v = max(0, min(100, v))
        t = v / 100.0

        # Couleurs de la palette Crimson Coast
        colors = [
            (11, 57, 84),  # #0B3954
            (8, 126, 139),  # #087E8B
            (191, 215, 234),  # #BFD7EA
            (255, 90, 95),  # #FF5A5F
            (200, 29, 37)  # #C81D25
        ]

        # Position des couleurs dans le gradient
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]

        for i in range(len(positions) - 1):
            if positions[i] <= t <= positions[i + 1]:
                local_t = (t - positions[i]) / (positions[i + 1] - positions[i])

                r = int(colors[i][0] + local_t * (colors[i + 1][0] - colors[i][0]))
                g = int(colors[i][1] + local_t * (colors[i + 1][1] - colors[i][1]))
                b = int(colors[i][2] + local_t * (colors[i + 1][2] - colors[i][2]))

                return f"#{r:02x}{g:02x}{b:02x}"

        return "#c81d25"

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

    st.subheader("Distribution des features géométriques (1 point par profil)")
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


# Page 5 : Prédiction ML pour un profil quelconque

_GEO_CONFIG = {
    "t"         : ("Épaisseur relative  t/c",         0.02,  0.40,  0.12,   0.005, "Ratio épaisseur max / corde"),
    "camber"    : ("Cambrure max  m/c",               0.00,  0.12,  0.02,   0.002, "Ratio cambrure max / corde"),
    "x_t"       : ("Position épaisseur max  x_t/c",   0.10,  0.70,  0.30,   0.01,  "Position longitudinale de l'épaisseur maximale"),
    "x_c"       : ("Position cambrure max  x_c/c",    0.00,  0.70,  0.40,   0.01,  "Position longitudinale de la cambrure maximale"),
    "LE_radius" : ("Rayon de bord d'attaque  r_LE/c", 0.001, 0.10,  0.016,  0.001, "Rayon de courbure au bord d'attaque normalisé"),
    "TE_angle"  : ("Angle de bord de fuite  β (°)",   0.0,   40.0,  12.0,   0.5,   "Demi-angle d'ouverture au bord de fuite"),
    "t_over_xt" : ("Ratio t / x_t",                   0.05,  1.20,  0.40,   0.01,  "Épaisseur relative divisée par sa position"),
    "area"      : ("Aire de section  A/c²",           0.01,  0.20,  0.077,  0.002, "Surface du profil normalisée"),
}

RE_VALEURS = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]


def _tracer_polaire_prediction(df: pd.DataFrame, disposition: str) -> None:
    """Trace les 4 graphiques de polaires à partir d'un DataFrame prédit - sans étoile de décrochage."""
    couleur = COLORS["accent"]

    spec_courbes = [
        ("CL",      "Coefficient de portance",  "c<sub>l</sub>"),
        ("CD",      "Coefficient de traînée",   "c<sub>d</sub>"),
        ("CM",      "Coefficient de moment",    "c<sub>m</sub>"),
        ("finesse", "Coefficient de finesse",   "c<sub>l</sub> / c<sub>d</sub>"),
    ]

    figs = []
    for col_data, titre_long, titre_court in spec_courbes:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["alpha"], y=df[col_data],
            mode="lines+markers",
            line=dict(color=couleur, width=2.5),
            marker=dict(size=6, symbol="circle", opacity=0.85),
            name="Prédiction ML",
            hovertemplate="α = %{x:.1f}°<br>" + titre_court + " = %{y:.4f}<extra></extra>",
        ))

        # Étoile de décrochage supprimée

        fig.update_layout(
            title=dict(text=f"<b>{titre_long}</b>", x=0.5, y=0.95,
                       xanchor="center", yanchor="top", xref="paper",
                       font=dict(size=16, color=COLORS["primary"])),
            xaxis=dict(
                title=dict(text="<b>Angle d'attaque α (°)</b>", font=dict(size=12)),
                showgrid=True, gridwidth=1, gridcolor="lightgray",
                showline=True, linewidth=2, linecolor="black", mirror=True,
                ticks="outside", tickwidth=2, ticklen=8,
            ),
            yaxis=dict(
                title=dict(text=f"<b>{titre_court}</b>", font=dict(size=12)),
                showgrid=True, gridwidth=1, gridcolor="lightgray",
                showline=True, linewidth=2, linecolor="black", mirror=True,
                ticks="outside", tickwidth=2, ticklen=8,
            ),
            height=430,
            margin=dict(t=80, b=50, l=65, r=30),
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


def page_prediction_ml(disposition: str) -> None:
    """Page 5 : génération des polaires ML pour un profil géométrique quelconque."""
    st.header("Prédiction ML — Polaires pour un profil quelconque")
    st.markdown("Entrez les **caractéristiques géométriques** de votre profil et les **conditions de vol**.")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        st.error(f"Fichiers modèle introuvables. Placez-les dans le même répertoire.")
        return

    model, pre = charger_modele()
    if model is None:
        st.error("Le modèle n'a pas pu être chargé.")
        return

    st.markdown("---")

    # Préréglages
    st.subheader("Préréglage rapide (optionnel)")
    presets = {
        "— Personnalisé —"  : None,
        "NACA 0012 (symétrique)"  : dict(t=0.12,  camber=0.0,  x_t=0.30, x_c=0.0,  LE_radius=0.0158, TE_angle=12.0, t_over_xt=0.400, area=0.0768),
        "NACA 2412 (légère cambrure)": dict(t=0.12, camber=0.02, x_t=0.30, x_c=0.40, LE_radius=0.0158, TE_angle=12.0, t_over_xt=0.400, area=0.0800),
        "NACA 4412 (haute portance)" : dict(t=0.12, camber=0.04, x_t=0.30, x_c=0.40, LE_radius=0.0158, TE_angle=12.0, t_over_xt=0.400, area=0.0832),
        "NACA 0006 (mince symétrique)": dict(t=0.06, camber=0.0,  x_t=0.30, x_c=0.0,  LE_radius=0.0040, TE_angle=6.5,  t_over_xt=0.200, area=0.0390),
        "NACA 6412 (planeur)"        : dict(t=0.12, camber=0.06, x_t=0.30, x_c=0.40, LE_radius=0.0158, TE_angle=12.0, t_over_xt=0.400, area=0.0864),
        "NACA 0018 (épais)"          : dict(t=0.18, camber=0.0,  x_t=0.30, x_c=0.0,  LE_radius=0.0358, TE_angle=17.5, t_over_xt=0.600, area=0.1152),
    }
    preset_choisi = st.selectbox("Partir d'un profil NACA connu", list(presets.keys()))
    valeurs_preset = presets[preset_choisi]

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
            geo[feat] = st.slider(
                label,
                min_value=f_min,
                max_value=f_max,
                value=float(valeur_init),
                step=f_step,
                format=f"%.{decimales}f",
                help=aide,
                key=f"slider_{feat}",
            )

    # Vérifications
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
        re_val = st.select_slider("Nombre de Reynolds", options=RE_VALEURS, value=1_000_000, format_func=re_label)
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
            naca_name_hint = ""
            if preset_choisi != "— Personnalisé —":
                import re as _re
                m = _re.search(r"(\d{4})", preset_choisi)
                if m:
                    naca_name_hint = f"naca{m.group(1)}"
            df_pred = predire_polaires(
                model, pre, geo, alphas, float(re_val),
                naca_name=naca_name_hint,
                source_name="naca_grid",
            )
        except Exception as exc:
            st.error(f"Erreur lors de la prédiction : {exc}")
            return

    st.success("Polaires générées avec succès !")

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    cl_max   = df_pred["CL"].max()
    alpha_cl = df_pred.loc[df_pred["CL"].idxmax(), "alpha"]
    fin_max  = df_pred["finesse"].max()
    alpha_fin = df_pred.loc[df_pred["finesse"].idxmax(), "alpha"]
    cd_min   = df_pred["CD"].min()

    m_col1.metric("CL max", f"{cl_max:.3f}", help=f"α = {alpha_cl:.1f}°")
    m_col2.metric("α @ CL max", f"{alpha_cl:.1f}°")
    m_col3.metric("(CL/CD) max", f"{fin_max:.1f}", help=f"α = {alpha_fin:.1f}°")
    m_col4.metric("α @ finesse max", f"{alpha_fin:.1f}°")
    m_col5.metric("CD min", f"{cd_min:.5f}")

    st.markdown("---")
    st.subheader("Polaires aérodynamiques — Modèle ML")

    _tracer_polaire_prediction(df_pred, disposition)

    with st.expander("Voir les données tabulaires"):
        df_affiche = df_pred.copy()
        df_affiche.columns = ["α (°)", "CL", "CD", "CM", "CL/CD"]
        st.dataframe(
            df_affiche.style.format({
                "α (°)": "{:.2f}",
                "CL"   : "{:.5f}",
                "CD"   : "{:.6f}",
                "CM"   : "{:.5f}",
                "CL/CD": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = df_affiche.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger les données (CSV)",
            data=csv_bytes,
            file_name=f"polaire_ML_Re{int(re_val)}_t{geo['t']:.3f}_m{geo['camber']:.3f}.csv",
            mime="text/csv",
        )


# ── Point d'entrée ───────────────────────────────────────────────

def main() -> None:
    """Construit la structure du dashboard et route vers la page choisie."""

    # Appliquer le style personnalisé pour la sidebar sombre
    st.markdown(f"""
    <style>
        [data-testid="stSidebar"] {{
            background-color: {COLORS['sidebar_bg']};
        }}
        [data-testid="stSidebar"] * {{
            color: {COLORS['sidebar_text']};
        }}
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stCheckbox label {{
            color: {COLORS['sidebar_text']} !important;
        }}
        [data-testid="stSidebar"] .st-emotion-cache-1v0mbdj {{
            background-color: {COLORS['sidebar_bg']};
        }}
        hr {{
            margin: 1rem 0;
            border-color: rgba(255,255,255,0.1);
        }}
        .stMetric {{
            background-color: {COLORS['card_bg']};
            border-radius: 8px;
            padding: 0.5rem;
        }}
        /* Suppression des emojis/icônes dans les titres de page */
        .stApp header {{
            display: none;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Rendu de la sidebar et récupération des paramètres
    page, disposition = render_sidebar()

    # Page Prédiction ML (indépendante des CSV)
    if page == "Prédiction ML":
        render_enhanced_header()
        page_prediction_ml(disposition)
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
