"""Dashboard interactif AeroPredict — MGA 802.

Visualisation des résultats du pipeline aérodynamique + Machine Learning :
  - Polaires par profil (XFoil vs modèle ML vs Ansys Fluent)
  - Performance globale du modèle ML (R², MAE, dispersion)
  - Optimisation : recherche du profil maximisant la finesse CL/CD
  - Exploration du dataset (familles, convergence, features géométriques)

Aucune dépendance à TensorFlow : le dashboard exploite les prédictions
pré-calculées (dataset_aeroXfoil_avec_predictions.csv).

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

# ── Configuration générale ───────────────────────────────────────

st.set_page_config(
    page_title="AeroPredict — Dashboard",
    page_icon="✈️",
    layout="wide",
)

#: Chemins des fichiers de données
CSV_XFOIL = "dataset_projet/dataset_aeroXfoil.csv"
CSV_ML = "dataset_projet/dataset_aeroXfoil_avec_predictions.csv"

#: Couleurs des séries
COULEUR_XFOIL = "#2196F3"
COULEUR_ML = "#F44336"
COULEUR_FLUENT = "#4CAF50"


# ── Chargement des données (mis en cache) ────────────────────────

@st.cache_data(show_spinner="Chargement des datasets...")
def charger_donnees():
    """Charge les datasets XFoil et ML, et fusionne sur (profil, alpha, Re).

    Returns:
        tuple: (df_xfoil, df_ml, df_merge) — datasets bruts et fusionné.
            Le fusionné contient les colonnes CL/CD/CM suffixées _xfoil et _ml.
    """
    df_xfoil = pd.read_csv(CSV_XFOIL)
    df_ml = pd.read_csv(CSV_ML)

    cles = ["naca", "alpha", "Re"]
    df_merge = df_xfoil[cles + ["source", "CL", "CD", "CM", "converged"]].merge(
        df_ml[cles + ["CL", "CD", "CM"]],
        on=cles,
        suffixes=("_xfoil", "_ml"),
    )
    return df_xfoil, df_ml, df_merge


@st.cache_data
def charger_fluent():
    """Découvre et charge les fichiers de validation Ansys Fluent.

    Les fichiers suivent le format <profil>_Re<reynolds>_FLUENT.csv avec
    les colonnes : airfoil, Re, alpha_deg, CL, CD, CM, LD.

    Returns:
        dict: {(profil, reynolds): DataFrame} pour chaque fichier trouvé.
    """
    donnees = {}
    for chemin in glob.glob("*_FLUENT.csv"):
        m = regex.match(r"(.+)_Re(\d+)_FLUENT\.csv", os.path.basename(chemin))
        if m:
            profil, re_val = m.group(1), float(m.group(2))
            donnees[(profil, re_val)] = pd.read_csv(chemin)
    return donnees


def calculer_metriques(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule R², MAE et RMSE entre valeurs de référence et prédictions.

    Args:
        y_vrai: Valeurs de référence (XFoil).
        y_pred: Valeurs prédites (modèle ML).

    Returns:
        Dictionnaire {"R2", "MAE", "RMSE"}.
    """
    residus = y_vrai - y_pred
    ss_res = float(np.sum(residus ** 2))
    ss_tot = float(np.sum((y_vrai - y_vrai.mean()) ** 2))
    return {
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "MAE": float(np.mean(np.abs(residus))),
        "RMSE": float(np.sqrt(np.mean(residus ** 2))),
    }


# ── Pages du dashboard ───────────────────────────────────────────

def page_polaires(df_merge: pd.DataFrame, fluent: dict, disposition: str) -> None:
    """Page 1 : polaires aérodynamiques d'un profil (XFoil vs ML vs Fluent)."""
    st.header("Polaires aérodynamiques")

    col_a, col_b = st.columns(2)
    profils = sorted(df_merge["naca"].unique())
    profils_fluent = sorted({p for p, _ in fluent})

    with col_a:
        defaut = profils.index("naca0106") if "naca0106" in profils else 0
        profil = st.selectbox(
            "Profil", profils, index=defaut,
            help=f"Profils avec validation Fluent : {', '.join(profils_fluent)}",
        )
    with col_b:
        re_dispo = sorted(df_merge[df_merge["naca"] == profil]["Re"].unique())
        re_val = st.select_slider(
            "Nombre de Reynolds", re_dispo,
            format_func=lambda r: f"{r:.0e}",
        )

    sous = (df_merge[(df_merge["naca"] == profil) & (df_merge["Re"] == re_val)]
            .sort_values("alpha"))
    df_flu = fluent.get((profil, re_val))

    if df_flu is not None:
        st.success(f"Données Ansys Fluent disponibles pour {profil} à Re = {re_val:.0e}")

    conv = sous["converged"].mean() * 100
    st.caption(f"{len(sous)} points · convergence XFoil : {conv:.0f} % "
               "(points non convergés interpolés)")

    courbes = [
        ("CL", "Coefficient de portance CL"),
        ("CD", "Coefficient de traînée CD"),
        ("CM", "Coefficient de moment CM"),
    ]

    figs = []
    for coef, titre in courbes:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sous["alpha"], y=sous[f"{coef}_xfoil"],
            name="XFoil", mode="lines", line=dict(color=COULEUR_XFOIL, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=sous["alpha"], y=sous[f"{coef}_ml"],
            name="Modèle ML", mode="lines",
            line=dict(color=COULEUR_ML, width=2, dash="dash"),
        ))
        if df_flu is not None and coef in df_flu.columns:
            fig.add_trace(go.Scatter(
                x=df_flu["alpha_deg"], y=df_flu[coef],
                name="Ansys Fluent", mode="markers",
                marker=dict(color=COULEUR_FLUENT, size=7, symbol="diamond"),
            ))
        fig.update_layout(
            title=titre, xaxis_title="Angle d'attaque α (°)", yaxis_title=coef,
            height=380, margin=dict(t=50, b=40),
            legend=dict(orientation="h", y=1.12),
        )
        figs.append(fig)

    # Finesse L/D recalculée depuis les coefficients de chaque source
    fig_ld = go.Figure()
    fig_ld.add_trace(go.Scatter(
        x=sous["alpha"], y=sous["CL_xfoil"] / sous["CD_xfoil"],
        name="XFoil", mode="lines", line=dict(color=COULEUR_XFOIL, width=2),
    ))
    fig_ld.add_trace(go.Scatter(
        x=sous["alpha"], y=sous["CL_ml"] / sous["CD_ml"],
        name="Modèle ML", mode="lines",
        line=dict(color=COULEUR_ML, width=2, dash="dash"),
    ))
    if df_flu is not None and "LD" in df_flu.columns:
        fig_ld.add_trace(go.Scatter(
            x=df_flu["alpha_deg"], y=df_flu["LD"],
            name="Ansys Fluent", mode="markers",
            marker=dict(color=COULEUR_FLUENT, size=7, symbol="diamond"),
        ))
    fig_ld.update_layout(
        title="Finesse aérodynamique L/D",
        xaxis_title="Angle d'attaque α (°)", yaxis_title="CL / CD",
        height=380, margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    figs.append(fig_ld)

    if disposition == "2 colonnes":
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            cols[0].plotly_chart(figs[i], use_container_width=True)
            if i + 1 < len(figs):
                cols[1].plotly_chart(figs[i + 1], use_container_width=True)
    else:
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True)


def page_performance(df_merge: pd.DataFrame) -> None:
    """Page 2 : performance globale du modèle ML face à XFoil."""
    st.header("Performance du modèle ML")

    seulement_convergees = st.checkbox(
        "Évaluer uniquement sur les points XFoil convergés", value=True,
        help="Les points non convergés sont des interpolations, pas des "
             "références physiques fiables.",
    )
    df_eval = df_merge[df_merge["converged"]] if seulement_convergees else df_merge
    st.caption(f"{len(df_eval):,} points d'évaluation")

    colonnes = st.columns(3)
    for col_st, coef in zip(colonnes, ["CL", "CD", "CM"]):
        m = calculer_metriques(
            df_eval[f"{coef}_xfoil"].values, df_eval[f"{coef}_ml"].values
        )
        with col_st:
            st.subheader(coef)
            st.metric("R²", f"{m['R2']:.4f}")
            st.metric("MAE", f"{m['MAE']:.5f}")
            st.metric("RMSE", f"{m['RMSE']:.5f}")

    st.divider()
    st.subheader("Dispersion prédictions vs XFoil")

    n_points = st.slider("Points affichés (échantillon aléatoire)",
                         1_000, 50_000, 10_000, step=1_000)
    echantillon = df_eval.sample(min(n_points, len(df_eval)), random_state=42)

    colonnes2 = st.columns(3)
    for col_st, coef in zip(colonnes2, ["CL", "CD", "CM"]):
        x = echantillon[f"{coef}_xfoil"]
        y = echantillon[f"{coef}_ml"]
        borne = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(size=3, color=COULEUR_XFOIL, opacity=0.3),
            name=coef, showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=borne, y=borne, mode="lines",
            line=dict(color=COULEUR_ML, dash="dash"), name="y = x",
        ))
        fig.update_layout(
            title=f"{coef} — prédiction vs référence",
            xaxis_title=f"{coef} XFoil", yaxis_title=f"{coef} ML",
            height=400, margin=dict(t=50, b=40), showlegend=False,
        )
        col_st.plotly_chart(fig, use_container_width=True)


def page_optimisation(df_ml: pd.DataFrame) -> None:
    """Page 3 : recherche du profil maximisant la finesse CL/CD."""
    st.header("Optimisation — meilleur profil pour vos conditions de vol")
    st.markdown(
        "Objectif du projet : identifier le profil **maximisant le ratio CL/CD** "
        "pour des conditions de vol spécifiées, à partir des prédictions du modèle ML."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        re_val = st.select_slider(
            "Nombre de Reynolds", sorted(df_ml["Re"].unique()),
            format_func=lambda r: f"{r:.0e}",
        )
    with col_b:
        alpha_min, alpha_max = st.slider(
            "Plage d'angles d'attaque (°)", -6.0, 23.5, (0.0, 10.0), step=0.5
        )
    with col_c:
        top_n = st.number_input("Nombre de profils affichés", 5, 50, 10)

    sous = df_ml[
        (df_ml["Re"] == re_val)
        & (df_ml["alpha"] >= alpha_min)
        & (df_ml["alpha"] <= alpha_max)
        & (df_ml["CD"] > 1e-5)
    ].copy()
    sous["finesse"] = sous["CL"] / sous["CD"]

    # Meilleure finesse atteinte par profil sur la plage demandée
    idx_max = sous.groupby("naca")["finesse"].idxmax()
    meilleurs = (sous.loc[idx_max]
                 .sort_values("finesse", ascending=False)
                 .head(int(top_n)))

    fig = go.Figure(go.Bar(
        x=meilleurs["finesse"][::-1],
        y=meilleurs["naca"][::-1],
        orientation="h",
        marker_color=COULEUR_XFOIL,
        text=[f"α = {a:+.1f}°" for a in meilleurs["alpha"][::-1]],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Top {int(top_n)} des finesses maximales (Re = {re_val:.0e}, "
              f"α ∈ [{alpha_min}°, {alpha_max}°])",
        xaxis_title="Finesse maximale CL/CD",
        yaxis_title="Profil",
        height=max(400, 35 * int(top_n)),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Détails des meilleurs profils")
    tableau = meilleurs[["naca", "source", "alpha", "CL", "CD", "CM", "finesse",
                         "t", "camber"]].rename(columns={
        "naca": "Profil", "source": "Source", "alpha": "α optimal (°)",
        "finesse": "CL/CD", "t": "Épaisseur", "camber": "Cambrure",
    })
    st.dataframe(
        tableau.style.format({
            "CL": "{:.4f}", "CD": "{:.5f}", "CM": "{:.4f}",
            "CL/CD": "{:.1f}", "α optimal (°)": "{:+.1f}",
            "Épaisseur": "{:.3f}", "Cambrure": "{:.3f}",
        }),
        use_container_width=True, hide_index=True,
    )


def page_dataset(df_xfoil: pd.DataFrame, disposition: str) -> None:
    """Page 4 : exploration du dataset (familles, convergence, features)."""
    st.header("Exploration du dataset")

    df_profils = df_xfoil.drop_duplicates("naca").copy()
    df_profils["famille"] = (df_profils["naca"]
                             .str.extract(r"^([a-zA-Z]+)")[0].str.lower())

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Lignes", f"{len(df_xfoil):,}")
    col_b.metric("Profils uniques", f"{df_xfoil['naca'].nunique():,}")
    col_c.metric("Familles", f"{df_profils['famille'].nunique()}")
    col_d.metric("Convergence XFoil", f"{df_xfoil['converged'].mean() * 100:.1f} %")

    # Familles de profils
    familles = (df_profils.groupby("famille")["naca"].count()
                .sort_values(ascending=False))
    fig_fam = go.Figure(go.Bar(
        x=familles.index, y=familles.values, marker_color=COULEUR_XFOIL,
        text=familles.values, textposition="outside",
    ))
    fig_fam.update_layout(
        title="Nombre de profils par famille",
        xaxis_title="Famille", yaxis_title="Profils",
        height=420, margin=dict(t=50, b=40),
    )

    # Convergence par Reynolds
    conv_re = df_xfoil.groupby("Re")["converged"].mean() * 100
    fig_conv = go.Figure(go.Bar(
        x=[f"{r:.0e}" for r in conv_re.index], y=conv_re.values,
        marker_color=COULEUR_FLUENT,
        text=[f"{v:.1f} %" for v in conv_re.values], textposition="outside",
    ))
    fig_conv.update_layout(
        title="Taux de convergence XFoil par nombre de Reynolds",
        xaxis_title="Re", yaxis_title="Convergence (%)",
        yaxis_range=[0, 105], height=420, margin=dict(t=50, b=40),
    )

    if disposition == "2 colonnes":
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_fam, use_container_width=True)
        col2.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.plotly_chart(fig_fam, use_container_width=True)
        st.plotly_chart(fig_conv, use_container_width=True)

    # Distributions des features géométriques
    st.subheader("Distribution des features géométriques (1 point par profil)")
    features = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle",
                "t_over_xt", "area"]
    feature = st.selectbox("Feature", features)
    fig_hist = go.Figure(go.Histogram(
        x=df_profils[feature], nbinsx=50, marker_color=COULEUR_XFOIL,
    ))
    mediane = float(df_profils[feature].median())
    fig_hist.add_vline(x=mediane, line_dash="dash", line_color=COULEUR_ML,
                       annotation_text=f"médiane = {mediane:.3f}")
    fig_hist.update_layout(
        xaxis_title=feature, yaxis_title="Nombre de profils",
        height=420, margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ── Point d'entrée ───────────────────────────────────────────────

def main() -> None:
    """Construit la structure du dashboard et route vers la page choisie."""
    st.title("✈️ AeroPredict — Optimisation de profils assistée par ML")
    st.caption("MGA 802 · Blanchard / Mechref / Condette — XFoil vs réseau de "
               "neurones multi-tâches, validation Ansys Fluent")

    for fichier in (CSV_XFOIL, CSV_ML):
        if not os.path.exists(fichier):
            st.error(f"Fichier introuvable : `{fichier}`. "
                     "Exécutez d'abord le pipeline via `python main.py`.")
            st.stop()

    df_xfoil, df_ml, df_merge = charger_donnees()
    fluent = charger_fluent()

    page = st.sidebar.radio(
        "Navigation",
        ["Polaires", "Performance ML", "Optimisation", "Dataset"],
    )
    st.sidebar.divider()
    disposition = st.sidebar.radio(
        "Disposition des graphiques",
        ["2 colonnes", "1 colonne"],
        horizontal=True,
    )
    st.sidebar.divider()
    st.sidebar.caption(
        f"{len(df_xfoil):,} lignes · {df_xfoil['naca'].nunique():,} profils · "
        f"{len(fluent)} cas Fluent"
    )

    if page == "Polaires":
        page_polaires(df_merge, fluent, disposition)
    elif page == "Performance ML":
        page_performance(df_merge)
    elif page == "Optimisation":
        page_optimisation(df_ml)
    else:
        page_dataset(df_xfoil, disposition)


main()
