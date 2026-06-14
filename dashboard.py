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
CSV_XFOIL = "dataset_aeroXfoil.csv"
CSV_ML    = "dataset_aeroXfoil_avec_predictions.csv"

#: Couleurs des séries
COULEUR_XFOIL  = "#2196F3"
COULEUR_ML     = "#F44336"
COULEUR_FLUENT = "#4CAF50"


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


# ── Pages du dashboard ───────────────────────────────────────────

def page_polaires(df_merge: pd.DataFrame, fluent: dict, disposition: str) -> None:
    """Page 1 : polaires aérodynamiques d'un profil (XFoil vs ML vs Fluent)."""
    st.header("📈 Polaires aérodynamiques")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    profils = sorted(df_merge["naca"].unique())
    profils_fluent = sorted({p for p, _ in fluent})

    with col_a:
        defaut = profils.index("naca0106") if "naca0106" in profils else 0
        profil = st.selectbox(
            "✈️ Sélectionner un profil", profils, index=defaut,
            help=f"Profils avec validation Fluent : {', '.join(profils_fluent)}",
        )
    with col_b:
        re_dispo = sorted(df_merge[df_merge["naca"] == profil]["Re"].unique())
        re_val = st.select_slider(
            "🌊 Nombre de Reynolds", re_dispo,
            format_func=re_label,
        )

    sous = (df_merge[(df_merge["naca"] == profil) & (df_merge["Re"] == re_val)]
            .sort_values("alpha"))
    df_flu = fluent.get((profil, re_val))

    if df_flu is not None:
        st.success(
            f"✅ Données Ansys Fluent disponibles pour **{profil}** "
            f"à {re_label(re_val)}"
        )

    conv = sous["converged"].mean() * 100
    st.caption(
        f"📊 {len(sous)} points · convergence XFoil : {conv:.0f} % "
        "(points non convergés interpolés)"
    )
    st.markdown("---")

    courbes = [
        ("CL", "Coefficient de portance", "c<sub>l</sub>"),
        ("CD", "Coefficient de traînée", "c<sub>d</sub>"),
        ("CM", "Coefficient de moment", "c<sub>m</sub>"),
    ]

    # Configuration des coordonnées des légendes pour chaque coefficient
    positions_legendes = {
        "CL": dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98), # En bas à droite
        "CD": dict(yanchor="top", y=0.98, xanchor="left", x=0.02),    # En haut à gauche
        "CM": dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02),   # En bas à gauche
    }

    figs = []
    for coef, titre_long, titre_court in courbes:
        fig = go.Figure()

        # XFoil
        fig.add_trace(go.Scatter(
            x=sous["alpha"], y=sous[f"{coef}_xfoil"],
            name="XFoil", mode="lines+markers",
            line=dict(color=COULEUR_XFOIL, width=2.5),
            marker=dict(size=6, symbol="circle", opacity=0.8),
            legendgroup="XFoil",
        ))

        # Modèle ML
        fig.add_trace(go.Scatter(
            x=sous["alpha"], y=sous[f"{coef}_ml"],
            name="Modèle ML", mode="lines+markers",
            line=dict(color=COULEUR_ML, width=2.5, dash="dash"),
            marker=dict(size=6, symbol="square", opacity=0.8),
            legendgroup="ML",
        ))

        # Ansys Fluent si disponible
        if df_flu is not None and coef in df_flu.columns:
            fig.add_trace(go.Scatter(
                x=df_flu["alpha_deg"], y=df_flu[coef],
                name="Ansys Fluent", mode="markers",
                marker=dict(color=COULEUR_FLUENT, size=10, symbol="diamond",
                            line=dict(width=1, color="white")),
                legendgroup="Fluent",
            ))

        # Configuration de la légende spécifique à ce coefficient
        legende_config = dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
            **positions_legendes[coef]
        )

        # Mise en forme et centrage absolu du titre (xref="paper")
        fig.update_layout(
            title=dict(
                text=f"<b>{titre_long}</b>",
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
                xref="paper",
                font=dict(size=16),
            ),
            xaxis=dict(
                title=dict(text="<b>Angle d'attaque α (°)</b>", font=dict(size=13)),
                showgrid=True, gridwidth=1, gridcolor="lightgray",
                showline=True, linewidth=2, linecolor="black", mirror=True,
                ticks="outside", tickwidth=2, ticklen=8,
            ),
            yaxis=dict(
                title=dict(text=f"<b>{titre_court}</b>", font=dict(size=13)),
                showgrid=True, gridwidth=1, gridcolor="lightgray",
                showline=True, linewidth=2, linecolor="black", mirror=True,
                ticks="outside", tickwidth=2, ticklen=8,
            ),
            height=450,
            margin=dict(t=80, b=50, l=60, r=30),
            legend=legende_config,
            plot_bgcolor="white",
            hovermode="x unified",
        )
        figs.append(fig)

    # ── Coefficient de finesse ──────────────────────────────────────
    fig_ld = go.Figure()
    fig_ld.add_trace(go.Scatter(
        x=sous["alpha"], y=sous["CL_xfoil"] / sous["CD_xfoil"],
        name="XFoil", mode="lines+markers",
        line=dict(color=COULEUR_XFOIL, width=2.5),
        marker=dict(size=6, symbol="circle", opacity=0.8),
    ))
    fig_ld.add_trace(go.Scatter(
        x=sous["alpha"], y=sous["CL_ml"] / sous["CD_ml"],
        name="Modèle ML", mode="lines+markers",
        line=dict(color=COULEUR_ML, width=2.5, dash="dash"),
        marker=dict(size=6, symbol="square", opacity=0.8),
    ))
    if df_flu is not None and "LD" in df_flu.columns:
        fig_ld.add_trace(go.Scatter(
            x=df_flu["alpha_deg"], y=df_flu["LD"],
            name="Ansys Fluent", mode="markers",
            marker=dict(color=COULEUR_FLUENT, size=10, symbol="diamond",
                        line=dict(width=1, color="white")),
        ))

    fig_ld.update_layout(
        title=dict(
            text="<b>Coefficient de finesse</b>",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
            xref="paper",
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text="<b>Angle d'attaque α (°)</b>", font=dict(size=13)),
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            showline=True, linewidth=2, linecolor="black", mirror=True,
            ticks="outside", tickwidth=2, ticklen=8,
        ),
        yaxis=dict(
            title=dict(text="<b>c<sub>l</sub> / c<sub>d</sub></b>", font=dict(size=13)),
            showgrid=True, gridwidth=1, gridcolor="lightgray",
            showline=True, linewidth=2, linecolor="black", mirror=True,
            ticks="outside", tickwidth=2, ticklen=8,
        ),
        height=450,
        margin=dict(t=80, b=50, l=60, r=30),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
            yanchor="bottom", y=0.02, xanchor="right", x=0.98, # En bas à droite
        ),
        plot_bgcolor="white",
        hovermode="x unified",
    )
    figs.append(fig_ld)

    # Affichage en grille ou en liste
    if disposition == "2 colonnes":
        st.subheader("Coefficients aérodynamiques")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figs[0], use_container_width=True, config={"displayModeBar": False})  # CL
            st.plotly_chart(figs[2], use_container_width=True, config={"displayModeBar": False})  # CM
        with col2:
            st.plotly_chart(figs[1], use_container_width=True, config={"displayModeBar": False})  # CD
            st.plotly_chart(figs[3], use_container_width=True, config={"displayModeBar": False})  # Finesse
    else:
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Données détaillées
    with st.expander("📋 Voir les données détaillées"):
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

    seulement_convergees = st.checkbox(
        "Évaluer uniquement sur les points XFoil convergés", value=True,
        help="Les points non convergés sont des interpolations, pas des références physiques fiables.",
    )
    df_eval = df_merge[df_merge["converged"]] if seulement_convergees else df_merge
    st.caption(f"{len(df_eval):,} points d'évaluation")

    labels_coef = {
        "CL": "c<sub>l</sub>",
        "CD": "c<sub>d</sub>",
        "CM": "c<sub>m</sub>",
    }
    colonnes = st.columns(3)
    for col_st, coef in zip(colonnes, ["CL", "CD", "CM"]):
        m = calculer_metriques(df_eval[f"{coef}_xfoil"].values, df_eval[f"{coef}_ml"].values)
        with col_st:
            st.markdown(f"<h3 style='margin-bottom:0'>{labels_coef[coef]}</h3>", unsafe_allow_html=True)
            st.metric("R²",   f"{m['R2']:.4f}")
            st.metric("MAE",  f"{m['MAE']:.5f}")
            st.metric("RMSE", f"{m['RMSE']:.5f}")

    st.divider()
    st.subheader("Dispersion prédictions vs XFoil")

    n_points = st.slider("Points affichés (échantillon aléatoire)", 1_000, 50_000, 10_000, step=1_000)
    echantillon = df_eval.sample(min(n_points, len(df_eval)), random_state=42)

    colonnes2 = st.columns(3)
    for col_st, coef in zip(colonnes2, ["CL", "CD", "CM"]):
        x     = echantillon[f"{coef}_xfoil"]
        y     = echantillon[f"{coef}_ml"]
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
            xaxis_title=f"{coef} XFoil",
            yaxis_title=f"{coef} ML",
            height=400,
            margin=dict(t=50, b=40),
            showlegend=False,
        )
        col_st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def page_optimisation(df_ml: pd.DataFrame) -> None:
    """Page 3 : recherche du profil maximisant la finesse CL/CD."""
    st.header("Optimisation — meilleur profil pour vos conditions de vol")
    st.markdown(
        "Objectif du projet : identifier le profil **maximisant le ratio CL/CD** "
        "pour des conditions de vol spécifiées, à partir des prédictions du modèle ML."
    )

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
        marker_color=COULEUR_XFOIL,
        text=[f"α = {a:+.1f}°" for a in meilleurs["alpha"][::-1]],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Top {int(top_n)} des finesses maximales — {re_label(re_val)}, α ∈ [{alpha_min}°, {alpha_max}°]",
        xaxis_title="Finesse maximale c<sub>l</sub>/c<sub>d</sub>",
        yaxis_title="Profil",
        height=max(400, 35 * int(top_n)),
        margin=dict(t=50, b=40),
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

    df_profils = df_xfoil.drop_duplicates("naca").copy()
    df_profils["famille"] = df_profils["naca"].str.extract(r"^([a-zA-Z]+)")[0].str.lower()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Lignes", f"{len(df_xfoil):,}")
    col_b.metric("Profils uniques", f"{df_xfoil['naca'].nunique():,}")
    col_c.metric("Familles", f"{df_profils['famille'].nunique()}")
    col_d.metric("Convergence XFoil", f"{df_xfoil['converged'].mean() * 100:.1f} %")

    # Familles de profils
    familles = df_profils.groupby("famille")["naca"].count().sort_values(ascending=False)
    fig_fam = go.Figure(go.Bar(
        x=familles.index, y=familles.values,
        marker_color=COULEUR_XFOIL, text=familles.values, textposition="outside",
    ))
    fig_fam.update_layout(title="Nombre de profils par famille", xaxis_title="Famille", yaxis_title="Profils", height=420, margin=dict(t=50, b=40))

    # Convergence par Reynolds
    conv_re = df_xfoil.groupby("Re")["converged"].mean() * 100
    re_labels = [re_tick(r) for r in conv_re.index]

    def _couleur_conv_gradient(v: float) -> str:
        v_norm = max(0, min(100, v)) / 100.0
        if v_norm <= 0.5:
            t = v_norm / 0.5
            r, g, b = int(229 + t * (251 - 229)), int(57 + t * (140 - 57)), int(53 + t * (0 - 53))
        else:
            t = (v_norm - 0.5) / 0.5
            r, g, b = int(251 + t * (94 - 251)), int(140 + t * (53 - 140)), int(0 + t * (177 - 0))
        return f"#{r:02x}{g:02x}{b:02x}"

    couleurs_conv = [_couleur_conv_gradient(v) for v in conv_re.values]

    fig_conv = go.Figure(go.Bar(
        x=re_labels, y=conv_re.values, marker_color=couleurs_conv,
        text=[f"{v:.1f} %" for v in conv_re.values], textposition="outside",
    ))
    fig_conv.update_layout(title="Taux de convergence XFoil par nombre de Reynolds", xaxis=dict(title="Re", type="category", tickangle=-30), yaxis=dict(title="Convergence (%)", range=[0, 105]), height=420, margin=dict(t=50, b=60))

    if disposition == "2 colonnes":
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_fam, use_container_width=True, config={"displayModeBar": False})
        col2.plotly_chart(fig_conv, use_container_width=True, config={"displayModeBar": False})
    else:
        st.plotly_chart(fig_fam, use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(fig_conv, use_container_width=True, config={"displayModeBar": False})

    # Distributions features géométriques
    st.subheader("Distribution des features géométriques (1 point par profil)")
    features = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area"]
    feature = st.selectbox("Feature", features)

    data = df_profils[feature].dropna().values
    from scipy import stats
    kde = stats.gaussian_kde(data)

    nbins = 50
    hist_counts, bin_edges = np.histogram(data, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    x_dense = np.linspace(data.min(), data.max(), 200)
    y_dense = kde(x_dense)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(x=bin_centers, y=hist_counts, name="Histogramme", marker_color=COULEUR_XFOIL, opacity=0.4, width=bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.01))
    fig_hist.add_trace(go.Scatter(x=x_dense, y=y_dense, name="Densité", mode="lines", line=dict(color=COULEUR_ML, width=3)))

    mediane = float(np.median(data))
    fig_hist.add_vline(x=mediane, line_dash="dash", line_color="#5E35B1", line_width=2, annotation_text=f"médiane = {mediane:.3f}", annotation_position="top")
    fig_hist.update_layout(title=f"Distribution de {feature}", xaxis_title=feature, yaxis_title="Densité", height=500, margin=dict(t=50, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


# ── Point d'entrée ───────────────────────────────────────────────

def main() -> None:
    """Construit la structure du dashboard et route vers la page choisie."""
    st.title("✈️ AeroPredict — Optimisation de profils assistée par ML")
    st.caption("MGA 802 · Blanchard / Mechref / Condette — XFoil vs réseau de neurones multi-tâches, validation Ansys Fluent")

    for fichier in (CSV_XFOIL, CSV_ML):
        if not os.path.exists(fichier):
            st.error(f"Fichier introuvable : `{fichier}`. Exécutez d'abord le pipeline.")
            st.stop()

    df_xfoil, df_ml, df_merge = charger_donnees()
    fluent                     = charger_fluent()

    page = st.sidebar.radio("Navigation", ["Polaires", "Performance ML", "Optimisation", "Dataset"])
    st.sidebar.divider()
    disposition = st.sidebar.radio("Disposition des graphiques", ["2 colonnes", "1 colonne"], horizontal=True)
    st.sidebar.divider()
    st.sidebar.caption(f"{len(df_xfoil):,} lignes · {df_xfoil['naca'].nunique():,} profils · {len(fluent)} cas Fluent")

    if page == "Polaires":
        page_polaires(df_merge, fluent, disposition)
    elif page == "Performance ML":
        page_performance(df_merge)
    elif page == "Optimisation":
        page_optimisation(df_ml)
    else:
        page_dataset(df_xfoil, disposition)


if __name__ == "__main__":
    main()
