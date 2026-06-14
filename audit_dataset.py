"""
audit_dataset.py — Analyse et visualisation du dataset AeroPredict
===================================================================
Produit des rapports PDF individuels pour chaque analyse :
  - Distribution par famille de profils
  - Distribution des features géométriques

Usage :
    python audit_dataset.py
    python audit_dataset.py --input mon_dataset.csv --output_dir rapport_pdf/
"""

import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import argparse

# ── Palette et style ────────────────────────────────────────────
BG       = "#0F1117"
PANEL    = "#1A1D27"
BORDER   = "#2A2D3A"
TEXT     = "#DDDDEE"
SUBTEXT  = "#AAAAAA"
PALETTE  = ["#2196F3","#F44336","#4CAF50","#FF9800","#9C27B0",
            "#00BCD4","#FFEB3B","#E91E63","#8BC34A","#FF5722"]

sns.set_theme(style="dark", palette=PALETTE)

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : PANEL,
    "axes.edgecolor"    : BORDER,
    "axes.labelcolor"   : SUBTEXT,
    "axes.titlecolor"   : TEXT,
    "axes.titlesize"    : 11,
    "axes.labelsize"    : 9,
    "xtick.color"       : SUBTEXT,
    "ytick.color"       : SUBTEXT,
    "xtick.labelsize"   : 8,
    "ytick.labelsize"   : 8,
    "grid.color"        : BORDER,
    "grid.linewidth"    : 0.5,
    "text.color"        : TEXT,
    "legend.facecolor"  : PANEL,
    "legend.edgecolor"  : BORDER,
    "legend.fontsize"   : 8,
    "font.size"         : 9,
})

TITLE_KW = dict(color=TEXT, fontsize=12, fontweight="bold", y=0.96)
SUB_KW   = dict(fontsize=10, fontweight="bold", pad=6)
LAB_KW   = dict(color=SUBTEXT, fontsize=9)


# ══════════════════════════════════════════════════════════════
# CHARGEMENT ET PRÉPARATION
# ══════════════════════════════════════════════════════════════

def load(filepath):
    df = pd.read_csv(filepath)
    # Famille de profil : lettres initiales du nom
    df["famille"] = df["naca"].str.extract(r"^([a-zA-Z]+)")[0].str.lower()
    return df


def clean_output_dir(output_dir):
    """Nettoie les anciens fichiers PDF du dossier sans supprimer le dossier lui-même"""
    if os.path.exists(output_dir):
        print(f"  Nettoyage des anciens PDF dans: {output_dir}")
        # Supprimer uniquement les fichiers PDF existants
        for f in glob.glob(os.path.join(output_dir, "*.pdf")):
            try:
                os.remove(f)
                print(f"    Supprime: {os.path.basename(f)}")
            except PermissionError:
                print(f"    Attention: Impossible de supprimer {os.path.basename(f)} (fichier ouvert?)")
    else:
        os.makedirs(output_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# RAPPORT 1 — FAMILLES DE PROFILS
# ══════════════════════════════════════════════════════════════

def report_familles(output_dir, df):
    """Génère un PDF avec les graphiques des familles de profils"""
    filepath = os.path.join(output_dir, "01_familles_profils.pdf")
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.88, bottom=0.08)

    fig.suptitle("Distribution par famille de profils", **TITLE_KW)

    # ── 1a. Barplot lignes par famille ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    fam_counts = (df.groupby("famille")["naca"]
                    .nunique()
                    .sort_values(ascending=False))
    pal = [PALETTE[i % len(PALETTE)] for i in range(len(fam_counts))]
    bars = ax1.bar(range(len(fam_counts)), fam_counts.values,
                   color=pal[:len(fam_counts)], edgecolor=BORDER)
    ax1.set_xticks(range(len(fam_counts)))
    ax1.set_xticklabels(fam_counts.index, rotation=45, ha='right')
    ax1.set_title("Nombre de profils par famille", **SUB_KW)
    ax1.set_xlabel("Famille", **LAB_KW)
    ax1.set_ylabel("Nombre de profils", **LAB_KW)
    for bar, val in zip(bars, fam_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha="center", color=TEXT, fontsize=7.5)
    ax1.set_facecolor(PANEL)

    # ── 1b. Top 10 familles — cambrure moyenne ───────────────
    ax2 = fig.add_subplot(gs[1, 0])
    top10 = fam_counts.head(10).index
    fam_camber = (df[df["famille"].isin(top10)]
                    .groupby("famille")["camber"]
                    .mean()
                    .reindex(top10))
    pal2 = [PALETTE[i % len(PALETTE)] for i in range(len(fam_camber))]
    bars2 = ax2.bar(range(len(fam_camber)), fam_camber.values,
                    color=pal2[:len(fam_camber)], edgecolor=BORDER)
    ax2.set_xticks(range(len(fam_camber)))
    ax2.set_xticklabels(fam_camber.index, rotation=45, ha='right')
    ax2.set_title("Cambrure moyenne — top 10 familles", **SUB_KW)
    ax2.set_xlabel("Famille", **LAB_KW)
    ax2.set_ylabel("camber moyen", **LAB_KW)
    ax2.set_facecolor(PANEL)

    # ── 1c. Top 10 familles — épaisseur moyenne ──────────────
    ax3 = fig.add_subplot(gs[1, 1])
    fam_thick = (df[df["famille"].isin(top10)]
                   .groupby("famille")["t"]
                   .mean()
                   .reindex(top10))
    pal3 = [PALETTE[i % len(PALETTE)] for i in range(len(fam_thick))]
    bars3 = ax3.bar(range(len(fam_thick)), fam_thick.values,
                    color=pal3[:len(fam_thick)], edgecolor=BORDER)
    ax3.set_xticks(range(len(fam_thick)))
    ax3.set_xticklabels(fam_thick.index, rotation=45, ha='right')
    ax3.set_title("Épaisseur moyenne (t) — top 10 familles", **SUB_KW)
    ax3.set_xlabel("Famille", **LAB_KW)
    ax3.set_ylabel("t moyen", **LAB_KW)
    ax3.set_facecolor(PANEL)

    with PdfPages(filepath) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Genere: {filepath}")


# ══════════════════════════════════════════════════════════════
# RAPPORT 2 — DISTRIBUTION DES FEATURES GÉOMÉTRIQUES
# ══════════════════════════════════════════════════════════════

def report_features(output_dir, df):
    """Génère un PDF avec la distribution des 8 features géométriques"""
    filepath = os.path.join(output_dir, "02_features_geometriques.pdf")
    features = ["t", "camber", "x_t", "x_c", "LE_radius", "TE_angle", "t_over_xt", "area"]
    labels   = ["Epaisseur (t)", "Cambrure", "Pos. epaisseur (x_t)",
                "Pos. cambrure (x_c)", "Rayon LE", "Angle TE (deg)",
                "t / x_t", "Aire section"]

    fig = plt.figure(figsize=(11.7, 8.3))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.5, wspace=0.4,
                           left=0.06, right=0.97,
                           top=0.88, bottom=0.08)

    fig.suptitle("Distribution des 8 features geometriques", **TITLE_KW)

    for i, (feat, lab) in enumerate(zip(features, labels)):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        sns.histplot(data=df.drop_duplicates("naca"), x=feat,
                     bins=40, ax=ax, color=PALETTE[i % len(PALETTE)],
                     edgecolor=BORDER, kde=True)
        ax.set_title(lab, **SUB_KW)
        ax.set_xlabel("")
        ax.set_ylabel("Profils", **LAB_KW)
        ax.set_facecolor(PANEL)

        # Ligne mediane
        med = df.drop_duplicates("naca")[feat].median()
        ax.axvline(med, color=PALETTE[1], lw=1, linestyle="--", alpha=0.7)
        ax.text(0.97, 0.95, f"med={med:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=SUBTEXT)

    with PdfPages(filepath) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Genere: {filepath}")


# ══════════════════════════════════════════════════════════════
# RAPPORT 3 — SCATTER PLOTS GÉOMÉTRIQUES
# ══════════════════════════════════════════════════════════════

def report_scatter(output_dir, df):
    """Génère un PDF avec les scatter plots entre features"""
    filepath = os.path.join(output_dir, "03_scatter_plots.pdf")
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.88, bottom=0.08)

    fig.suptitle("Relations entre features geometriques - par source", **TITLE_KW)

    df_uniq = df.drop_duplicates("naca")
    palette = {"naca_grid": PALETTE[0], "uiuc": PALETTE[2]}
    pairs = [
        ("t", "camber", "Epaisseur vs Cambrure"),
        ("t", "area", "Epaisseur vs Aire"),
        ("camber", "x_c", "Cambrure vs Pos. cambrure"),
        ("LE_radius", "t", "Rayon LE vs Epaisseur"),
        ("TE_angle", "t", "Angle TE vs Epaisseur"),
        ("x_t", "t_over_xt", "Pos. epaisseur vs t/x_t"),
    ]

    for i, (x, y, title) in enumerate(pairs):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        sns.scatterplot(data=df_uniq, x=x, y=y,
                        hue="source", palette=palette,
                        ax=ax, alpha=0.6, s=15, edgecolor="none")
        ax.set_title(title, **SUB_KW)
        ax.set_xlabel(x, **LAB_KW)
        ax.set_ylabel(y, **LAB_KW)
        ax.set_facecolor(PANEL)
        if i != 0:
            ax.get_legend().remove()

    with PdfPages(filepath) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Genere: {filepath}")


# ══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

def audit_dataset(filepath="dataset.csv", output_dir="audit_reports"):

    print(f"Chargement : {filepath}")
    df = load(filepath)

    print(f"  {len(df):,} lignes  ·  {df['naca'].nunique()} profils  ·  {df['famille'].nunique()} familles")

    # ── Audit console ────────────────────────────────────────
    print(f"\n-- Audit qualite --")
    print(f"  Doublons          : {df.duplicated().sum()}")
    print(f"  Valeurs manquantes: {df.isnull().sum().sum()}")
    counts = df.groupby("naca")["alpha"].count()
    print(f"  Profils complets  : {(counts == 300).sum()} / {len(counts)}")
    print(f"  Profils incomplets: {(counts != 300).sum()}")

    # ── Creation du repertoire de sortie (sans erreur de permission) ────
    clean_output_dir(output_dir)
    print(f"\nGeneration des PDF dans : {output_dir}")

    # ── Generation des rapports ─────────────────────────────
    report_familles(output_dir, df)
    report_features(output_dir, df)
    report_scatter(output_dir, df)

    # Afficher les fichiers generes
    fichiers = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
    print(f"\nTermine. {len(fichiers)} PDFs generes dans '{output_dir}':")
    for f in sorted(fichiers):
        taille = os.path.getsize(os.path.join(output_dir, f))
        print(f"  - {f} ({taille:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset.csv", help="Fichier CSV")
    parser.add_argument("--output_dir", default="audit_reports", help="Dossier de sortie pour les PDFs")
    args = parser.parse_args()
    audit_dataset(args.input, args.output_dir)
