import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def audit_dataset(filepath="dataset.csv"):
    print(f"--- Début de l'audit : {filepath} ---")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("Erreur : Fichier non trouvé.")
        return

    # 1. Analyse structurelle et doublons
    print(f"\nDimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Doublons détectés : {df.duplicated().sum()}")
    print(f"Valeurs manquantes par colonne :\n{df.isnull().sum()}")

    # 2. Vérification de l'équilibre des données
    # Pour chaque profil, on attend : 5 Re * 60 alpha = 300 lignes
    profil_counts = df.groupby('naca')['alpha'].count()
    
    anomalies = profil_counts[profil_counts != 300]
    if not anomalies.empty:
        print(f" ATTENTION : {len(anomalies)} profils n'ont pas exactement 300 lignes.")
        print(anomalies.head())
    else:
        print(" Tous les profils possèdent exactement 300 lignes (Répartition équilibrée).")
    
    # 3. Différents plots pour voir les anomalies et la distribution des données
    sns.set_theme(style="whitegrid")
    
    # Distribution par Reynolds
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='Re')
    plt.title("Répartition globale par Reynolds")
    plt.show()

    # Distribution du nombre de lignes par profil (pour détecter les trous)
    plt.figure(figsize=(8, 4))
    sns.histplot(profil_counts, bins=30, kde=True)
    plt.title("Distribution du nombre de lignes par profil")
    plt.xlabel("Nombre de lignes")
    plt.ylabel("nombre d'occurrences")
    plt.show()

    # Vérification de la distribution de tous les angles alpha
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='alpha')
    plt.title("Fréquence de chaque angle d'attaque dans tout le dataset")
    plt.xticks(rotation=90) 
    plt.xlabel("Angle d'attaque (alpha)")
    plt.ylabel("Nombre d'occurrences")
    plt.show()
    # 4. Résumé statistique
    print("\n--- Résumé statistique ---")
    print(df.describe())
    
    # Exemple de contrôle de borne pour l'épaisseur (t)
    if df['t'].max() > 0.25 or df['t'].min() < 0.05:
        print(" Attention : Des valeurs d'épaisseur semblent hors des plages NACA standards.")
        
    
if __name__ == "__main__":
    audit_dataset()
