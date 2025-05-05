# %%
from functions import get_data_dir, load_higgs_data, compute_weights_vectorized, compute_labeled_unlabeled_weights_vectorized, problem_to_solve, gradient, generate_semi_supervised_data
import numpy as np
data_dir = get_data_dir()
# %%
X_lab, y_lab, X_unlab = generate_semi_supervised_data(
    n_samples=10000, noise=0.8, labeled_proportion=0.15, random_state=42
)
# y_unlab ?
# You can now use X_lab, y_lab, and X_unlab for your semi-supervised model
# For example, visualize the data:
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
# Plot labeled data
plt.scatter(X_lab[y_lab == 1, 0], X_lab[y_lab == 1, 1], c='blue', label='Labeled Class 1', marker='o', s=50)
plt.scatter(X_lab[y_lab == -1, 0], X_lab[y_lab == -1, 1], c='red', label='Labeled Class -1', marker='o', s=50)
# Plot unlabeled data
plt.scatter(X_unlab[:, 0], X_unlab[:, 1], c='gray', label='Unlabeled Data', marker='.', alpha=0.5)
plt.title('Semi-Supervised Dataset (Two Moons)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
print("Matplotlib not found. Skipping visualization.")
print("Install it using: pip install matplotlib")

print("\nLabeled Features Shape:", X_lab.shape)
print("Labeled Labels Shape:", y_lab.shape)
print("Unlabeled Features Shape:", X_unlab.shape)
# %%
#now i have to define a weight function to measure the distance between two generic points, or even better for each pair of points
# ...existing code...
# now i have to compute weights between
# Use the vectorized functions
W_bar = compute_weights_vectorized(X_unlab)
W = compute_labeled_unlabeled_weights_vectorized(X_lab, X_unlab)
# %%
#now, i need to cisider the problem, and solve the problem with
# gradient descend, Block Coordinate Grandient Descent (BCGD) with GS rule (with dimension 1) and coordinate minimization

# i need to define the gradient of the problem, and then implement the BCGD with GS rule (with dimension 1) and coordinate minimization

"""Distinzione Sottile: A volte, si usa "Coordinate Gradient Descent" quando si fa un passo di gradiente lungo la coordinata 
e "Coordinate Minimization" quando si minimizza esattamente la funzione obiettivo rispetto a quella coordinata. 
Se il BCGD richiesto nella prima parte implica un passo di gradiente, la seconda parte ("Coordinate Minimization") 
potrebbe riferirsi alla versione con minimizzazione esatta."""


# %% REAL DATA LOADING
# --- Loading Spambase Dataset ---
import os
import pandas as pd
data_filename = os.path.join(data_dir, "spambase.data")

# Definisci i nomi delle colonne basandoti sulla descrizione in spambase.names
# (57 attributi + 1 colonna target 'is_spam')

# Nomi generici per le frequenze delle parole (48 colonne)
# Potresti estrarre i nomi specifici da spambase.DOCUMENTATION se necessario
word_freq_names = [f'word_freq_{i+1}' for i in range(48)]

# Nomi per le frequenze dei caratteri (6 colonne)
char_freq_names = ['char_freq_;', 'char_freq_(', 'char_freq_[',
                   'char_freq_!', 'char_freq_$', 'char_freq_#'] # Nomi specifici da documentazione

# Nomi per le lunghezze delle sequenze di maiuscole (3 colonne)
capital_run_names = [
    'capital_run_length_average',
    'capital_run_length_longest',
    'capital_run_length_total'
]

# Nome della colonna target (1 colonna)
target_name = ['is_spam']

# Combina tutti i nomi in ordine
column_names = word_freq_names + char_freq_names + capital_run_names + target_name

# Verifica che il numero totale di nomi sia corretto (58)
if len(column_names) != 58:
    print(f"Attenzione: Generati {len(column_names)} nomi, ma ne sono attesi 58.")

try:
    # Carica il file .data usando pandas
    # header=None perché il file non ha intestazioni
    # names=column_names per assegnare i nomi definiti sopra
    df = pd.read_csv(data_filename, header=None, names=column_names)

    print("Dati caricati con successo in un DataFrame pandas:")
    print(df.head()) # Mostra le prime 5 righe

    # Puoi vedere informazioni riassuntive sul DataFrame
    print("\nInformazioni sul DataFrame:")
    df.info()

    # Ora puoi usare il DataFrame 'df' per analisi o addestramento modelli
    # Esempio: separare features (X) e target (y)
    # X = df.drop('is_spam', axis=1)
    # y = df['is_spam']
    # print("\nPrime 5 righe delle features (X):")
    # print(X.head())
    # print("\nPrime 5 righe della variabile target (y):")
    # print(y.head())


except FileNotFoundError:
    print(f"Errore: File non trovato '{data_filename}'.")
    print(f"Assicurati che il file esista nella directory: '{os.path.abspath(data_dir)}'")
    print("Esegui prima lo script 'data downloader.py'.")
except Exception as e:
    print(f"Errore durante il caricamento del file '{data_filename}': {e}")

# %%
# --- PCA Visualization for Spambase Dataset ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd # Assicurati che pandas sia importato

# Verifica che il DataFrame 'df' esista e contenga i dati Spambase
if 'df' not in locals():
    print("Errore: Il DataFrame 'df' con i dati Spambase non è stato trovato.")
    print("Assicurati di aver eseguito la cella precedente che carica 'spambase.data'.")
else:
    # Separa features (X) e target (y)
    spambase_target_col = 'is_spam'
    spambase_feature_cols = df.columns.drop(spambase_target_col)

    X_spam = df[spambase_feature_cols]
    y_spam = df[spambase_target_col]

    # Standardizza le features
    scaler_spam = StandardScaler()
    X_spam_scaled = scaler_spam.fit_transform(X_spam)

    # Applica PCA con 3 componenti
    pca_spam = PCA(n_components=3)
    X_pca_3d_spam = pca_spam.fit_transform(X_spam_scaled)

    # Crea un DataFrame per il plotting
    # Usa l'indice originale di df per mantenere l'allineamento
    pca_df_spam = pd.DataFrame(data=X_pca_3d_spam,
                               columns=['PC1', 'PC2', 'PC3'],
                               index=df.index)

    # Aggiungi la colonna target originale al DataFrame PCA
    pca_df_spam[spambase_target_col] = y_spam

    # Plotta i dati nello spazio PCA 3D usando Plotly
    fig_spam = px.scatter_3d(pca_df_spam,
                             x='PC1',
                             y='PC2',
                             z='PC3',
                             color=spambase_target_col, # Colora per 'is_spam'
                             title='Spambase Dataset PCA Projection (3 Components, Standardized)',
                             labels={'PC1': 'Principal Component 1',
                                     'PC2': 'Principal Component 2',
                                     'PC3': 'Principal Component 3',
                                     spambase_target_col: 'Is Spam (1=Spam, 0=Not Spam)'}, # Etichetta legenda
                             opacity=0.7,
                             color_continuous_scale=px.colors.sequential.Viridis, # Scala di colori per 0/1
                             # Se preferisci colori discreti:
                             # color_discrete_map={0: 'blue', 1: 'red'} # Mappa 0 e 1 ai colori
                             )

    fig_spam.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    # Rendi i punti leggermente più piccoli se sono troppi
    fig_spam.update_traces(marker=dict(size=2))
    fig_spam.show()

# %%
