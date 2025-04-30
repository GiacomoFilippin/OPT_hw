# %%
from functions import get_data_dir, load_higgs_data, compute_weights_vectorized, compute_labeled_unlabeled_weights_vectorized, problem_to_solve, gradient, generate_semi_supervised_data
import numpy as np

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


# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import plotly.express as px

# read the data from the file data/raw/archive/train.csv e test.csv
data_dir = get_data_dir()
higgs_dataset = load_higgs_data(data_dir)

# --- Verifica Nomi Colonne (IMPORTANTE) ---
# Stampa i nomi delle colonne per assicurarti che 'EventId' e 'Label' siano corretti
# e per identificare le colonne delle features.
print("Columns:", higgs_dataset.columns)
# Esempio: Se la prima è 'EventId', l'ultima 'Label', e le altre sono features:
index_col_name = higgs_dataset.columns[0] # Es: 'EventId'
label_col_name = higgs_dataset.columns[-1] # Es: 'Label'
feature_col_names = higgs_dataset.columns[1:-1] # Colonne tra indice e label
# -----------------------------------------

# Crea una copia per sicurezza (opzionale ma consigliato)
df_work = higgs_dataset.copy()

# Standardize only the feature columns
scaler = StandardScaler()
# Applica lo scaler solo alle colonne delle features, mantenendo il DataFrame
df_work[feature_col_names] = scaler.fit_transform(df_work[feature_col_names])

# apply PCA with 3 components to the scaled feature data
pca = PCA(n_components=3)
# Applica PCA solo alle colonne delle features scalate
X_pca_3d = pca.fit_transform(df_work[feature_col_names])

# Create a DataFrame for plotting, using the original index for alignment
pca_df = pd.DataFrame(data=X_pca_3d, columns=['PC1', 'PC2', 'PC3'], index=df_work.index)

# Aggiungi la colonna Label originale al DataFrame PCA, pandas allinea automaticamente usando l'indice
pca_df[label_col_name] = df_work[label_col_name]

# Ora pca_df contiene le componenti PCA e la label corretta per ogni riga originale

# plot the data in the 3D PCA space using Plotly
fig = px.scatter_3d(pca_df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color=label_col_name, # Usa il nome corretto della colonna label
                    title='Higgs Dataset PCA Projection (3 Components, Standardized)',
                    labels={'PC1': 'Principal Component 1',
                            'PC2': 'Principal Component 2',
                            'PC3': 'Principal Component 3'},
                    opacity=0.7,
                    color_discrete_map={'s': 'red', 'b': 'blue'}) # Mappa le etichette ai colori

fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig.update_traces(marker=dict(size=3))
fig.show()
# %%
