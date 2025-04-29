
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import numpy as np

def generate_semi_supervised_data(n_samples=10000, noise=0.15, labeled_proportion=0.1, random_state=42):
    """
    Generates a semi-supervised dataset with two classes (-1 and 1).

    Args:
        n_samples (int): Total number of samples to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.
        labeled_proportion (float): Proportion of the data to be labeled (between 0 and 1).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - X_labeled (np.ndarray): Features of the labeled data.
            - y_labeled (np.ndarray): Labels (-1 or 1) of the labeled data.
            - X_unlabeled (np.ndarray): Features of the unlabeled data.
    """
    # Generate a two-moon dataset
    #X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise, random_state=random_state)
    # Convert labels from {0, 1} to {-1, 1}
    y = y * 2 - 1

    # Split into labeled and unlabeled sets
    n_labeled = int(n_samples * labeled_proportion)

    # Ensure at least one sample per class in the labeled set if possible
    # This simple split might not guarantee class balance in the small labeled set
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X, y, train_size=n_labeled, random_state=random_state, stratify=y
    )

    # For the unlabeled set, we typically only use the features (X_unlabeled)
    # The y_unlabeled is usually discarded or ignored in semi-supervised settings,
    # but we return it here for potential analysis or alternative approaches.
    # In many algorithms, you'd just pass X_unlabeled.

    print(f"Generated dataset with {n_samples} total samples.")
    print(f"Labeled samples: {len(X_labeled)}")
    print(f"Unlabeled samples: {len(X_unlabeled)}")
    print(f"Labeled class distribution: {np.unique(y_labeled, return_counts=True)}")

    return X_labeled, y_labeled, X_unlabeled

#now i have to define a weight function to measure the distance between two generic points, or even better for each pair of points
def weight_function(distances):
    """
    Computes weights based on distances. Handles zero distances.

    Args:
        distances (np.ndarray): Array of distances.

    Returns:
        np.ndarray: Array of weights.
    """
    # Avoid division by zero by adding a small epsilon or handling zeros explicitly
    weights = np.zeros_like(distances)
    non_zero_mask = distances != 0
    weights[non_zero_mask] = 1.0 / distances[non_zero_mask]
    return weights

def compute_weights_vectorized(X):
    """
    Computes the weight matrix for a given dataset using vectorized operations.

    Args:
        X (np.ndarray): Dataset of shape (n_samples, n_features).

    Returns:
        np.ndarray: Weight matrix of shape (n_samples, n_samples).
    """
    # Compute pairwise Euclidean distances
    distances = cdist(X, X, metric='euclidean')

    # Compute weights using the vectorized weight function
    weights = weight_function(distances)

    # Ensure diagonal elements are zero (weight of a point with itself)
    # This is already handled by weight_function if distance is 0,
    # but explicitly setting it reinforces the requirement.
    np.fill_diagonal(weights, 0)

    return weights

# i also need a function that computes weights between each labeled and unlabeled pair
def compute_labeled_unlabeled_weights_vectorized(X_labeled, X_unlabeled):
    """
    Computes the weight matrix between labeled and unlabeled data using vectorized operations.

    Args:
        X_labeled (np.ndarray): Labeled dataset of shape (n_labeled_samples, n_features).
        X_unlabeled (np.ndarray): Unlabeled dataset of shape (n_unlabeled_samples, n_features).

    Returns:
        np.ndarray: Weight matrix of shape (n_labeled_samples, n_unlabeled_samples).
    """
    # Compute pairwise Euclidean distances between labeled and unlabeled points
    distances = cdist(X_labeled, X_unlabeled, metric='euclidean')

    # Compute weights using the vectorized weight function
    weights = weight_function(distances)

    return weights

# now we define the two terms loss funcion, as:
"""
L(y_unlab) = Σ_{i ∈ labeled, j ∈ unlabeled} W_ij * (y_lab_i - y_unlab_j)² + (1/2) * Σ_{k ∈ unlabeled, m ∈ unlabeled} W_bar_km * (y_unlab_k - y_unlab_m)²

Spiegazione dei termini:

Primo Termine (Consistenza con le etichette note):

Σ_{i ∈ labeled, j ∈ unlabeled} W_ij * (y_lab_i - y_unlab_j)²
Somma su tutte le coppie formate da un punto etichettato i e un punto non etichettato j.
W_ij è il peso (similarità) tra il punto etichettato i e il punto non etichettato j (elementi della matrice W).
(y_lab_i - y_unlab_j)² penalizza la differenza quadratica tra l'etichetta nota y_lab_i e l'etichetta incognita y_unlab_j.
Questo termine spinge le etichette dei punti non etichettati ad essere simili alle etichette dei punti etichettati vicini.
Secondo Termine (Regolarizzazione/Levigatezza tra le etichette incognite):

(1/2) * Σ_{k ∈ unlabeled, m ∈ unlabeled} W_bar_km * (y_unlab_k - y_unlab_m)²
Somma su tutte le coppie di punti non etichettati k e m.
W_bar_km è il peso (similarità) tra i punti non etichettati k e m (elementi della matrice W_bar).
(y_unlab_k - y_unlab_m)² penalizza la differenza quadratica tra le etichette incognite y_unlab_k e y_unlab_m.
Questo termine spinge le etichette dei punti non etichettati vicini ad essere simili tra loro, promuovendo una soluzione "liscia" sul grafo. Il fattore 1/2 è spesso incluso per convenzione o per semplificare i calcoli del gradiente.
L'obiettivo è trovare il vettore y_unlab che minimizza questa funzione di perdita L(y_unlab).
"""
# implement the loss function
def problem_to_solve(W_bar, W, y_lab, y_unlab_pred):
    """
    Computes the loss function for the semi-supervised learning problem.

    Args:
        W_bar (np.ndarray): Weight matrix for unlabeled data (n_unlabeled_samples, n_unlabeled_samples).
        W (np.ndarray): Weight matrix between labeled and unlabeled data (n_labeled_samples, n_unlabeled_samples).
        y_lab (np.ndarray): Labels for the labeled data (n_labeled_samples,).
        y_unlab (np.ndarray): Labels for the unlabeled data (n_unlabeled_samples,).

    Returns:
        float: The computed loss value.
    """
    # First term: consistency with known labels
    first_term = np.sum(W * np.square(y_lab[:, np.newaxis] - y_unlab_pred[np.newaxis, :]))

    # Second term: smoothness among unlabeled points
    second_term = 0.5 * np.sum(W_bar * np.square(y_unlab_pred[:, np.newaxis] - y_unlab_pred[np.newaxis, :]))

    return first_term + second_term

def gradient(W_bar, W, y_lab, y_unlab_pred):
    """
    Computes the gradient of the loss function with respect to y_unlab.

    Args:
        W_bar (np.ndarray): Weight matrix for unlabeled data (n_unlabeled_samples, n_unlabeled_samples).
        W (np.ndarray): Weight matrix between labeled and unlabeled data (n_labeled_samples, n_unlabeled_samples).
        y_lab (np.ndarray): Labels for the labeled data (n_labeled_samples,).
        y_unlab_pred (np.ndarray): Current predictions for the unlabeled data (n_unlabeled_samples,).

    Returns:
        np.ndarray: The computed gradient vector.
    """
    # Gradient with respect to y_unlab
    grad_first_term = -2 * np.sum(W * (y_lab[:, np.newaxis] - y_unlab_pred[np.newaxis, :]), axis=0)
    grad_second_term = -np.sum(W_bar * (y_unlab_pred[:, np.newaxis] - y_unlab_pred[np.newaxis, :]), axis=1)

    return grad_first_term + grad_second_term
    
