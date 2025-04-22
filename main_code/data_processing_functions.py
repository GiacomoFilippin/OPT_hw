# %%
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def generate_semi_supervised_data(n_samples=1000, noise=0.1, labeled_proportion=0.1, random_state=42):
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
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

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


# %%
X_lab, y_lab, X_unlab = generate_semi_supervised_data(
    n_samples=500,
    labeled_proportion=0.05, # 5% labeled data
    random_state=123
)

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