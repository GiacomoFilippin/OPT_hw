# %%
from functions import compute_weights_vectorized, compute_labeled_unlabeled_weights_vectorized, problem_to_solve, gradient, generate_semi_supervised_data
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


