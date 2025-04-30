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


# %% EXPERIMENTAL


def compute_accuracy(X, y_sparse, weights):
    y_pred = predict(X, weights)
    y_true = y_sparse.argmax(axis=1).A1
    accuracy = np.mean(y_pred == y_true)
    return accuracy

def gradient_descent(X, y, X_val, y_val, weights, learning_rate, num_epochs):
    m, n = X.shape
    trainloss_history = []
    valloss_history = []
    trainacc_history = []
    valacc_history = []

    for epoch in range(num_epochs):
        grad = gradient(W_bar, W, y_lab, y_unlab_pred)
        weights -= learning_rate * grad

        if epoch % 5 == 0:
            train_cost = compute_cost(X, y, weights)
            val_cost = compute_cost(X_val, y_val, weights)
            train_accuracy = compute_accuracy(X, y, weights)
            val_accuracy = compute_accuracy(X_val, y_val, weights)
        
            trainloss_history.append(train_cost)
            valloss_history.append(val_cost)
            trainacc_history.append(train_accuracy)
            valacc_history.append(val_accuracy)

            print(f'Epoch {epoch} completed')
            print(f'Train Cost: {train_cost}, Train Accuracy: {train_accuracy}')
            print(f'Validation Cost: {val_cost}, Validation Accuracy: {val_accuracy}')
    
    return weights, trainloss_history, valloss_history, trainacc_history, valacc_history

# function that finds the block with higher norm of the gradient
# why? because it is the block that will have the most impact on the cost function
# and will be the most beneficial to update
def gauss_southwell_rule(gradient, block_size):
    n = gradient.shape[0]
    n_blocks = (n + block_size - 1) // block_size
    norms = [np.linalg.norm(gradient[i*block_size:(i+1)*block_size]) for i in range(n_blocks)]
    return np.argmax(norms)


def block_coordinate_gradient_descent_gs(X, y, X_val, y_val, weights, learning_rate, num_epochs, block_size):
    m, n = X.shape
    n_blocks = (n + block_size - 1) // block_size
    trainloss_history = []
    valloss_history = []
    trainacc_history = []
    valacc_history = []
    
    for epoch in range(num_epochs):
        # what does this do?
        # it computes the gradient of the cost function with respect to the weights
        # it is the same as the gradient descent algorithm, but we will only update a block of weights
        logits = X.dot(weights)
        probs = softmax(logits)
        gradient = X.T.dot(probs - y) / m
        
        # Chooses the block to opdate with the max norm of the gradient
        block_idx = gauss_southwell_rule(gradient, block_size)
        start = block_idx * block_size
        end = min((block_idx + 1) * block_size, n)
        
        # updates selected block
        weights[start:end, :] -= learning_rate * gradient[start:end, :]

        if epoch % 5 == 0:        
            train_cost = compute_cost(X, y, weights)
            val_cost = compute_cost(X_val, y_val, weights)
            train_accuracy = compute_accuracy(X, y, weights)
            val_accuracy = compute_accuracy(X_val, y_val, weights)
        
            trainloss_history.append(train_cost)
            valloss_history.append(val_cost)
            trainacc_history.append(train_accuracy)
            valacc_history.append(val_accuracy)

            print(f'Epoch {epoch} completed')
            print(f'Train Cost: {train_cost}, Train Accuracy: {train_accuracy}')
            print(f'Validation Cost: {val_cost}, Validation Accuracy: {val_accuracy}')
    
    return weights, trainloss_history, valloss_history, trainacc_history, valacc_history

def predict(X, weights):
    logits = X.dot(weights)
    return np.argmax(logits, axis=1)

# %%
"""Distinzione Sottile: A volte, si usa "Coordinate Gradient Descent" quando si fa un passo di gradiente lungo la coordinata 
e "Coordinate Minimization" quando si minimizza esattamente la funzione obiettivo rispetto a quella coordinata. 
Se il BCGD richiesto nella prima parte implica un passo di gradiente, la seconda parte ("Coordinate Minimization") 
potrebbe riferirsi alla versione con minimizzazione esatta."""