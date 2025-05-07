#%% this contains the algorithms for the project,
# in particular:
# GRADIENT DESCEND
# BCGD w GS rule (with dimension 1)
# COORDINATE MINIMIZATION
import numpy as np
from scipy.spatial.distance import cdist
import os
import math
import time
import matplotlib.pyplot as pl
import numpy as np
import math
PLOT_DPI=100


# we have this parameters:
w, w_bar = _, _ # weights between unlabeled and labeled data, and between unlabeled data
x_lab, y_lab = _, _ # labeled data
x_unlab, y_unlab = _, _ # unlabeled data
y_unlab_pred_initial, y_unlab_pred = _, _ # initial and target prediction for unlabeled data
# dobbiamo assicurarci di usarle e sostituirle nel modo giusto
# common functins:
# %% now i have to define a weight function to measure the distance between two generic points, or even better for each pair of points
def weight_function_scaled(dist):
    return math.e**(-10* dist**2)

X = np.linspace(-2, 2, 100)
pl.figure(dpi=PLOT_DPI)
ax = pl.gca()
ax.legend_= None
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.set_ylabel('weight')
ax.set_xlabel('distance')
ax.axvline(x=0, color='k')
ax.plot(X, [weight_function_scaled(d) for d in X], label="asd")
# %%
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
    weights = weight_function_scaled(distances)
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
    weights = weight_function_scaled(distances)

    return weights

#W_bar = compute_weights_vectorized(X_unlab)
#W = compute_labeled_unlabeled_weights_vectorized(X_lab, X_unlab)
# %%
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
def loss_iterative(W_bar, W, y_lab, y_unlab_pred):
    first_term = second_term = 0.0
    for i in range(len(y_lab)):
        for j in range(len(y_unlab_pred)):
            first_term += W[i][j] * (y_unlab_pred[j] - y_lab[i])**2
    for i in range(len(y_unlab_pred)):
        for j in range(len(y_unlab_pred)):
            second_term += W_bar[i][j] * (y_unlab_pred[i] - y_unlab_pred[j])**2
    return first_term + second_term/2

# implement the loss function
def loss(W_bar, W, y_lab, y_unlab_pred):
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


# %%
"""
THIS IS UNCLEAR: why do i have y_unlabeled_target? shouldn't it be unknown being a semi sup problem?
# answer:
??
"""
#y_unlabeled_target = w_bar = w = y_unlabeled_initial = y_lab = 0

def accuracy_round(current):
    return np.sum(
        np.equal(
            (current/2 + .5).round() * 2 - 1,
            y_unlabeled_target
        )
    ) / len(y_unlabeled_target)
max_loss = np.sum(w) * 4 + np.sum(w_bar) * 2
def accuracy_loss(current):
    return 1 - loss(y_lab, current) / max_loss

initial_distance = np.linalg.norm(y_unlabeled_initial - y_unlabeled_target, ord=2)
def accuracy_norm(current):
    current_distance = np.linalg.norm(current - y_unlabeled_target, ord=2)
    return 1 - current_distance / initial_distance

def accuracy(current):
    return accuracy_round(current) #loss(y_labeled, current)

# %%
"""
THIS IS NOT CLEAR: where is he actually doing the second order derivative?
HESSIAN MATRIX used to compute the second order derivative of the loss function.
used to compute lipschitz constant for the gradient descent algorithm.
The Hessian matrix is a square matrix of second-order partial derivatives of a scalar-valued function.
"""
import scipy
def hessian_matrix(w_unlabeled_unlabeled, w_labeled_unlabeled):
    mat = np.copy(-w_unlabeled_unlabeled)
    for i in range(len(y_unlabeled_target)):
        if i % (len(y_unlabeled_target) * OUTPUT_STEP) == 0 :
            print(f"{int(i/len(y_unlabeled_target)*100):02}% ... ", end="")
        mat[i][i] = 2 * np.sum(w_labeled_unlabeled[:,i]) + np.sum(w_unlabeled_unlabeled[:,i]) - w_unlabeled_unlabeled[i][i]
    print()
    return mat

def estimate_lipschitz_constant(hessian):
    return scipy.linalg.eigh(hessian, subset_by_index=(len(hessian)-1, len(hessian)-1))[0][0]
def estimate_degree_strongly_convex(hessian):
    return scipy.linalg.eigh(hessian, subset_by_index=(0,0))[0][0]

print("Calculating the Hessian matrix")
hessian = hessian_matrix(w_bar, w)
print("Calculating sigma (strongly convex)")
sigma = estimate_degree_strongly_convex(hessian)
strongly_convex = sigma > 0
print(f"Sigma: {sigma}, {'' if strongly_convex else 'not'} strongly convex")
print("Estimating Lipschitz constant for the whole function")
L = estimate_lipschitz_constant(hessian)
print(f"Lipschitz constant: {L}")
print("Estimating Lipschitz constant for each single variable")
Li = np.array([hessian[i][i] for i in range(len(hessian))], dtype='float64') \
    if OPTIMIZE_LIPSCHITZ_CONSTANT_FOR_BCGM \
    else np.repeat(L, len(hessian))

# %% gradient_efficient computation
print("Calculating coefficients of the gradient")
sum_row_label_unlabel = np.sum(w, axis=0).reshape((-1,1))
sum_row_unlabel_unlabel = np.sum(w_bar, axis=0).reshape((-1,1))
grad_first_coeff_vec = (2 * sum_row_label_unlabel + sum_row_unlabel_unlabel)
grad_second_coeff_vec = 2 * np.multiply(
    y_labeled.reshape((-1,1)),
    w
).sum(0).reshape((-1,1))
print("Done")
def gradient_efficient(unlabeled):
    unlabeled = unlabeled.reshape((-1,1))
    third_coeff = np.multiply(
        unlabeled,
        w_bar
    ).sum(0).reshape((-1,1))
    return (grad_first_coeff_vec * unlabeled - grad_second_coeff_vec - third_coeff).reshape((-1,))

# %% gradient descent
# inizialize the solution:
y_unlabeled_initial = np.random.choice([-1, 1], size=(len(x_unlabeled),))
y_unlabeled_gd = np.copy(y_unlabeled_initial)
# %% NOTE learning rate has been improved WRT 1/L
ticks_gd_improved = [0]
accuracies_gd_improved = [accuracy(y_unlabeled_gd_improved)]
losses_gd_improved = [loss(y_labeled, y_unlabeled_gd_improved)]
init_time = time.time() * 1000
for i in range(NUM_ITERS):
    if i % (NUM_ITERS * OUTPUT_STEP) == 0 :
        print(f"{int(i/NUM_ITERS*100):02}% - iter {i} / {NUM_ITERS}", end="")
    start = time.process_time()
    grad = gradient_efficient(y_unlabeled_gd_improved)

    if EPSILON >= np.linalg.norm(grad):
        break

    y_unlabeled_gd_improved = y_unlabeled_gd_improved - (2 if strongly_convex else 1)/(L + (sigma if strongly_convex else 0)) * grad
    accuracies_gd_improved.append(accuracy(y_unlabeled_gd_improved))
    losses_gd_improved.append(loss(y_labeled, y_unlabeled_gd_improved))
    ticks_gd_improved.append(time.time() * 1000 - init_time)

    if CREATE_GIF:
        create_frame(i, x_unlabeled, y_unlabeled_gd_improved, "./gif/gd_improved/", "Gradient Descent Improved Rate")

    if i % (NUM_ITERS * OUTPUT_STEP) == 0 :
        print(f" - final accuracy {(accuracies_gd_improved[-1])*100:05.2f}% - loss {loss(y_labeled, y_unlabeled_gd_improved):05.2f}")
print(f"""
Final accuracy: {round(accuracies_gd_improved[-1]* 100, 2)}%
Norm gradient: {np.linalg.norm(gradient_efficient(y_unlabeled_gd_improved))}
Loss: {loss(y_labeled, y_unlabeled_gd_improved)}
""")

# %% stampa i dati unlabelled colorati per previsione
# stampa andamento accuracy e loss
# CPU time

# %% BCGD with GS rule (with dimension 1)

def derivative(unlabeled, j):
# computes partial derivative of loss wrt j-th component of unlab. vector
# Nel contesto del BCGD, dove si aggiorna una coordinata alla volta, questa derivata parziale 
# # è ciò che serve per decidere la direzione e l'entità dell'aggiornamento per quella specifica coordinata.
    return grad_first_coeff_vec[j] * unlabeled[j] - \
           grad_second_coeff_vec[j] - \
           np.matmul(w_bar[:,j], unlabeled.T)

# okay (?)
def update_gradient(previous_grad, Li, index):
    global w_bar, grad_first_coeff_vec
    step = 1/Li[index] * previous_grad[index]
    new_grad = np.copy(previous_grad)
    new_grad += w_bar[index] * step
    new_grad[index] -= grad_first_coeff_vec[index] * step
    return step, new_grad

# %%
y_unlabeled_bcgd_gs = np.copy(y_unlabeled_initial).astype("float64")

ticks_bcgd_gs = [0]
accuracies_bcgd_gs = [accuracy(y_unlabeled_bcgd_gs)]
losses_bcgm_gs = [loss(y_labeled, y_unlabeled_bcgd_gs)]

max_iterations = NUM_ITERS*len(y_unlabeled_bcgd_random)
previous_grad_gs = gradient(y_unlabeled_bcgd_gs)

init_time = time.time() * 1000

for i in range(max_iterations):
    if i % (max_iterations * OUTPUT_STEP) == 0 :
        print(f"{int(i/max_iterations*100):02}% - iter {i} / {max_iterations}", end="")

    # this is the essence of the Gauss-Southwell rule:
    # it chooses the coordinate with the largest absolute value of the gradient
    index = np.argmax(np.abs(previous_grad_gs))

    step, new_grad = update_gradient(previous_grad_gs, Li, index)
    y_unlabeled_bcgd_gs[index] = y_unlabeled_bcgd_gs[index] - step
    previous_grad_gs = new_grad

    if EPSILON >= np.linalg.norm(previous_grad_gs):
        break

    if (i+1) % len(y_unlabeled_target) == 0:

        if CREATE_GIF:
            create_frame(i // len(y_unlabeled_target), x_unlabeled, y_unlabeled_bcgd_gs, "./gif/bcgd_gs/", "BCGD Gauss-Southwell")

        accuracies_bcgd_gs.append(accuracy(y_unlabeled_bcgd_gs))
        losses_bcgm_gs.append(loss(y_labeled, y_unlabeled_bcgd_gs))
        ticks_bcgd_gs.append(time.time() * 1000 - init_time)

    if i % (max_iterations * OUTPUT_STEP) == 0 :
        print(f" - final accuracy {(accuracies_bcgd_gs[-1])*100:05.2f}% - loss {loss(y_labeled, y_unlabeled_bcgd_gs):05.2f}")


print(f"""
Final accuracy: {round(accuracies_bcgd_gs[-1]* 100, 2)}%
Norm gradient: {np.linalg.norm(gradient(y_unlabeled_bcgd_gs))}
Loss: {loss(y_labeled, y_unlabeled_bcgd_gs)}
""")
# %% stampa i dati unlabelled colorati per previsione
# stampa andamento accuracy e loss
# CPU time

