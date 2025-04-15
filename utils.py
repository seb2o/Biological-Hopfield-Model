import numpy as np
import matplotlib.pyplot as plt

# TODO: edge case with np.sign returns 0 for 0 values

def get_n_binary_patterns(n_patterns, pattern_dim, probability=0.5, plot=False, check_overlaps=False):
    """
    Generate n binary patterns of dimension pattern_dim with a given probability (default is balanced).

    Args:
        :param n_patterns: (int) Number of patterns to generate.
        :param pattern_dim: (int): Dimension of each pattern.
        :param probability: (float): default 0.5. Probability of 1 in the generated patterns.
        :param plot: (bool): If True, plot the generated patterns.
        :param check_overlaps: if True, show overlap between pairs of patterns, sanity check to ensure pattern overlap is near 0 for different patterns and 1 for same patterns.
    Returns:
        np.ndarray: Array of shape (n_patterns, pattern_dim) containing the generated binary patterns.

    """
    p = np.random.binomial(1, probability, size=(n_patterns, pattern_dim)) * 2 - 1
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(p, cmap='gray')
        ax.set_title(f"Generated {n_patterns} binary patterns of dimension {pattern_dim}")
    if check_overlaps:
        fig, ax = plt.subplots(1, 1)
        ovlps = np.zeros((n_patterns, n_patterns))
        for i in range(n_patterns):
            for j in range(n_patterns):
                ovlps[i, j] = np.dot(p[i], p[j]) / pattern_dim
        # ax.imshow(ovlps.T, cmap='gray')
        plt.colorbar(ax.imshow(ovlps.T, cmap='gray_r'), ax=ax)
        ax.set_title("Overlaps between patterns")
        ax.set_xlabel("Pattern index")
        ax.set_ylabel("Pattern index")
    return p


def get_weights(patterns, self_connections=False, show_weights=False):
    n_patterns = patterns.shape[0]
    dim_patterns = patterns.shape[1]
    weights = np.zeros((dim_patterns, dim_patterns))
    #the weights matrix is the sum of the outer products of each pattern with itself
    for i in range(n_patterns):
        weights += np.outer(patterns[i], patterns[i])
    weights /= n_patterns

    if not self_connections:
        np.fill_diagonal(weights, 0)

    # show the weights, with colorbar to map color and value
    if show_weights:
        plt.figure()
        plt.imshow(weights, cmap='gray_r', interpolation='nearest')
        plt.colorbar()
        plt.title("Weights")
        plt.xlabel("Presynaptic neuron")
        plt.ylabel("Postsynaptic neuron")

    return weights

def get_weights_with_loops(patterns):
    n_patterns, dim_patterns = patterns.shape

    weights = np.zeros((dim_patterns, dim_patterns))
    #the weights matrix is the sum of the outer products of each pattern with itself
    for i in range(dim_patterns):
        for j in range(dim_patterns):
            for mu in range(n_patterns):
                weights[i, j] += patterns[mu, i] * patterns[mu, j]
    weights /= n_patterns

    np.fill_diagonal(weights, 0)

    return weights

def compute_next_state(current_state, weights, return_h=False):
    h = weights @ current_state
    return h if return_h else np.sign(h)

def compute_next_state_with_loops(current_state, weights):
    h = np.zeros(current_state.shape)
    for current_unit in range(len(h)):
        for previous_unit in range(len(h)):
            h[current_unit] += weights[current_unit, previous_unit] * current_state[previous_unit]
    return np.sign(h)

def network_step(current_state, patterns):
    weights = get_weights(patterns, self_connections=False)
    next_state = compute_next_state(current_state, weights)
    return next_state

def overlap(x,y):
    if len(x) != len(y):
        raise ValueError("Vectors must be the same length")
    return np.dot(x,y) / len(x)

def compute_next_state_with_overlaps(current_state, patterns, return_h=False):
    h = np.zeros(current_state.shape)
    for p in patterns:
        o = overlap(current_state, p)
        h += (o * p)
    return h if return_h else np.sign(h)

def compute_next_state_with_overlaps_vectorized(current_state, patterns):
    n_patterns = patterns.shape[0]
    n_dim = patterns.shape[1]
    overlaps = patterns @ current_state / n_dim
    weighted_patterns = overlaps[:, np.newaxis] * patterns
    h = weighted_patterns.sum(axis=0)
    return np.sign(h)



def compute_next_state_with_overlaps_loops(current_state, patterns):
    h = np.zeros(current_state.shape)
    overlaps = np.zeros(patterns.shape[0])

    for mu in range(len(patterns)):
        overlaps[mu] = overlap(current_state, patterns[mu])

    for unit_index in range(len(h)):
        for pattern_index in range(len(patterns)):
            h[unit_index] += overlaps[pattern_index] * patterns[pattern_index][unit_index]

    return np.sign(h)














