import numpy as np
import matplotlib.pyplot as plt

# TODO: edge case with np.sign returns 0 for 0 values

def get_n_binary_patterns(n_patterns, pattern_dim, plot_patterns=False, plot_patterns_overlaps=False):
    """
    Generate n binary patterns of dimension pattern_dim with equal number of -1 and 1s.

    Args:
        :param n_patterns: (int) Number of patterns to generate.
        :param pattern_dim: (int): Dimension of each pattern.
        :param plot_patterns: (bool): If True, plot the generated patterns.
        :param plot_patterns_overlaps: if True, show overlap between pairs of patterns, sanity check to ensure pattern overlap is near 0 for different patterns and 1 for same patterns.
    Returns:
        np.ndarray: Array of shape (n_patterns, pattern_dim) containing the generated binary patterns.

    """
    # p = np.random.binomial(1, probability, size=(n_patterns, pattern_dim)) * 2 - 1

    p = np.zeros((n_patterns, pattern_dim), dtype=int) - 1

    for i in range(n_patterns):
        indices = np.random.choice(pattern_dim, pattern_dim // 2, replace=False)
        p[i, indices] = 1


    if plot_patterns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(p, cmap='gray')
        ax.set_title(f"Generated {n_patterns} binary patterns of dimension {pattern_dim}")
        ax.grid(False)
    if plot_patterns_overlaps:
        fig, ax = plt.subplots(1, 1)
        ovlps = np.zeros((n_patterns, n_patterns))
        for i in range(n_patterns):
            for j in range(n_patterns):
                ovlps[i, j] = np.dot(p[i], p[j]) / pattern_dim
        # for i in range(n_patterns):
        #     ovlps[i, i] = 0
        # ax.imshow(ovlps.T, cmap='gray')
        plt.colorbar(ax.imshow(ovlps.T, cmap='gray_r'), ax=ax)
        ax.set_title("Overlaps between patterns")
        ax.set_xlabel("Pattern index")
        ax.set_ylabel("Pattern index")
        ax.grid(False)
    return p

def generate_masks(N):
    masks = np.zeros((N, N), dtype=int)
    for i in range(N):
        indices = np.random.choice(N, N // 2, replace=False)
        masks[i, indices] = 1
    return masks


def get_weights(patterns, self_connections=True, show_weights=False, diluted=False):
    n_patterns = patterns.shape[0]
    dim_patterns = patterns.shape[1]
    weights = np.zeros((dim_patterns, dim_patterns))
    #the weights matrix is the sum of the outer products of each pattern with itself
    for i in range(n_patterns):
        weights += np.outer(patterns[i], patterns[i])
    weights /= dim_patterns

    if not self_connections:
        np.fill_diagonal(weights, 0)

    if diluted:
        mask = generate_masks(dim_patterns)
        weights *= mask


    # show the weights, with colorbar to map color and value
    if show_weights:
        plt.figure()
        plt.imshow(weights, cmap='gray_r', interpolation='nearest')
        plt.colorbar()
        plt.title("Weights")
        plt.xlabel("Presynaptic neuron")
        plt.ylabel("Postsynaptic neuron")
        plt.grid(False)

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
    n_dim = patterns.shape[1]
    overlaps = patterns @ current_state / n_dim
    weighted_patterns = overlaps[:, np.newaxis] * patterns
    h = weighted_patterns.sum(axis=0)
    return np.sign(h)



def compute_next_state_with_overlaps_loops(current_state, patterns, return_h = False):
    h = np.zeros(current_state.shape)
    overlaps = np.zeros(patterns.shape[0])
    N = len(current_state)
    P = len(patterns)

    for mu in range(P):
        overlaps[mu] = np.sum([patterns[mu,i]*current_state[i] for i in range(N)])/N

    for i in range(N):
        h[i] = np.sum([overlaps[mu]*patterns[mu,i] for mu in range(P)])

    return h if return_h else np.sign(h)


def compute_next_state_stochastic(current_state, weights, b):
    h = current_state @ weights
    prob = (1 + np.tanh(b*h))/2
    return np.random.rand(len(current_state)) < prob

def overlap_stochastic(a, b):
    return 2*np.dot(a, b)/len(a)

def compute_next_state_stochastic_refractory(current_state, weights, b):
    h = current_state @ weights
    prob = (1 + np.tanh(b*h))/2
    prob[current_state == 1] = 0
    return np.random.rand(len(current_state)) < prob









