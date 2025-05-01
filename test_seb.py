import numpy as np
import matplotlib.pyplot as plt

def get_weights(patterns):
    n_patterns = patterns.shape[0]
    network_dim = patterns.shape[1]
    weights = np.zeros((network_dim, network_dim))
    #the weights matrix is the sum of the outer products of each pattern with itself
    for i in range(n_patterns):
        weights += np.outer(patterns[i], patterns[i])
    weights /= n_patterns

    return weights

def compute_next_state(current_state, weights, return_h=False):
    h = weights @ current_state
    return h if return_h else np.sign(h)



def compute_next_state_with_overlaps_loops(current_state, patterns, return_h=False):
    h = np.zeros(current_state.shape)
    overlaps = np.zeros(patterns.shape[0])
    N = len(current_state)
    P = len(patterns)

    for mu in range(P):
        for i in range(N):
            overlaps[mu] += patterns[mu][i] * current_state[i]
        overlaps[mu] /= N

    for unit_index in range(len(h)):
        for pattern_index in range(P):
            h[unit_index] += overlaps[pattern_index] * patterns[pattern_index][unit_index]

    return h if return_h else np.sign(h)


if __name__ == "__main__":
    NETWORK_DIM = 1340
    c = 0.134
    n_patterns = int(c * NETWORK_DIM)

    patterns = np.random.binomial(1, 0.5, size=(n_patterns, NETWORK_DIM)) * 2 - 1
    current_state = np.random.binomial(1, 0.5, size=NETWORK_DIM) * 2 - 1

    weights = get_weights(patterns)
    ns_standard_h = compute_next_state(current_state, weights, return_h=True)

    ns_overlap_h = compute_next_state_with_overlaps_loops(current_state, patterns, return_h=True)


    plt.hist(ns_standard_h - ns_overlap_h, bins=100)
    plt.show()