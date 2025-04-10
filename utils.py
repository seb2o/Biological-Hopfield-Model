import numpy as np
import matplotlib.pyplot as plt


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
