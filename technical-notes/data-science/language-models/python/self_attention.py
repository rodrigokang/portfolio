"""
Implementation of causal scaled dot-product self-attention from first
principles using NumPy.
"""

import numpy as np

def softmax(x):
    """
    Compute the softmax function along the last dimension.

    Input
    -----
    x : np.ndarray
        Input array.

    Output
    ------
    np.ndarray
        Softmax-normalised array.
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def causal_self_attention(X, W_Q, W_K, W_V):
    """
    Compute causal scaled dot-product self-attention.

    Input
    -----
    X : np.ndarray
        Token representations.
    W_Q, W_K, W_V : np.ndarray
        Query, key, and value projection matrices.

    Output
    ------
    O : np.ndarray
        Contextualised token representations.
    A : np.ndarray
        Attention weight matrix.
    """
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    d_k = Q.shape[-1]

    scores = Q @ K.T
    scores = scores / np.sqrt(d_k)

    N = X.shape[0]

    mask = np.triu(
        np.full((N, N), -np.inf),
        k=1,
    )

    masked_scores = scores + mask

    A = softmax(masked_scores)
    O = A @ V

    return O, A


def main():
    rng = np.random.default_rng(42)

    N = 4
    d = 6
    d_k = 3
    d_v = 3

    X = rng.normal(size=(N, d))

    W_Q = rng.normal(size=(d, d_k))
    W_K = rng.normal(size=(d, d_k))
    W_V = rng.normal(size=(d, d_v))

    O, A = causal_self_attention(
        X,
        W_Q,
        W_K,
        W_V,
    )

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    print("Input shape:")
    print(X.shape)

    print("\nAttention weights:")
    print(A)

    print("\nRow sums:")
    print(A.sum(axis=1))

    print("\nOutput shape:")
    print(O.shape)


if __name__ == "__main__":
    main()