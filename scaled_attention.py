"""CS5720 Home Assignment 4, Part II Q3: scaled dot-product attention."""
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = weights @ V
    return weights, output

if __name__ == '__main__':
    Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    weights, output = scaled_dot_product_attention(Q, K, V)
    print('Attention weights:\n', np.round(weights, 3))
    print('Final output:\n', np.round(output, 3))
