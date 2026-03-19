import numpy as np

def center_kernel_alignment(x, y):
    """
    Linear Centered Kernel Alignment (CKA) between two activation matrices.
    
    Args:
        x (np.ndarray): Activation matrix of shape (batch, features_x).
        y (np.ndarray): Activation matrix of shape (batch, features_y).
        
    Returns:
        float: CKA similarity in [0, 1].
    """
    # x, y: (batch, features)
    def gram_linear(x):
        return x @ x.T

    def center(k):
        n = k.shape[0]
        one_n = np.ones((n, n)) / n
        return k - one_n @ k - k @ one_n + one_n @ k @ one_n

    def hsic(k, l):
        return np.sum(center(k) * center(l))

    k = gram_linear(x)
    l = gram_linear(y)
    
    hsic_kl = hsic(k, l)
    hsic_kk = hsic(k, k)
    hsic_ll = hsic(l, l)
    
    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)

def ssvca_proxy(x, y, n_components=10):
    """Simple SSVCA proxy: average cosine similarity between top PCA components."""
    from sklearn.decomposition import PCA
    
    # x, y: (batch, features)
    # PCA to reduce dimensionality for stability
    pca_x = PCA(n_components=min(n_components, x.shape[0], x.shape[1])).fit(x)
    pca_y = PCA(n_components=min(n_components, y.shape[0], y.shape[1])).fit(y)
    
    comp_x = pca_x.components_
    comp_y = pca_y.components_
    
    # Compute similarity between subspaces
    # Canonical correlation proxy: sum of singular values of (comp_x @ comp_y.T)
    s = np.linalg.svd(comp_x @ comp_y.T, compute_uv=False)
    return np.mean(s)
