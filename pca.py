# implement PCA from scikit-learn for dimensionality reduction

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class pca_dr:
    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.pca = PCA(n_components = self.n_components)
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.pca.fit(X_scaled)

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)