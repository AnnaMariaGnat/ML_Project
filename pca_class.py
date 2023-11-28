# implement PCA from scikit-learn for dimensionality reduction

from sklearn.decomposition import PCA
import sklearn.preprocessing as pre

'''
TODO:
    accept args for different parameters like which scaler to use:
    StandardScaler, MinMaxScaler, etc.
'''

class pca_class:
    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.pca = PCA(n_components = self.n_components)
        self.scaler = pre.StandardScaler() # standardize features by removing the mean and scaling to unit variance
        # self.scaler = pre.MinMaxScaler() # transform features by scaling each feature to a given range
        # self.scaler = pre.RobustScaler() # scale features using statistics that are robust to outliers
        # self.scaler = pre.Normalizer() # normalize samples individually to unit norm

    def fit(self, X):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.pca.fit(X_scaled)

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)