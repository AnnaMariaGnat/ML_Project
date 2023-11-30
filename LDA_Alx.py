import numpy as np
import scipy.linalg as la

class LDA_AJP:
    def __init__(self, X, y):
        """
        Initializes the LDA class with data and class labels.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,) representing class labels
        """
        self._validate_input(X, y)
        self.X = X
        self.y = y
        self.class_means = self._compute_class_means()
        self.central_point = self._compute_central_point()

    def _validate_input(self, X, y):
        """
        Validates the input data and labels.
        Ensures X is a 2D array and y is a 1D array of the same length.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be the same.")

    def _compute_class_means(self):
        """
        Computes the mean of each class in the dataset.
        """
        return np.array([np.mean(self.X[self.y == class_label], axis=0) for class_label in np.unique(self.y)])

    def _compute_central_point(self):
        """
        Computes the central point (mean of class means) of the dataset.
        """
        return np.mean(self.class_means, axis=0)

    def _compute_scatter_within(self):
        """
        Computes the within-class scatter matrix.
        """
        return sum([np.cov(self.X[self.y == class_label].T, bias=True) * np.sum(self.y == class_label)
                    for class_label in np.unique(self.y)])

    def _compute_scatter_between(self):
        """
        Computes the between-class scatter matrix.
        """
        return sum([np.sum(self.y == class_label) * np.outer(self.class_means[class_label] - self.central_point,
                                                             self.class_means[class_label] - self.central_point)
                    for class_label in np.unique(self.y)])

    def _compute_lda_matrix(self):
        """
        Computes the LDA matrix by combining the within and between class scatter matrices.
        """
        s_w = self._compute_scatter_within()
        s_b = self._compute_scatter_between()
        s_w_inv = la.pinv(s_w)
        return np.dot(s_w_inv, s_b)

    def compute_linear_discriminants(self):
        """
        Computes the linear discriminants (eigenvalues and eigenvectors) of the LDA matrix.
        """
        lda_matrix = self._compute_lda_matrix()
        eigvals, eigvects = la.eigh(lda_matrix)
        idx = eigvals.argsort()[::-1]
        return eigvals[idx], eigvects[:, idx]

    def project_data(self, n_discriminants=2):
        """
        Projects the data onto the specified number of top linear discriminants.
        
        Parameters:
        - n_discriminants: number of top linear discriminants to project onto
        """
        if n_discriminants > len(np.unique(self.y)) - 1:
            raise ValueError("n_discriminants cannot be greater than the number of classes minus one.")
        eigenvals, eigenvects = self.compute_linear_discriminants()
        return np.dot(eigenvects[:, :n_discriminants].T, self.X.T).T

    def main_linear_discriminants(self, n_discriminants=2):
        """
        Extracts the specified number of main linear discriminants (eigenvectors).
        
        Parameters:
        - n_discriminants: number of top linear discriminants to return
        """
        _, eigenvects = self.compute_linear_discriminants()
        return eigenvects[:, :n_discriminants]