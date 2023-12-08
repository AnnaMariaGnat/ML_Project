''' Implementation of different classifiers, dimensionality reduction, and
    feature selection methods (own implementations and from sklearn).
    By Anna Maria Gnat, Josefine Nyeng and Pedro Prazeres
    for the Machine Learning course at ITU CPH. '''



''' Required libraries '''
from sklearn.decomposition import PCA
import sklearn.preprocessing as pre
import numpy as np



class pca_ajp:
    ''' Principal Component Analysis class,
        from sklearn's library '''

    def __init__(self, n_components = 2):
        ''' Initializes the class with the data and classes '''
        self.n_components = n_components # Number of components to keep
        self.pca = PCA(n_components = self.n_components) # Initialize PCA from sklearn
        self.scaler = pre.StandardScaler() # Standardize features by removing the mean and scaling to unit variance

    def fit(self, X):
        ''' Fits the PCA model to the data '''
        self.scaler.fit(X) # Fit the scaler with sklearn
        X_scaled = self.scaler.transform(X) # Scale the data with sklearn
        self.pca.fit(X_scaled) # Fit the PCA model with sklearn

    def transform(self, X):
        ''' Transforms the data with the PCA model '''
        X_scaled = self.scaler.transform(X) # Scale the data with sklearn
        return self.pca.transform(X_scaled) # Transform the data with sklearn
    
    def explained_variance_ratio(self):
        ''' Returns the variance ratio of the components '''
        return self.pca.explained_variance_ratio_
    



class lda_ajp:
    ''' Linear Discriminant Analysis class implemented by 
        Anna Maria Gnat, Josefine Nyeng and Pedro Prazeres
        for the Machine Learning course at ITU CPH '''
    
    def __init__(self, X, y):
        ''' Initializes the class with the data and classes '''
        self.X = X # Data
        self.y = y # Classes
        # Mean of each class (used several times so just initialize it once)
        self.class_means = self.class_means()
        # Central point (mean) of the means of each class (used several times so just initialize it once)
        self.central_point = self.central_point()


    def class_means(self):
        ''' Finds the mean of each class' data points and saves them as a numpy array '''
        class_means = np.array(
            [np.mean(self.X[self.y == i], axis=0) # Mean of all data points in class i...
                     for i in np.unique(self.y)]) # ...for each class in y.
        return class_means


    def central_point(self):
        ''' Finds the central point of the data and saves it as a numpy array '''
        central_point = np.mean(self.class_means, axis=0) # Mean of all class means.
        return central_point


    def scatter_within(self):
        ''' Finds the within-class scatter matrix and saves it as a numpy array '''
        scatter_within = np.zeros((self.X.shape[1], self.X.shape[1])) # Initialize scatter matrix.
        for i in range(len(np.unique(self.y))): # For each class in y...
            scatter_within += np.dot( # ...add the dot product of the difference...
                (self.X[self.y == i] - self.class_means[i]).T, # ...between the data points in the class and the class mean...
                                     (self.X[self.y == i] - self.class_means[i])) # ...and the transpose of the same difference.
        scatter_within = scatter_within / self.X.shape[0] # Divide by the number of data points to scale the matrix.
        return scatter_within


    def scatter_between(self):
        ''' Finds the between-class scatter matrix and saves it as a numpy array '''
        scatter_between = np.zeros((self.X.shape[1], self.X.shape[1])) # Initialize scatter matrix with zeros.
        for i in range(len(np.unique(self.y))): # For each class in y...
            number_observations = self.X[self.y == i].shape[0] # ...find the number of data points in the class...
            scatter_between += number_observations * np.outer( # ...multiply the number of data points with the dot product of...
                (self.class_means[i] - self.central_point).T, # ...the difference between the class mean and the central point...
                (self.class_means[i] - self.central_point)) # ...and the transpose of the same difference.
        scatter_between = scatter_between / self.X.shape[0] # Divide by the number of data points to scale the matrix.
        return scatter_between


    def lda_matrix(self):
        ''' Finds the LDA matrix and saves it as a numpy array '''
        scatter_within = self.scatter_within() # Find the within-class scatter matrix.
        scatter_between = self.scatter_between() # Find the between-class scatter matrix.
        lda_matrix = np.dot( # Find the LDA matrix by multiplying...
            np.linalg.inv(scatter_within), # ...the inverse of the within-class scatter matrix...
            scatter_between) # ...with the between-class scatter matrix.
        return lda_matrix


    def lin_discs(self):
        ''' Finds the linear discriminants and saves them as numpy arrays '''
        lda_matrix = self.lda_matrix() # Find the LDA matrix.
        eigenvals, eigenvects = np.linalg.eig(lda_matrix) # Find the eigenvalues and eigenvectors of the LDA matrix.
        eigenvals = np.real(eigenvals) # Find and remove complex eigenvalues.
        eigenvects = np.real(eigenvects) # Find and remove complex eigenvectors.
        idx = eigenvals.argsort() # Sort the eigenvalues and eigenvectors (ascending order).
        idx = idx[::-1] # Reverse the order of the sorted eigenvalues and eigenvectors (descending order).
        eigenvals = eigenvals[idx] # Sort the eigenvalues according to the sorted indices.
        eigenvects = eigenvects[:,idx] # Sort the eigenvectors according to the sorted indices.
        return eigenvals, eigenvects


    def projection_matrix(self, n_ld=2):
        ''' Creates a matrix of the projected data with n linear discriminants and saves it as a numpy array '''
        top_eigenvects = self.main_lds(n_ld) # Find the main linear discriminants.
        X_lda = np.dot(top_eigenvects.T, self.X.T) # Project the data onto the main linear discriminants.
        X_lda = X_lda.T # Transpose the data to get the correct shape.
        return X_lda


    def main_lds(self, n_ld=2):
        ''' Returns the n main linear discriminants as numpy arrays '''
        _, vects = self.lin_discs() # Find the eigenvectors of the LDA matrix.
        main_lds = vects[:, :n_ld] # Find the main linear discriminants by taking the first n_ld eigenvectors.
        return main_lds