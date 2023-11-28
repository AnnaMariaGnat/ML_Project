from sklearn import discriminant_analysis
import numpy as np
import sys

class lda_skl:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def lda(self):
        lda = discriminant_analysis.LinearDiscriminantAnalysis()
        lda.fit(self.X, self.y)
        return lda
    

class lda_ajp:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.normalized_scatter_within = None

    def class_means(self):
        class_means = np.array([np.mean(self.X[self.y == i], axis=0) for i in np.unique(self.y)])
        return class_means
    
    def central_point(self):
        class_means = self.class_means()
        central_point = np.mean(class_means, axis=0)
        return central_point

    def normalized_det(self, matrix, min, max):
        print("Normalizing matrix and trying again...")
        normalized_matrix = (matrix - min) / (max - min)
        try:
            determinant = np.linalg.det(normalized_matrix)
            print("Determinant:", determinant)
            self.normalized_scatter_within = normalized_matrix
            return determinant
        except Exception as e:
            print(f"Determinant could not be computed due to following exception: {e}")
            return None

    def scatter_det(self, matrix=None):
        if matrix is None:
            matrix = self.scatter_within()
        max_value = np.max(matrix)
        min_value = np.min(matrix)
        if max_value > sys.float_info.max or min_value < sys.float_info.min:
            print("Matrix contains extreme values, attempting to compute determinant:")
            try:
                determinant = np.linalg.det(matrix)
                if determinant == 0:
                    print("Matrix is singular and cannot be inverted (determinant is 0)")
                    determinant = self.normalized_det(matrix, min_value, max_value)
                elif determinant == np.inf:
                    print("Matrix is singular and cannot be inverted (determinant is inf)")
                    determinant = self.normalized_det(matrix, min_value, max_value)
                else:
                    print("Determinant:", determinant)
            except Exception as e:
                print(f"Determinant could not be computed due to following exception: {e}")
                determinant = self.normalized_det(matrix, min_value, max_value)
        else:
            determinant = np.linalg.det(matrix)
        return determinant


    def scatter_within(self):
        class_means = self.class_means()
        if self.normalized_scatter_within is not None:
            return self.normalized_scatter_within
        scatter_within = np.zeros((self.X.shape[1], self.X.shape[1]))
        for i in range(len(np.unique(self.y))):
            scatter_within += np.dot((self.X[self.y == i] - class_means[i]).T, (self.X[self.y == i] - class_means[i]))
        return scatter_within


    def scatter_between(self):
        class_means = self.class_means()
        central_point = self.central_point()
        scatter_between = np.zeros((self.X.shape[1], self.X.shape[1]))
        for i in range(len(np.unique(self.y))):
            number_observations = self.X[self.y == i].shape[0]
            scatter_between += number_observations * np.dot((class_means[i] - central_point).T, (class_means[i] - central_point))
        return scatter_between
    
    def lda_matrix(self):
        scatter_within = self.scatter_within()
        scatter_between = self.scatter_between()
        lda_matrix = np.dot(np.linalg.inv(scatter_within), scatter_between)
        return lda_matrix
    
    def lin_discs(self):
        lda_matrix = self.lda_matrix()
        eigenvals, eigenvects = np.linalg.eig(lda_matrix)
        idx = eigenvals.argsort()
        idx = idx[::-1]
        eigenvals = eigenvals[idx]
        eigenvects = eigenvects[:,idx]
        return eigenvals, eigenvects
    
    def main_lds(self, n_ld=1):
        vals, vects = self.lin_discs()
        main_lds = vects[:, :n_ld]
        return main_lds