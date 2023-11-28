from sklearn import discriminant_analysis
import numpy as np

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

    def class_means(self):
        class_means = np.array([np.mean(self.X[self.y == i], axis=0) for i in np.unique(self.y)])
        return class_means
    
    def central_point(self):
        class_means = self.class_means()
        central_point = np.mean(class_means, axis=0)
        return central_point
    
    def scatter_within(self):
        class_means = self.class_means()
        scatter_within = np.zeros((self.X.shape[1], self.X.shape[1]))
        for i in range(len(np.unique(self.y))):
            scatter_within += np.dot((self.X[self.y == i] - class_means[i]).T, (self.X[self.y == i] - class_means[i]))
        print(f"Scatter within determinant: {np.linalg.det(scatter_within)}")
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
        return eigenvects
    
    def main_ld(self):
        lin_discs = self.lin_discs()
        main_ld = lin_discs[:,0]
        return main_ld