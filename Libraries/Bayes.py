''' Required libraries '''
import numpy as np
class Bayes_classifier:

    def __init__(self, X, y, proj):
        ''' Initializes the class with the data and classes '''
        self.X = X
        self.y = y
        self.projection_matrix = proj
        self.testx = ""
        self.class_means = self.means()
        self.class_variance = self.variances()

    def means(self): 
        means = dict({})
        for i in np.unique(self.y):
            meanfeatures = []
            for feature in range(self.projection_matrix.shape[1]):
                mean=np.mean(self.projection_matrix[self.y==i][:,feature])
                meanfeatures.append(mean)
            means[i] = meanfeatures 
        return means
    
    def variances(self): 
        variances = dict({})
        for i in np.unique(self.y):
            variancefeatures = []
            for feature in range(self.projection_matrix.shape[1]):
                variance=np.var(self.projection_matrix[self.y==i][:,feature])
                variancefeatures.append(variance)
            variances[i] = variancefeatures 
        return variances
    
    def class_priors(self): 
        class_priors=[]
        n = self.X.shape[0]
        for i in np.unique(self.y):
            class_prior = self.X[self.y==i].shape[0]/n
            class_priors.append(class_prior)
        return class_priors
    
    # def pdf(mean,variance, testx): 

    
