''' Required libraries '''
import numpy as np
import math

class Bayes_classifier1:

    def __init__(self, X, y, proj):
        ''' Initializes the class with the data and classes '''
        self.X = X
        self.y = y
        self.projection_matrix = proj
        self.testx = ""
        self.class_means = self.means()
        self.class_variance = self.variances()

    def means(self): 
        means = dict()
        for i in np.unique(self.y):
            meanfeatures = []
            for feature in range(self.projection_matrix.shape[1]):
                mean=np.mean(self.projection_matrix[self.y==i][:,feature])
                meanfeatures.append(mean)
            means[i] = meanfeatures 
        return means
    
    def variances(self): 
        variances = dict()
        for i in np.unique(self.y):
            variancefeatures = []
            for feature in range(self.projection_matrix.shape[1]):
                variance=np.std(self.projection_matrix[self.y==i][:,feature])
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
    
    def pdf(self,testx): 
        means = self.means()
        variances = self.variances()
        pdfs= dict()
        for i in np.unique(self.y):
            pdfs_features = []
            for feature in range(self.projection_matrix.shape[1]):
                exponent = np.exp((-1/2)*((testx[feature] - means[i][feature])/variances[i][feature])**2)
                pdf =  (1/(variances[i][feature]*math.sqrt(2*math.pi)))*exponent 
                pdfs_features.append(pdf)
            pdfs[i]=pdfs_features 
        return pdfs 
    
    def total_pdf(self,testx): 
        pdfs = self.pdf(testx) 
        total_pdfs = dict()
        for i in np.unique(self.y):
            total_pdfs[i]=math.prod(pdfs[i])
        return total_pdfs 
        
    def sum(self,testx): 
        total_pdfs = self.total_pdf(testx) 
        class_priors = self.class_priors()
        sum = 0
        for i in np.unique(self.y):
            sum += class_priors[i]*total_pdfs[i]
        return sum


    def posterior_prob(self,testx): 
        total_pdfs = self.total_pdf(testx) 
        class_priors = self.class_priors()
        sum = self.sum(testx)
        posterior_prob = dict()
        for i in np.unique(self.y):
            posterior_prob[i]=(class_priors[i]*total_pdfs[i])/sum 
        return posterior_prob

    def prediction(self,testx):
        posterior_prob = self.posterior_prob(testx)
        return max(posterior_prob, key=posterior_prob.get)
    


