import numpy as np
import math
from ajp import lda_ajp
# This BaseEstimator is only imported to be able to use our Bayes classifier (implemented from scratch) as an estimator in the sklearn cross validation functions
from sklearn.base import BaseEstimator


class Bayes_classifier(BaseEstimator):

    def __init__(self, h):
        self.h=h
        
    
    def fit(self, X, y): 
        self.X = X
        self.y = y
        self.lda = lda_ajp(X,y)
        self.proj_matrix = self.lda.projection_matrix()
        self.n_features = self.proj_matrix.shape[1]
        self.n_observations = self.proj_matrix.shape[0]


    def class_priors(self): 
        class_priors = []
        for c in np.unique(self.y):
            outcome_count = sum(self.y == c)
            class_priors.append(outcome_count /self.n_observations)
        return class_priors
    
    def kernel(self, mean, test_x): 
        kernel = 1/(self.h*math.sqrt(2 * math.pi)) * math.exp(-(test_x-mean)**2/(2*self.h**2))
        return kernel

    def kde(self,test_x):
        kde = []
        for i in np.unique(self.y): 
            Class_data = self.proj_matrix[self.y==i]
            kde_feature = np.array([])
            class_n = Class_data.shape[0]
            for feature in range(self.n_features): 
                kernel_sum = 0 
                for observation in Class_data:
                    input = (test_x[feature]-observation[feature])/self.h
                    kernel_sum += self.kernel(observation[feature], input)
                kde_f = 1/class_n * kernel_sum
                kde_feature =np.append(kde_feature, kde_f)
            kde.append(kde_feature)
        return kde 

    def pdf(self, test_x): 
        pdf = []
        kde = self.kde(test_x)
        for i in range(len(kde)):
            pdf.append(np.prod(kde[i]))
        return pdf 
        
    def posterior_prob(self,test_x):
        class_probs = []
        for i in np.unique(self.y): 
            num = self.class_priors()[i] * self.pdf(test_x)[i]
            denom = sum(np.multiply(self.class_priors(),self.pdf(test_x)))
            class_prob = num / denom
            class_probs.append(class_prob)
        return class_probs

    def classification(self,test_x): 
        probs = self.posterior_prob(test_x)
        prediction = np.argmax(probs)
        return prediction 


    def predict(self, test_data): 
        predictions = []
        new_data = self.lda.transform(test_data)
        print(new_data.T.shape)
        for i in new_data.T:
            prediction = self.classification(i)
            predictions.append(prediction)
        return predictions 


        

    