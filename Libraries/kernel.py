import numpy as np

def K(x, sd=1):
    ''' Returns the Kernel function for value x. '''
    return np.exp(-(x-sd)**2/2) / (sd * np.sqrt(2*np.pi))


def kdf(x, Class_feature, h, sd=1):
    ''' Returns the kernel density estimation for a given point x, array X and bandwidth h. '''
    n = len(Class_feature)
    kdf = []
    for i in Class_feature:
        kernel_density = K((x-i)/h, sd)
        kdf += kernel_density
    print(kdf/h*n)


def looper(x, X, Y, h, sd=1):
    ''' Returns the kernel density estimation for a given dataset, classes and bandwidth h. '''
    kds = []
    for i in np.unique(Y):
        Class = X[Y==i]
        class_tuple = []
        for feature in range(Class.shape[1]):
            class_feature = Class[:,feature]
            kdf(x, class_feature, h, sd)
            class_tuple.append(kdf)
        kds.append(class_tuple)
    return kds


def class_kde(kds):
    ''' Returns the kernel density estimation for a given dataset, classes and bandwidth h. '''
    final = []
    for i in kds:
        class_kde = i[0] * i[1]
        final.append(class_kde)
    return final

