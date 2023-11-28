from sklearn import discriminant_analysis

class lda_class:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def lda(self):
        lda = discriminant_analysis.LinearDiscriminantAnalysis()
        lda.fit(self.X, self.y)
        return lda