from lda_lib import lda_ajp
import numpy as np

dataset = np.random.rand(4, 10)

# add one column with labels (0, 0, 1, 2)
labels = np.array([0, 0, 1, 2]).reshape(-1, 1)
dataset = np.hstack((dataset, labels))

X = dataset[:, :-1]
y = dataset[:, -1]

our_thingie = lda_ajp(X, y)

print(our_thingie.lin_discs())