import bart_rs
import numpy as np

X = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]])
print(X)
print(X.shape)

y = np.array([20.0, 21.0, 22.3])

result_rs = bart_rs.shape(X)
print(result_rs)

bart_rs.initialize(X, y)
