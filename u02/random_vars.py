import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

vars = np.random.rand(50, 2)

def residual_func(x):
    return x[1] * vars[:,0] + x[0] - vars[:, 1]

def residual_func2(x):
    return x[1] * vars[:, 1] + x[0] - vars[:, 0]

x0 = np.zeros(2)
res1 = so.least_squares(residual_func, x0)
res2 = so.least_squares(residual_func2, x0)

r = np.sqrt(res1.x[1] * res2.x[1])
print(1 - r ** 2)
print(r)