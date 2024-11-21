import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

vars = np.random.rand(50, 2)

def residual_func(x):
    return x[1] * vars[:,0] + x[0] - vars[:, 1]

def residual_func2(x):
    return x[1] * vars[:, 1] + x[0] - vars[:, 0]

x0 = np.zeros(2)


def as_in_a(filename):
    res1 = so.least_squares(residual_func, x0)
    res2 = so.least_squares(residual_func2, x0)
    plt.scatter(vars[:, 0], vars[:, 1])
    x_coordinat = np.linspace(0, vars[:, 0].max())
    plt.plot(x_coordinat, res1.x[1] * x_coordinat + res1.x[0], label='y=f(x)')
    x_coordinat = np.linspace(0, vars[:, 1].max())
    plt.plot(res2.x[1] * x_coordinat + res2.x[0], x_coordinat, label='x = f(y)')
    plt.legend()
    plt.title('correlation coefficient r = ' + str(np.sqrt(res1.x[1] * res2.x[1])))

    plt.savefig(filename + '.png')
    plt.close()

# task a)
as_in_a('task3_a')

# task b)
vars1 = vars[25:vars[:, 1].size, :]
vars2 = vars[0:25, :]

vars_copy = vars

vars = vars1

as_in_a('task3_b_firsthalf')

vars = vars2

as_in_a('task3_b_secondhalf')

# task c)
vars = vars_copy

vars[1, :] = vars[1, :] * 10
as_in_a('task3_c')
