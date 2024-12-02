if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import numpy as np
from dezero.utils import plot_dot_graph

# =========================================
# 优化函数

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

def matyas(x, y): 
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

def goldstein(x, y): 
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) **2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)


