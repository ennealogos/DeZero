# 如果 __file__ 存在于全局变量中，说明当前代码是从文件运行的
if '__file__' in globals():
    import os, sys
    # 如果当前脚本作为文件运行，则将其父目录添加到 Python 的模块搜索路径中。
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import numpy as np
from dezero.utils import plot_dot_graph
import dezero.functions as F
import matplotlib.pyplot as plt

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

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 7
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
# 绘制计算图
gx = x.grad
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=True, to_file='dot_output/step35/tanh_7.png')