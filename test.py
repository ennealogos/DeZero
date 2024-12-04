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

# toy dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 权重的初始化
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

# 神经网络的推理
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y
lr = 0.2
iters = 10000
# 神经网络的训练
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:  # 每隔1000次输出一次信息
        print(loss)