import unittest
import numpy as np


class Variable:
    def __init__(self, data):
        # 仅支持ndarray
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    # 设定生成函数
    def set_creator(self, func):
        self.creator = func

    # 反向传播
    def backward(self):
        # 若对该变量调用反向传播时，导数仍为空，则初始化导数为1.0（说明是反向传播起始点）
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 取出函数
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            # 只要x还有创建者，将其创建者添加到反向传播的函数列表
            if x.creator is not None:
                funcs.append(x.creator)


# 将标量转化为numpy数组
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y)) # 防止出现不支持的数据类型
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy

def exp(x):
    return Exp()(x)

def square(x):
    return Square()(x)


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    # 梯度检验
    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) # 生成一个随机的输入
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad) # 两个值是否接近
        self.assertTrue(flg)

# 程序测试入口
unittest.main()