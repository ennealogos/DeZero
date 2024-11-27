import unittest
import numpy as np
import weakref

class Variable:
    def __init__(self, data):
        # 仅支持ndarray
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # 用于确定反向传播的顺序

    # 设定生成函数
    def set_creator(self, func):
        self.creator = func
        self.generation = self.creator.generation + 1

    # 导数归零
    def cleargrad(self):
        self.grad = None

    # 反向传播
    def backward(self):
        # 若对该变量调用反向传播时，导数仍为空，则初始化导数为1.0（说明是反向传播起始点）
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation) # 按照generation排序
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop() # 取出函数
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # 将计算得到的导数赋值给对应的输入
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                # 只要x还有创建者，将其创建者添加到反向传播的函数列表
                if x.creator is not None:
                    add_func(x.creator)


# 将标量转化为numpy数组
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs): # 接受一个可变长参数
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple): # 对于非元组的输出，打包为元组
            ys = (ys,)
        # as_array防止出现不支持的数据类型
        outputs = [Variable(as_array(y)) for y in ys]
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x) * gy

# ===========================================
# 将函数类封装为函数，便于调用
def add(x0, x1):
    return Add()(x0, x1)

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

# ===========================================
# 程序测试入口
# unittest.main()

for i in range(10):
    x = Variable(np.random.randn(10000)) # 大量数据
    y = square(square(square(x))) # 进行复杂的计算
