import unittest
import numpy as np
import weakref
import contextlib
import math
from dezero import Function
from dezero import plot_dot_graph

class Variable:
    def __init__(self, data, name = None):
        # 仅支持ndarray
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.name = name
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # 用于确定反向传播的顺序

    # 返回变量形状，且让方法像变量的属性一样易用
    @property
    def shape(self):
        return self.data.shape
    
    # 维度数
    @property
    def ndim(self):
        return self.data.ndim
    
    # 元素个数
    @property
    def size(self):
        return self.data.size

    # 数据类型
    @property
    def dtype(self):
        return self.data.dtype
    
    # 使得len(variable)可用
    def __len__(self):
        return len(self.data)
    
    # 使得print()能用于variable
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 8) # 保证多行输出时对齐
        return 'variable(' + p + ')'

    # 运算符重载
    # def __mul__(self, other):
    #     return mul(self, other)

    # 设置运算符优先级
    __array_priority__ = 200

    # 设定生成函数
    def set_creator(self, func):
        self.creator = func
        self.generation = self.creator.generation + 1

    # 导数归零
    def cleargrad(self):
        self.grad = None

    # 反向传播
    def backward(self, retain_grad=False): # 若为True则保留中间变量的导数
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
            # 清除中间变量的导数
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y是弱引用
           


# ===========================================
# 配置类
class Config:
    enable_backprop = True # 实现模式切换，只有在训练模式中才需要记录中间变量和变量与函数之间的生成关系

@contextlib.contextmanager
def using_config(name, value):
    # 进入函数后将enable_backprop设置为value
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    # 在with作用域内执行自己想要执行的操作
    try: 
        yield
    # 离开作用域时将属性复原
    finally:
        setattr(Config, name, old_value)

# 封装模式切换 不存储中间变量导数
def no_grad():
    return using_config('enable_backprop', False)


# ===========================================
# 函数类
class Function:
    def __call__(self, *inputs): # 接受一个可变长参数
        inputs = [as_variable(x) for x in inputs] # 确保输入均为Variable实例
        # 取出数据
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple): # 对于非元组的输出，打包为元组
            ys = (ys,)
        # as_array防止出现不支持的数据类型
        outputs = [Variable(as_array(y)) for y in ys]
        # 只有在训练模式中才会记录元素之间的连接关系
        if Config.enable_backprop:
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

# ===========================================
# 运算函数
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

# 负数
class Neg(Function):
    def forward(self, x):
        return -x

    def bacward(self, gy):
        return -gy

# 减法
class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy
# 除法
class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)

# 乘法
class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gys):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gys * x1, gys * x0

# 幂运算
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

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
    x1 = as_array(x1)
    return Add()(x0, x1)

# 简单的运算符重载方法：
Variable.__add__ = add
Variable.__radd__ = add # 右加运算符重载，对于加法交换两个操作数结果一样

def neg(x):
    return Neg()(x)

Variable.__neg__ = neg

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

Variable.__sub__ = sub

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) # 交换顺序

Variable.__rsub__ = rsub

def pow(x, c):
    return Pow(c)(x)

Variable.__pow__ = pow

def exp(x):
    return Exp()(x)

def square(x):
    return Square()(x)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)
# 简单的运算符重载方法：
Variable.__mul__ = mul
Variable.__rmul__ = mul # 右乘运算符重载，对于乘法交换两个操作数结果一样

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001): # threshold用于精度控制
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i +1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold: # 若余项小于threshold则退出
            break
    return y



# ===========================================
# 工具函数
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

# 将标量转化为numpy数组
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# ===========================================
# 测试用例
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

x = Variable(np.array(np.pi/4))
y = my_sin(x,threshold=1e-150)
# y = my_sin(x)
y.backward()
print(y.data)
print(x.grad)
plot_dot_graph(y, verbose=False, to_file='my_sin.png')

