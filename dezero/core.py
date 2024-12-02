import numpy as np
import weakref
import contextlib
import dezero
import dezero.functions

# ===========================================
# 配置类
class Config:
    enable_backprop = True # 实现模式切换，只有在训练模式中才需要记录中间变量和变量与函数之间的生成关系

# ===========================================
# 配置类
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
# 变量类
class Variable:
        # 设置运算符优先级
    __array_priority__ = 200

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
        p = str(self.data).replace('\n', '\n' + ' ' * 9) # 保证多行输出时对齐
        return 'variable(' + p + ')'

    # 运算符重载
    # def __mul__(self, other):
    #     return mul(self, other)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    
    def transpose(self):
        return dezero.functions.transpose(self)
    
    @property
    def T(self):
        return dezero.functions.transpose(self)
    
    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    # 设定生成函数
    def set_creator(self, func):
        self.creator = func
        self.generation = self.creator.generation + 1

    # 导数归零
    def cleargrad(self):
        self.grad = None

    # 反向传播
    def backward(self, retain_grad=False, create_graph=False): # retain_grad若为True则保留中间变量的导数
        # 若对该变量调用反向传播时，导数仍为空，则初始化导数为1.0（说明是反向传播起始点）
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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

            with using_config('enable_backprop', create_graph):
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
# 函数基类
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
# 基础运算函数类
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

    def backward(self, gy):
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
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)

# 乘法
class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gys):
        x0, x1 = self.inputs
        return gys * x1, gys * x0

# 幂运算
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

# ===========================================
# 函数类

# ===========================================
# 将函数类封装为函数，便于调用
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) # 交换顺序

def pow(x, c):
    return Pow(c)(x)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

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
# 运算符重载
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
