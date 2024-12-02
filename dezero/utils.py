import os
import subprocess


# 将dezero变量转换为dot语言
# verbose表示是否显示变量的形状和数据类型
def _dot_var(v, verbose=False):
    # 生成变量v对应的计算图节点，以dot语言描述
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
            name += str(v.shape) + ' ' + str(v.dtype)
    # 通过id()函数获取python对象的id作为dot节点的唯一id
    return dot_var.format(id(v), name) 

# 将dezero函数转化为dot语言
def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n' # 节点之间的连接关系
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y())) # y一定是weakref
    return txt

# 将计算图转化为dot语言
def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set() # 一个不包含重复对象的集合

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g{\n' + txt + '}'

# 自动执行dot命令的函数
def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    # 生成以dot语言描述的计算图
    dot_graph = get_dot_graph(output, verbose)

    # 将dot数据保存至文件
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    # 如果~/.dezero目录不存在，创建该目录
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)
    
    # 调用dot命令
    extension = os.path.splitext(to_file)[1][1:] # 以graph.png为例，extension=png
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)


