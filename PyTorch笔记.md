# PyTorch概念

在学术领域中，PyTorch 的适用范围远大于 TensorFlow

# 张量基础

PyTorch 中的数据基础就是张量，或者称为多维数组更合适一些，与物理上的张量概念有区别，而 PyTorch 中的张量也就是 Tensor 可以在 GPU 等加速器上运行，从而加快计算速度

可以从矩阵开始了解，学过线性代数的人都知道，矩阵是一个由数字组成的矩形阵列，有行和列两个维度，一个 N 行 M 列的矩阵的形状就是 (N,M)，实际上一个矩阵就是一个二维的张量，有两个维度

那么如何去创建张量呢？在代码中，可以通过多种多样的方式去创建张量对象，如从嵌套列表创建、从 numpy 创建

```python
import torch
data=[[1,2,3],[4,5,6]]
a=torch.Tensor(data)
n=np.array(data)
b=torch.Tensor(n)

//
```

