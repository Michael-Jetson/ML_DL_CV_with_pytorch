# CNN架构

上一节课我们谈到了很多卷积神经网络的部分，但是仍然有一个问题，我们并不知道如何将这些部分组建成一个完整的神经网络来识别图像

这节课的内容就是构建了一整个卷积神经网络来实现图像分类，并且介绍了CNN发展的历史（2012-2019）并且分析了不同的CNN架构

同时，本人会附上PyTorch的代码，大家可以自行学习

## CNN历史：ImageNet挑战赛的发展

## LeNet

这个网络可以说是真正意义上的第一个CNN网络，只不过受限于当时的水平，只能做的很小并且在简单的数据集上进行训练，所以应用面受限，只能用于邮局的手写数字识别

## AlexNet

AlexNet是计算机视觉领域历史性的论文，或者说是深度学习计算机视觉的开创性论文，代表了计算机视觉或计算机科学领域的重要进步

AlexNet的架构如下，当初为了在两个GPU上计算，使用了两个通道，这也是一种工程技巧，后面随着GPU的发展和深度学习框架的出现，我们无需完成那么多的工作量即可轻松复现网络

![](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/AlexNet-Fig_03.png)

第一层是卷积层，卷积核大小为11x11，数量为64，步长为4，填充为2，所以输出大小为64通道，宽高为56

![EECS498_L8_28](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L8_28.png)

![EECS498_L8_31](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L8_31.png)

Alexnet 早期卷积层需要较多的内存开销，且相比全连接层需要更多的浮点数计算；全连接层占用的参数远大于卷积层，且展开后的第一个全连接层需要的参数最多。

不过AlexNet为什么使用两层全连接层，李沐老师的解释是前面卷积层提取的特征不够深，需要两个全连接层来继续深化，如果只使用一个全连接层那么效果会变差

AlexNet的PyTorch代码实现如下

```python
# 导入pytorch库
import torch
# 导入torch.nn模块
from torch import nn
# nn.functional：(一般引入后改名为F)有各种功能组件的函数实现，如：F.conv2d
import torch.nn.functional as F
 
# 定义AlexNet网络模型，继承于父类nn.Module
class AlexNet(nn.Module):
    # 子类继承中重新定义Module类的__init__()和forward()函数
    # init()：进行初始化，申明模型中各层的定义
    def __init__(self):
        # super：引入父类的初始化方法给子类进行初始化
        super(AlexNet, self).__init__()
        # 二维卷积层，输入大小为224*224，输出大小为55*55，输入通道为3，输出为96，卷积核为11，步长为4
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # 使用ReLU作为激活函数，当然也可以使用Sigmoid函数等
        self.ReLU = nn.ReLU()
        # MaxPool2d：最大池化操作
        # 二维最大池化层，输入大小为55*55，输出大小为27*27，输入通道为96，输出为96，池化核为3，步长为2
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为27*27，输出大小为27*27，输入通道为96，输出为256，卷积核为5，扩充边缘为2，步长为1
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 最大池化层，输入大小为27*27，输出大小为13*13，输入通道为256，输出为256，池化核为3，步长为2
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为256，输出为384，卷积核为3，扩充边缘为1，步长为1
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为384，卷积核为3，扩充边缘为1，步长为1
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为256，卷积核为3，扩充边缘为1，步长为1
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 最大池化层，输入大小为13*13，输出大小为6*6，输入通道为256，输出为256，池化核为3，步长为2
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()
        # 全连接层
        # Linear（in_features，out_features）
        # in_features指的是[batch_size, size]中的size,即样本的大小
        # out_features指的是[batch_size，output_size]中的output_size，样本输出的维度大小，也代表了该全连接层的神经元个数
        self.f6 = nn.Linear(6*6*256, 4096)
        self.f7 = nn.Linear(4096, 4096)
        # 全连接层&softmax
        self.f8 = nn.Linear(4096, 1000)
        self.f9 = nn.Linear(1000, 2)
 
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
         # Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
        # “失活的”神经元不再进行前向传播并且不参与反向传播，这个技术减少了复杂的神经元之间的相互影响
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)
        return x
 
# 每个python模块（python文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
# 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
# “__main__” 始终指当前执行模块的名称（包含后缀.py）
# if确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行
if __name__ == '__main__':
    # rand：返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数，此处为四维张量
    x = torch.rand([16, 3, 224, 224])
    # 模型实例化
    model = MyAlexNet()
    y = model(x)

```

这是第一个真正意义上的现代神经网络，并且使用了大量的工程技巧，如对图片进行翻转、裁剪，Dropout正则化等等

![YSAI_ImageClassification_L2_19](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/YSAI_ImageClassification_L2_19.png)

![YSAI_ImageClassification_L2_21](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/YSAI_ImageClassification_L2_21.png)

## ZF Net

![EECS498_L8_34](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L8_34.png)

这个网络实际上就是一个更大的AlexNet，并没有做出太大的创新，只是在某些超参数上进行了调整，思想上与AlexNet一致

ZFNet的第一层使用两倍的下采样因子（步长为2），相比于AlexNet有更大的空间分辨率，也就意味着有更大计算量

从AlexNet到ZFNet的启示就是，更大的网络可以更好的工作，只不过这时候并没有方法可以使得网络变换大小

## VGG网络

VGG网络带来的启示就是，没必要直接使用更大的卷积核，可以使用多个更小的卷积核来代替，比如说一个5x5的卷积核，就可以使用两次3x3卷积来代替，或者说他们发现，同样的计算开销的情况下，使用更深更窄的网络效果更好，或者说两次3x3卷积的效果比一次5x5卷积更好，这样堆更多卷积核的效果更好

![](https://zh-v2.d2l.ai/_images/vgg.svg)

VGG的设计比较标准，或者说架构设计上比较有原则，非常有规则（不像AlexNet那样不规则）

其设计原则是，所有卷积层的卷积核大小都为3x3步幅为1填充为1，所有池化层的都是2x2的最大池化层且步幅为2，池化层之后会将通道数量加倍

VGG不是卷积层堆叠，而是使用块（或者叫卷积块），每个块都是几个卷积层和一个池化层，几个块堆叠起来，后面就是几个全连接层

VGG会使用一些可复用的卷积块来构建网络，不同的卷积块个数和其他超参数可以得到不同复杂度的网络（更方便地设计不同网络），块的构建代码如下，代码来源李沐老师的动手学深度学习

```python
import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```



或许可以在两个卷积层之间添加ReLU层来提供非线性计算，来提高其拟合能力

## NiN：网络中的网络

NiN（Network in Network）是一种由Lin等人在2013年提出的深度卷积神经网络结构。NiN网络的设计旨在增强卷积神经网络的表达能力，并改善模型的非线性建模能力。

NiN网络的核心思想是引入一种称为"Micro Network"的结构，通过使用全连接层的思想来代替传统的卷积层，从而提供更丰富的特征表示。具体而言，NiN网络在卷积层中引入1x1的卷积核，以增加非线性变换和特征提取的能力。这样的1x1卷积核将每个输入通道的特征进行组合，产生多个输出通道的特征图。

此外，NiN网络还引入了全局平均池化层，将每个特征图中的所有元素进行平均，得到一个全局特征向量，用于最后的分类。这样的设计使得网络更加紧凑，并减少了参数数量，降低了过拟合的风险。

NiN网络的优势在于提供了更强的非线性建模能力，同时减少了网络的参数量和计算复杂度。它在图像分类、目标检测和语义分割等任务中取得了良好的效果。NiN网络的设计思想也对后续的深度卷积神经网络的发展起到了一定的启示作用。

## GoogleLeNet

GoogleLeNet是谷歌团队研究出的网络，发表于2014年，这是谷歌团队为了致敬Yann LeCun及其创建的LeNet所命名的，其特点之一是尝试更高效的神经网络，降低其复杂性，使得可以在手机上运行

当然，这个也是第一个真正意义上的深度神经网络，其深度可以超过100层

![EECS498_L8_52](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L8_52.png)

另一个特点就是使用Inception块来代替传统卷积，这个模块是一种重复的局部结构，由四路并行的卷积/池化模块组成,并在最终使用全局平均池化层来折叠空间维度。

四个路径从不同层面抽取信息，然后在输出通道维度进行合并，或者说每个块不会改变高宽，但是会大量增加通道数量，这种方式相比于单种卷积层，参数更少，计算复杂度更低

但是，GoogleLeNet并没有使用批量归一化方法，或者说这时候并没有批量归一化方法，所以这个网络使用的是其他方法来更高效的处理，即输出三种不同的类分数，因为传播通过三个分类器返回的梯度，可以使得反向传播更容易，使得模型更容易收敛

但是在批量归一化出现之前，想训练一个大型的网络是相当困难的，所以有一种解决方法，就是先训练一个浅层网络，然后在其收敛之后插入新的层，继续训练

## 批量归一化

批量归一化是一种很好的技术，可以加快神经网络的收敛速度

我们知道，损失是在网络最后，当进行反向传播的时候，上流的梯度较大，但是下层的梯度会很小，这是因为每一层的梯度可能都不大，反向传播过程中不断累乘会导致梯度越来越小，也就是说，越靠近数据层，梯度更新的越慢，学习的也就越慢，这就导致了每一层更新下层之后，上层也要重新开始训练，所以批量归一化要解决的问题就是在学习底部层的时候避免变化顶部层

实际上，出现上面这种问题的原因是均值和方差的分布会在不同的层之间变化，一个简单的方法就是将分布进行固定，这也就是批量归一化的思想

批量归一化的数学表达式如下，其中$B$是一个批量的样本数据集
$$
\mu_B=\frac{1}{|B|}\sum_{i\in B}x_i\\
\sigma_B^2=\frac{1}{|B|}\sum_{i\in B}(x_i-\mu_B)^2+\epsilon \\
x_{i+1}=\gamma \frac{x_i-\mu_B}{\sigma_B}+\beta
$$
这样就可以学习一个新的均值和方差来实现更好的性能，也就是说其中的变量$\gamma$和$\beta$都是可以学习的，但是会限制变化的范围，而且这是一个线性函数

我们可以将这个操作变成一个层，其作用在全连接层和卷积层的输出上（即激活函数之前）和输入上

## ResNet

2015年ResNet出现，这是具有划时代的意义，因为批量归一化的方法出现（这是一个很有意义的创新），然后神经网络的深度可以大大加深，一年内网络深度从22层增加到152层，在这之前，更深的网络表现的可能比浅层网络更差

## MobileNet

这是一个非常轻量化的网络，可能其准确率并不会高过那些网络，但是轻量化的结构可以使得在计算的时候非常快速，并且在手机登设备上实现这些计算

# 神经网络架构研究

设计一个神经网络架构是困难的，所以需要自动化实现

我们有一个称为控制器的神经网络，这个神经网络将输出另一个神经网络，所以训练过程是我们采取