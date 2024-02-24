# 车道线检测概述

在自动驾驶感知任务中，主要是有两个任务，地面元素检测和地面障碍物检测，其中车道线检测任务是非常重要的，甚至要高于其他地面元素检测任务，这是因为车道线检测是规控的核心，必须针对车道线进行规划才可以更好的按照交规行驶

高精地图主要是成本非常高，并且难以保证实时性（比如说修路），同时在有遮挡的情况下会有偏差，完全依赖于车辆传感器的精度和定位信息，同时成本也高

车道线检测具有实时性好等优点，但是在一些车道线构成复杂的情况下，车道线检测也会难以良好工作，比如说农村道路，重庆复杂道路等

并且下游对车道线检测要求更高，因为一旦检测出错，那么规控就会出错

车道线检测的任务要素如下图所示

![AutoDriveHeart_LaneDetection_L1_6](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L1_6.png)

当然，车道线检测的任务并不局限于车道线本身，也会涉及检测其他的地面标识

在车道线检测任务中，会分为 2D/3D/BEV 等不同层次，而且标注形式也会有线条标注和多边形标注（类似于分割形式），当然多数公司使用的是线条标注，并且还会有标注遮挡与否的情况，不过最大的问题就是在路口的时候标注的复杂性

3D 标注相对容易，4D 标注则严重依赖于定位的准确性，但是投影到 2D 的过程非常依赖内外参，并且也会有遮挡问题

车道线的数据集有如下所示

![AutoDriveHeart_LaneDetection_L1_9](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L1_9.png)

不过，并没有一个可以满足 L3/4 级别的开源数据集，在这方面都是各个公司自行采集，而且 3D 的标注非常依赖于传感器参数，如果参数变化可能会对车道线成像质量造成影响

当然最近还有一个方法叫做高精地图或者说 HD Map，这种方法与实时感知的对比如下

![AutoDriveHeart_LaneDetection_L1_11](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L1_11.png)

## 发展趋势

首先就是整体从高速车道线往市区车道线发展，但是这是非常困难的，在高速上车道线清晰切简单，但是在市区会有复杂、不清晰、车辆行人多和遮挡的情况

![AutoDriveHeart_LaneDetection_L1_12](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L1_12.png)

然后，最开始的车道线是基于 2D 的，但是这种是不足以应用到规控上的，因为规控需要真实世界的车道线坐标，所以后面主要是基于 3D 以及 BEV 等方法进行检测

# 2D 检测

总的来说，二维上的车道线检测有以下特点

![AutoDriveHeart_LaneDetection_L2_6](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_6.png)

## 传统检测

实际上在深度学习出现之前，都是使用图像处理的方法进行检测的，比如说标准的霍夫变换这种

这种方法主要是使用霍夫坐标系和笛卡尔坐标系的转换，一个坐标系下的点就是另一个坐标系下的线，并且如果在笛卡尔坐标系下几个点共线，那么在霍夫坐标系中对应的线就共点

![AutoDriveHeart_LaneDetection_L2_8](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_8.png)

当然还有基于 IPM+投票的方法，投影到 BEV 空间下进行检测，但是这种方法只能检测直线，并且鲁棒性差和检测范围有限

## 分割：Down-top 方法-LaneNet

车道线本质是在做实例分割，我们不但要检测车道线，还有确定 ID，但是常用方案二值化分割是没办法解决的，所以 LaneNet 提出了自底向上的方案，不但预测二值化，还会检测二值化相合平面图，也就是相同大小但是通道更多的一个 Embedding 图，其中二值图确定前景点和背景点，Embedding 图确定每条线的 ID 信息

![AutoDriveHeart_LaneDetection_L2_10](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_10.png)

然后为了确定车道线的 ID，使用了 push-pull Loss，其训练方法就是缩小类内距离，扩大类间距离，如下图所示，具体操作就是将所有的 ID 相同的 GT 分为一个个簇，然后取一个簇的均值为类别中心，然后求距离，然后缩小这种距离，并且缩小到一个阈值之后结束，扩大类间距离的方法就是扩大两个簇之间所有点的距离并且也有一个阈值

![AutoDriveHeart_LaneDetection_L2_11](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_11.png)

那么在推理的时候，首先根据二值图把 Embedding 中所有可能是前景点的取出来，会遍历所有的点计算距离，如果小于某一个阈值就粗浅认为是一个类，遍历几遍就可以得到最终的聚类结果，但是聚类所花费的时间就会很久，并且没有利用车道线细长的几何特点

当然，还有一个 H Net被提出，因为车道线在二维上往往近大远小，不会是横平竖直的，所以就需要去拟合，并且有时候难以直接在二维上进行拟合，所以会先把检测结果投影到 IPM 的 BEV 图上，这时候往往会是平行的，然后去拟合即可，但是 IPM 方法是依赖于透视变换完成的，这种方法假设地面是平面，所以在道路不平的情况下表现并不好

![AutoDriveHeart_LaneDetection_L2_12](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_12.png)

作者提出了一个求解 IPM 矩阵的预测方法，如果原始二维图像预测点左（上图右下角第一张图）乘矩阵 H 投影到 IPM 坐标系下（第二张图像），然后做一个曲线方程拟合得到红色点（第三张），然后乘以 H 的逆阵返回真实二维图像中拟合的点，所以 H 是为了更方便的进行拟合，适应道路不平的情况，对 H 的监督学习损失就是使用均方损失

## GA-Net

这是 CVPR2022 的工作，并且排名非常高，同时实时性高，并且与 LaneNet 异曲同工，预测两个特征图，其中 confidence Map 做前景点分割，Offsets Map 取代 Embedding 做聚类并且不需要后聚类，然后使用 LFA 做聚合

"Offset"通常指的是一个元素（像素、特征点等）相对于其原始位置的位移。在车道线检测等任务中，"Offset Map"是一种常见的表示形式，它为图像中的每个像素点提供一个偏移量，这个偏移量指示了该点应该移动多远以对齐到某个特定的结构（如车道线）。

![AutoDriveHeart_LaneDetection_L2_13](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_13.png)

首先对于 confidence Map，GANet 使用了类似 CenterNet 的关键点估计方法（预测中心点并且给出高斯分布），相比 LaneNet 更加细粒度化，把一条线上的点进行离散化，进行高斯核卷积操作，并且做了方差等参数上人为的先验设计，并且使用一个量化损失衡量下采样的损失，去使网络学习如何预测前景点，强化回归预测的效果

![AutoDriveHeart_LaneDetection_L2_14](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_14.png)

然后在聚类判断 ID 上，使用了 Offset 取代 Embedding 方法，其预测了车道线的起始点，每个起始点对应了一个车道线，一个车道线上所有的点都要计算与其起始点的距离，这样的好处就是起始点距离较近，较为明显，更容易区分，同时，预测起始点的用处就是，可以使用一个损失函数衡量这个距离的话，这个车道线上的点就很容易聚类到一起，这是一个很巧妙的思路，并且不需要做迭代，只需要计算哪些点的 Offset 小于阈值（比如说1），然后就可以找到起始点（只有起始点的距离小于1）

![AutoDriveHeart_LaneDetection_L2_15](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_15.png)

训练的时候加入了所有点及其起始点之间的 Offset，测试的时候就先判断哪些点的 Offset 小于1，就可以判断有几个车道线，然后进行聚类来判断 ID，这样的后处理很简单

当然，还有 SA 模块增加全局感受野以及 LFA 模块进行细长化卷积或者说可变形卷积，充分利用车道线细长的信息，实际上可变形卷积会计算一个点的几个 Offset，并且有可学习的偏移量，使得卷积核可以自适应地调整其形状以更好地适应输入特征的几何变化。这种机制使得可变形卷积能够更有效地处理图像中的形状变化和不规则物体，从而提高模型的表现力和准确性。

先预测位置偏差，然后根据位置去完成卷积，决定往哪边进行卷积，并且加入监督学习，在车道线前景点下采样的时候应该从周围的前景点进行采样（下图所示），具体偏移量的计算中，会先做一个匈牙利匹配，决定哪些点最应该进行匹配，这个 LSA 模块也是一个核心点

![AutoDriveHeart_LaneDetection_L2_17](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_17.png)

在 PyTorch 中，可变形卷积可以通过 `torchvision` 库中的 `DeformConv2d` 类实现：

```python
class OffsetPredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OffsetPredictor, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1)

    def forward(self, x):
        offset = self.conv(x)
        return offset

# 示例
# 假设输入特征图的通道数为256，输出的偏移量为2维（水平和垂直方向）
model = OffsetPredictor(in_channels=256, out_channels=2)

# 输入特征图
input_feature = torch.randn(1, 256, 64, 64)  # batch_size, channels, height, width

# 预测偏移量
predicted_offset = model(input_feature)
```



## row-wise方法-UltraFastLane

充分利用了车道线自底向上且单调的性质，比如说每个车道线在每个行上只有一个前景点，当然这种方法在工程上有局限性，比如说在转弯处未必只有一个前景点

这种方法的思想就是设置行 Anchor 然后按列分类，也就是分为若干行，每个行做若干分类（会加一个空分类，表示无车道线）

![AutoDriveHeart_LaneDetection_L2_18](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_18.png)

网络架构如下图所示，其中骨干网络使用 ResNet，辅助监督部分使用语义分割，不过这个部分只是为了帮助收敛，只会在训练的时候起作用，推理的时候会砍掉，然后使用全连接层和 Reshape 来进行分类（使用 FC 层强化全局感受野），总的来说，这个网络非常轻量化，一直在进行下采样

![AutoDriveHeart_LaneDetection_L2_19](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_19.png)

在分类部分，会把特征图拉平，然后进行全连接层处理，因为全连接层有比较强的全局感受野，并且在这里 FC 层会扩充一个通道维度，表示空 Anchor

不过在 V1 版本中，通道维度是固定的，而不是根据车道线数量确定的，在 V2 中则是使用一些 Anchor ，并且会根据数据集确定通道维度

而且输出只会是一些关键点，因为在下游任务中会进行车道线拟合，而且有限数量的关键点就可以使用了

![AutoDriveHeart_LaneDetection_L2_20](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_20.png)

row-wise 的损失是分类损失，每一行按列进行分类任务然后计算损失

还有一个损失为 Shape Loss，作者认为车道线是连续的，两行的同一个车道线的关键点是相邻的，所以会计算这种损失，但是极限情况下会有损失为零的情况，所以权重不会很大，可以防止车道线跑偏，当然也可以使用二阶差分的损失

但是这种操作需要取出位置，而且这种操作是很难反向传播的，所以就将位置进行编码，计算位置的期望代替直接差分操作，来使得两行中前景点的位置接近

## CondLaneNet

这是一个更复杂的 row-wise 方法，cond 是 conditional 的意思，是借鉴了 conditional instance segment 方法，就是在卷积过程中用一个动态向量代替 instance 进行卷积

网络中骨干网络使用 ResNet + Transformer Encoder 叠加使用，强化感受野和特征提取能力

Proposal Head 部分先做起始点的预测（下图中间图的白点），认为有多少起始点就有多少车道线，然后把起始点所在的特征向量取出，代表了车道线的核心特征，也是此部分的 Proposal Map，这些特征和取出的 Feature Map 进行动态卷积或者说求相似度，然后得到若干新特征图，新特征图代表了车道线的分割，这种方法参考了论文 SOLO 和 CondInstance

![AutoDriveHeart_LaneDetection_L2_23](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_23.png)

其中，在 Proposal 部分的 RIM 模块，是进行分叉预测的，也是 row-wise 的方法

其 row-wise 机制是这样的，与 FC 层预测是否有车道线或者车道线类别不同，这里不会增加通道维度，而是使用线性层进行压缩操作，将 HxW 的特征图压缩为 Hx1 的向量，去判断每一行是否有车道线，相当于去进行二分类，然后此处使用二分类损失进行学习，这里称为 Vertical Range Loss

![AutoDriveHeart_LaneDetection_L2_24](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_24.png)

然后此工作中还有一个 row-wise 的损失，类似于之前的 Shape Loss，对每一行进行 softmax，然后对位置进行加权，然后和真实位置进行 row-wise Loss，公式如上图右边所示

总的来说，row-wise 的预测并不是很准确的，因为经历多次下采样之后，特征图的尺寸会减小，所以需要使用 Offset 损失来学习预测偏移量，弥补下采样的损失

然后起始点的预测也有显式监督学习，使用 Focal Loss，借鉴 CenterNet 中的 Center 概念，使用高斯分布等预测起始点

![AutoDriveHeart_LaneDetection_L2_25](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_25.png)

然后就是此工作的特殊结构，也就是 RIM 部分，这个部分怎么预测车道线是否分叉的呢？这里借鉴 LSTM 结构，随机初始化 $C$ 和 $h$ ,然后使用全连接层得到 $s$ 和 $k$ 两个特征，前者表示状态，为零的时候分叉结束，比如说一开始就为 0 则表示没有分叉，经过三个单元变为 0 则表示有三个分叉，后者则是真正的动态卷积核，有几个分叉就有几个 $k$ 进行动态卷积，而 $f$ 只是一个中间的特征；然后这里使用一个损失进行监督学习

不过注意一下，大多数数据集并不存在分叉的情况，只对 CO-Lane 数据集开放这个模块

![AutoDriveHeart_LaneDetection_L2_26](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_26.png)

不过在工程中使用 row-wise 方法不多，因为在方案在掉头拐弯等情况下会失效或者走形

## 端到端方法- PolyLaneNet

端到端指的是视觉上的端到端，在二维车道线检测中，我们检测的是车道线检测结果，但是在使用的时候要使用拟合好的车道线曲线或者直线来进行规控，或者说我们要把检测的车道线进行拟合才可以进行应用

PolyLaneNet 是一个早起的端到端工作，思路是直接预测系数，网络结构很标准，提取特征并且展平，然后预测固定数量的车道线还有各自的起止点，而且因为发表时间早，所用的数据集也较为原始并且只有标准五根车道线，然后会有一个阈值，超过阈值的才会进行输出

![AutoDriveHeart_LaneDetection_L2_27](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_27.png)

总的来说，这个方法简单粗暴直接，输出是固定数量，而且没有利用车道线的性质

## LSTR

这个方法基础结构也是 PolyLaneNet，只不过进行了修改，进行了升级，预测非固定数量的车道线，并且丰富了车道线带带表达形式，前面我们只使用了多项式曲线进行拟合，但是对于一些特殊情况，比如说车道线的弯曲，单值函数就无法进行拟合了，在这里作者提出了一种新的形式去拟合，这种形式不是简单的纵坐标因变量、横坐标自变量的表达

![AutoDriveHeart_LaneDetection_L2_28](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_28.png)

这种方法实际上是按照小孔成像的原理然后进行数学推理，并且有很强的实用主义，但是这种思路并不建议，因为后面的 BEV 方法可以更好的替代这种小孔成像的方案

![AutoDriveHeart_LaneDetection_L2_29](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_29.png)

这种方法是预测了固定数量的车道线，比如说预测 20 条（这个数量多于数据集中的车道线数量），那就使用匈牙利匹配来完成预测和标签的配对并且计算 Loss，这里会使用距离函数来判断哪些应该进行匹配

## Anchor-based 方案——LineCNN



![AutoDriveHeart_LaneDetection_L2_30](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/AutoDriveHeart_LaneDetection_L2_30.png)