# 三维场景理解

我们这里介绍如何使用点云深度学习网络完成**三维场景理解（3D Scene Understanding）**，可以说PointNet等网络的发展，让我们可以更好的完成三维场景理解，而不是单纯的通过图像去理解

这里主要有两个主题，三维目标检测（物体识别）和三维场估计

![72](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_88.jpg)

与二维的视觉理解（包括分类、检测、分割等）一致，想对三维场景进行理解，也要先进行分类，然后再进行分割和检测，同时，因为实际上三维场景理解通常会使用多种传感器一起进行感知，所以任务会负责许多

## 基于点云的分类任务

分类网络中最经典的就是PointNet系列，可以实现直接基于点集的学习，可以说是三维点云深度学习的开山之作了

当然，现在Point-based已经有了很多工作，主要有pointwise MLP、convolution-based、graph-based、hierarchical data structure-based等方式

- pointwise MLP：这种方式主要是依靠若干共享的MLP，独立地对单个点进行处理，并且使用对称函数生成一个全局特征，如PointNet，但是这种方式很难学习到局部结构信息
- convolution-based：

### 处理点云输入：PointNet

我们如何去构建一种可以处理点云的神经网络模型呢？答案就是PointNet，这是一种常用的处理点云的神经网络模型，PointNet的关键创新之处在于它能够直接从原始点云数据中学习特征，而无需转换成其他类型的数据表示，这是一种端到端的学习方法，在此之前都是使用手工设计特征和点云转化为规则的体素格式

当然，PointNet实际上并不是一种卷积神经网络，因为此网络中没有实现卷积操作的层

这个经典网络的作者是斯坦福大学的祁芮中台博士，其在B站有过对此网络的讲解

[点云上的深度学习及其在三维场景理解中的应用](https://www.bilibili.com/video/BV1As411377S/?spm_id_from=333.337.search-card.all.click&vd_source=eea47a16439992e41b232bc5d5684e27)

#### 设计思想

PointNet的主要设计思想是设计一个对点云的置换（permutation）和变换（transformation）具有不变性的网络。这意味着无论点云中的点的顺序如何改变，或者如何旋转、平移、缩放点云，PointNet的输出都应该保持一致。

PointNet的网络结构主要由两部分组成：

1. 输入转换网络：这部分的作用是学习一个对齐网络，用于将输入的点云对齐到一个规范化的坐标系统。它包括几个MLP（多层感知机）层和一个max pooling层，最后输出一个转换矩阵，这个矩阵会被应用到输入的点云上，以此解决点云的平移旋转不变性。
2. 特征提取网络：这部分的作用是从对齐后的点云中提取全局特征。它也包括几个MLP层（每一层MLP都是权重共享的）和一个max pooling层。max pooling层在所有的点上取最大值，从而实现了对点的置换不变性。

![9](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_ShenlanOpen_LiuYongcheng_10.png)

这里的公式里面有两个函数，对称函数$g$和权重共享的变换函数$h$，其中前者是最大池化实现，后者是MLP实现

PointNet可以进行点分类，也可以进行点分割。对于点分类，网络最后输出的是一个全局特征，表示整个点云的分类。对于点分割，网络会为每一个输入点输出一个特征，表示这个点的分类。

#### 工作原理

PointNet具体是怎么工作的呢？我们想输入一组点云，假设有P个点，每个点都有一个三维空间的XYZ位置信息

![32](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/EECS498_L17_33.jpg)

我们想做的第一件事就是对形状进行分类，当然我们不希望点在点云中的顺序很重要，或者说我们不希望点在点云中的存储顺序会影响分类结果，实际上存储结构的确也不会有所影响，就有点类似于之前的Transformer结构

首先我们对每个点独立运行一个MLP（这个是共享权重的，也就是前面提到的函数$h$），然后输出D维的特征向量，一个有P个点的点云就可以得到PxD维的特征矩阵，然后我们使用最大池化方法去提取特征，得到一个统一的D维特征向量，然后使用全连接层进行分类，这是因为最大池化函数并不关心输入张量上点的表示顺序，所以很适合处理这些点云

实际上，PointNet是对关键点进行的学习，或者说，输入一系列的点，PointNet会从中选择一些对分类最关键的点（或者说特征值最大的点）去进行分类（会反推，寻找对分类影响最大的点），这样就是对点云进行总结

![10](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_ShenlanOpen_LiuYongcheng_11.png)

由此我们可以推出其上界（Upper-bound），我们直接使用一个立方体，将其中的所有离散点送入网络，只要这个点对应的Max Pool的特征不大于对应的形状的特征，我们就认为这个点对于点云的识别是没有影响的，这样就可以得到上界的形状（上图左侧所示，第二行为关键点的集合，第三行为上界形状，两种点云分类结果一致），或者说，只要噪声点在这个上界范围内，就不会对分类结果产生影响，说明其具有一定鲁棒性

PointNet虽然简单，但却非常强大，它已经在很多3D视觉任务上取得了很好的效果，如物体分类、语义分割和部分分割等。但是，PointNet也有一些局限性，比如它不能很好地捕捉局部结构信息和点之间的关系，为此后来的研究者也提出了很多改进版本，如PointNet++等。

接下来我们详细分析这个网络各个部分的作用和原理

#### 不变性

首先我们看一下点云的输入，一组点云有N个点，每个点有D个特征（最基本的有左边，还有法向量等），这样就使用一个NxD矩阵表示，因为点云的无序性，那么任意置换行，不会影响其中信息，或者说矩阵虽然不同，但是代表的点集一样

![16](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_27.jpg)

我们知道，神经网络本质上就是一个函数，所以需要设计一个具有置换不变性的网络，这样就可以处理点云了，我们就需要一个**对称函数（Symmetric Function）**，也就是具备以下性质的函数
$$
f(x_1,x_2,\cdots,x_n)\equiv f(x_{\pi_1},x_{\pi_2},\cdots,x_{\pi_n}),x_i\in \mathbb{R}^D
$$
类似的函数有求和函数、最大值函数、平均值函数（这些是最简单的形式），核心的性质就是输入的顺序对结果不构成影响

当然，这些方法也有问题，比如说我们直接对每个维度依次取最大值，比如说输入一组点云（下图所示），第一个维度最大是2，第二个维度是2，第三个维度是4，或者平均值来求重心，但是这种方法会丢失很多几何信息，所以我们不能直接在数据集上应用对称函数

![18](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_31.jpg)

我们可以使用一个高维空间映射，比如说使用一个一千维的向量来表示一个点，这样信息肯定是冗余的，然后我们就可以使用另一个网络$y$来消化这些信息，这样，只要函数$g$是对称的，那么这个结构就是对称的，试验证明，最大池化是一个比较好的方式

![21](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_34.jpg)

#### 理论分析

我们前面知道了，我们可以使用神经网络构建的对称函数去保证对称，那么这种结构在所有的对称函数中是什么样的一个情况呢？或者说能不能表达任意对称函数呢？

![22](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_35.jpg)

实际上，PointNet是一种**通用逼近器（Universal Approximation）**，可以任意逼近任意一个集合上的对称函数，只要函数在Hausdorff空间中是连续的，我们可以通过增加网络宽度和深度去任意逼近

![23](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_36.jpg)

#### 输入对齐

实际上，同一个物体可能因为视角不同所以看起来不一样（或者说点云存在几何变换问题），比如说一辆车在不同角度看，那么空间位置就会不同，所以我们需要解决这种变换，方法就是增加一个基于数据的变换函数（或者叫T-Net），下图中的Transform就是一个矩阵乘法，不过点云是一种很容易做几何变换的数据，不想图片变换那样复杂，所以只需要进行一次乘法就可以（或者对齐就是**归一化处理**）

![25](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_38.jpg)

当然，我们不只是在输入的时候进行变换，还可以在中间的时候进行变换

#### 嵌入的空间对齐（Embedding Space Alignment）

我们前面使用输入对齐来处理输入的数据，然后使用对称函数来解决点云的不变性问题，点云在经过这种处理之后变成了一个NxK的矩阵，我们还可以再进行一个特征空间的变换，再使用一个T-Net生成一个KxK维的变换矩阵

![27](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_42.png)

变换之后我们就得了了另外一组特征，使用网络再进行处理

当然，这是一个高维的空间，优化起来难度相对高，所以我们使用一个正则损失，比如说我们希望这个矩阵接近正交矩阵（如上图所示）

#### 网络结构

网络结构如下

![35](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_50.jpg)

- 网络的输入是一个nx3的矩阵（包括n个点），

- 然后进行对齐（通过T-Net的结构，或者说一次几何变换），也就是输入变换部分，
- 进行MLP点云处理（对每个点进行升维），维度增加到64，得到nx64维的特征
- 基于高维空间再进行一个T-Net的结构进行特征变换和对齐，转化到一个更归一化的特征空间（64维空间）
- 然后再使用MLP进行升维，这里使用三个MLP，分别将维度变换到64、128和1024维，然后就得到一个nx1024的一个特征
- 在每个维度上进行max pooling操作（对称性操作），就得到了一个1024维的**全局特征（Global Feature）**，然后再使用三个MLP进行任务的处理，分类结果是k种

如果我们想完成点云分割任务

![38](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_53.jpg)

基于全局坐标是没办法对每个点进行分割的，所以一个简单有效方法就是将局部特征和全局坐标结合起来，相当于单个点在全局特征中进行了一次检索，查看自己在全局特征中的位置

#### 性能指标

这是点云深度学习中最经典最早的工作，跟传统的3D CNN相比性能更好

![40](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_55.jpg)

同时，相比于传统的一些网络，更加轻量化也更加高效，可以用在移动设备上

![44](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_59.jpg)

![45](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_60.jpg)

#### 效果展示

![41](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_56.jpg)

![43](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_58.jpg)

#### 鲁棒性

我们发现，随着点云数量的减少（或者说数据丢失），网络的效果不会下降多少

![PointCloudDL_TechBeatTalk_CharlesQi_61](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_61.png)

那么为什么PointNet对数据丢失这么鲁棒呢？我们先看原始的点云输入

![47](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_63.jpg)

我们想知道哪些点对于全局特征做出来重要贡献，这些称为**关键点（Critical Point）**，我们可以看到，这些关键点表示了物体的几何形状轮廓或者骨骼形状，只要保存了这些关键点，那么就可以分类正确，这就解释了PointNet的鲁棒性

#### 不足：空间复杂度过高

当然，PointNet也有不足，就是占用的空间太多，当有K个点云输入的时候，就会有$K\times C_{in}\times C_{out}$大小的中间权重张量存在，当进行反向传播的时候，会产生巨大的空间占用，所以在2017年PointNet的工作中，只能构建一些小的网络，这也是其问题所在

![20](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_ShenlanOpen_ConvNet-On-PointCloud_LiFuxin_20.png)

李伏欣在2019年论文中的工作是，发现PointNet中的操作主要是乘积和加法，可以交换次序，比如说图中M处就可以移动到最后进行，然后中间就不需要复制$C_{out}$次了，就可以节省大量计算量，并且还有理论分析

![21](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_ShenlanOpen_ConvNet-On-PointCloud_LiFuxin_21.png)

改进后的网络架构，极大地节省了参数量（可节省约六十倍），大大加快了计算的速度，并且可以构建更大的网络（可达三十层），下图中就相当于一个卷积层，可以进行叠加组合构成一个网络

![22](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_ShenlanOpen_ConvNet-On-PointCloud_LiFuxin_22.png)

### PointNet++：更强大的PointNet

#### 动机

在解决置换性方面，PointNet使用两个操作，首先，针对一个点的特征向量，进行一个一维卷积，这相当于乘以一个同样维度的矩阵（下图第一栏所示），得到一个新的特征向量，这是针对单个点的操作，如果想针对多个点，就使用最大池化操作（下图第二栏所示），当然，只使用这两个操作是不够的，或者说PointNet太简单了，没有使用卷积的思想，缺少对局部特征的处理手段

![8](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_ShenlanOpen_ConvNet-On-PointCloud_LiFuxin_6.png)

虽然PointNet很强大，但是还是有缺点的，我们先从三维卷积开始了解，进行对比

三维卷积跟二维卷积一样，不断提取和学习不同层次的特征

但是PointNet不一样，一开始对每个点做MLP低维到高维的映射，把所有点映射到高维的特征通过Max pooling结合到一起，本质上来说，要么对一个点做操作，要么对所有点做操作，实际上没有局部的概念(local context) ，比较难对精细的特征做学习，在分割上有局限性。同时在平移不变性上也有缺陷，比如说你对点云做了一个平移，那么所有的坐标就变了，所有的特征就变了，全局特征就变了，分类也不一样了

对于单个的物体还好，可以将其平移到坐标系的中心，在一个场景中有多个物体不好办，对哪个物体做归一化呢，这也是个问题

![49](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_65.jpg)

所以PointNet++就被提出来了

#### 方法

这种网络的方法就是在局部区域重复性、迭代的使用pointnet ，在小区域使用PointNet生成新的点，新的点定义新的小区域，进行多级的特征学习。
因为是在区域中，可以用局部坐标系，可以实现平移的不变性，同时在小区域中对点的顺序是无关的，保证置换不变性。

我们可以通过一个例子来了解多级特征学习是如何实现的

![55](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_71.jpg)

二维空间中的点有二维坐标，我们先找到一个局部区域（红点区域），我们想学习这个小区域的特征，因为我们不想受整体平移的影响，就将这个小区域移动到局部坐标系下，这样整体点云就不会影响这个小区域的局部坐标了，然后对这个小区域进行PointNet处理来提取特征

然后我们就可以得到一个新的点，除了XY坐标，表示这个区域在整个点云中的位置，还有个向量特征F（高维度的特征空间），代表小区域的几何特征

![56](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_72.jpg)

这个方法可以称为点集的简化，或者可以理解为一个层，包括了小区域的选取、采样、PointNet处理等操作

我们不断重复这种操作，就可以不断进行特征的提取，点的数量越来越少，但是所代表的区域越来越大，与卷积神经网络的情况很接近

![62](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_78.jpg)

最后，我们将原来的点通过某些方法（比如说插值或者转置卷积），传回到原来的位置上，实现语义分割

实际上，可以将这些操作封装为层，或者说，实际上的三层（Sampling、Grouping、PointNet）就相当于卷积神经网络中的一层，如下图所示

![11](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/11.png)

在实际操作的时候，Grouping层可以使用八叉树这些算法来实现快速的搜索，效率更高



![13](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/13-1686049525640-4.png)

#### 问题：采样率不均匀

但是，这里有一个问题，就是这么选择每个小区域的大小，在二维网络中我们知道，大家都会选择非常小的感受野大小，比如说3x3，但是这在点云中不可能，场景的问题是采样率的不均匀（比如说远处的点更为稀疏，近处的点更为密集），最极端的情况下只有个别点，这种情况下，如果感受野太小会受到采样率不均匀问题的影响

![64](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_80.jpg)

如果我们量化地研究这个问题，我们在不同点云数量的情况下分析

![65](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_81.jpg)

在刚开始1024个点的时候PointNet++比较强大，得到更高的精确度，随着点云密度的下降，其性能受到了极大的影响（由于核太小，采样率不均匀的影响很大），在小于500个点以后性能低于PointNet

点云的卷积也是卷积，也是针对连续卷积的离散化处理，不过点云中的卷积要考虑采样率不均匀的问题并且进行应对，考虑卷积在CNN中是一种均匀的采样，但是在点云中明显不合适，所以要做离散化的话就要除以一个距离

![17](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/17-1688221354674-5.png)

#### Multi-Scale Grouping（MSG）

我们面对采样率不均匀问题，可以使用这种方法去解决：每个点周围的多个不同尺度的邻域内分别进行特征提取（综合不同大小的区域的特征，在密集的地方相信这个特征，在稀疏的地方不相信这个区域并且查看更大区域的特征），然后将这些特征进行组合，形成一个多尺度的特征表示，这类似于GoogleLeNet中的Inception结构，不过这里增加了一个Dropout，来随机丢弃输入，来迫使网络去学习如何应对缺失数据和不同尺度的数据

![66](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_82.jpg)

当然还有一个MRG（Multi-res Grouping）方法，这种方法的好处就是可以节省一些计算

#### 效果

可以看到，增加了这种结构之后，面对数据丢失，网络变的更加鲁棒，哪怕丢失75%的数据，精度也不会下降多少

![67](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_83.jpg)

同时，在很多数据集上，PointNet++的精度也有所提高

![68](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_84.jpg)

![69](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_85.jpg)

同时，PointNet++还可以拓展到非欧式空间上（或者说任意的测度空间），只需要有一个定义好的距离函数，我们将其用在一个可变形物体数据集上，比如说图中的ab两个物体，看起来很像但是不是同一个类别，ac两个物体看起来差异很大但是属于同一个类别

![70](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_86.jpg)

这种就无法通过坐标变换来分类了，而是通过表面的学习去完成

### PointWeb：增强点云处理中的局部邻域特征

这是PointNet++之后的一个工作，其核心思想就是尽可能利用局部邻域中的上下文特征，以便于更好地表示该区域，其核心单元就是**自适应特征调整模块（Adaptive Feature Adjustment，AFA）**，此模块作用为寻找点之间的相互作用，以便于更好地对一个区域进行编码

## 三维物体检测

### 方法分类

#### Point-based

这种方法是完全集运点云的，将所有的点云信息送入网络进行学习，优点是不会遗漏任何空间信息，精度上可能有优势，但是点一般是数以万计的，会造成计算量十分庞大，所以后面并不是主流方法

![AutoDriveHeart_PointCloudDetection3D_L1_10](C:/Users/pengf/Desktop/%E6%96%B0%E5%BB%BA%E6%96%87%E4%BB%B6%E5%A4%B9/L1/AutoDriveHeart_PointCloudDetection3D_L1_10.png)

#### Voxel-based

这种方法主要是点云空间的体素化，然后体素中包含特征信息，这种表示方法更规范化，卷积起来效率比点云更高，但是会造成一些信息的损失和精度损失

#### Graph-based

将点云转化为有序的图（点和边的集合），图理论更有数学理论支持，可以更加理论化，但是这种方法的计算量更大，更加耗时，甚至慢于点云卷积方法

#### Pillar-based

工业界的主流方法，非常高效，并且精度也不差，核心思想是将无序点云转化为伪图像然后使用二维卷积网络处理，具体方法是在俯视平面上划分网格，然后统计网格中点云数量并且形成一个个柱子（Pillar），并且使所有的柱子中的点云数量为N（多的剔除，少的补零）

这种方法较好的平衡了效率和精度，在保持了非常高效率的同时保证了精度

#### Point+Voxel

结合点和体素方法的优点，核心方法是只使用关键点的信息进行refine，而不是使用所有的点云信息，这样子可以保留一些点云的原始信息，同时主要使用Voxel的信息，使得网络非常高效

我们输入RGB-D数据，然后我们希望使用一组边界框去标出这些物体，下图中的示例是在图像中的表达，实际上我们还会有在空间中的表达，相对二维检测，空间维度上输出的信息会多很多

![74](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_90.jpg)

![75](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_91.jpg)

### 数据集

### 思路

我们这里的思路可能与二维中的Faster R-CNN一致，对一些候选区域进行分类

实际上，之前的工作也是这样的，第一种思路就是先在三维空间中筛选出候选区域，然后将其投影到二维图像上进行分类，然后将二维特征结合到三维特征上，但是这样有问题的，三维点云空间很大，计算量也会很大，同时因为点云的特性，也无法发现一些小物体

![76](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_92.jpg)

第二种工作就是基于图片的，这种方法依赖于对物体大小的先验知识，也无法精确估计物体深度和大小

![93](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_93.jpg)

然后，祁芮中台的思路是结合2D和3D的优点，我们先在二维图片上使用检测算法得到一个区域，如下图中我们检测出的车辆（红色框中），然后根据这个区域去生成一个**三维视锥（3D Frustum）**，即下图中红色四棱锥，然后我们就可以在这个视锥中搜索，或者在点云中搜索

![78](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_94.jpg)

这种方法大大减小了搜索的计算量和复杂度，同时在视锥内，我们可以直接在点云上做操作，我们可以利用点云几何的精确性或者PointNet等去直接分析和处理点云数据，得到非常精确的三维边界框

### 挑战

首先，相比于二维目标检测，三维目标检测的表示多了一个维度的位置和尺寸，同时还增加了三个姿态角

此外，在有遮挡的情况下想去识别物体还是有挑战的

第一个就是有遮挡的情况，如下图所示，这个房子前面的人的一部分被遮挡了，我们根据二维区域去在视锥中进行搜索的时候，我们以上帝视角可以发现，这里的点云分布在非常大的尺度上，而这个人所对应的区域是非常小的一块点云，会有非常多的前景遮挡和干扰点，很难使用三维CNN方法等去处理

![80](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_96.jpg)

这里我们使用PointNet去处理

![84](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_100.jpg)

我们先在二维图像中找到这个区域，生成视锥，然后在视锥内搜索，找到后使用网络进行分割，检测出关键点，分割除去前景点和背景点，然后运行另一个网络进行姿态估计和标出边界框

这种方法的效果在KITTI数据集上很长时间排名第一

![85](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_101.jpg)

在一些小的物体上，比如说行人和自行车，这种方法的效果要优于其他算法

![86](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_102.jpg)

这是因为二维图像的分辨率是很高的，可以很容易的筛选出来这些目标

### 成功关键因素

这种方法成功的原因有两个

第一个是这种方法选择了三维的表示，相比于二维，有更好的分割效果，我们可以在二维上分割出一个掩码，然后将其投射到三维中

![90](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_106.jpg)

第二个就是可以进行输入归一化操作，简化学习问题

比如说我们有一个车处于视锥范围内，但是其不在中心位置，我们也只能看到一部分的侧面，同时因为视锥范围很大，X坐标的变化也很大，如下图所示

![92](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_108.jpg)

我们可以对坐标轴进行旋转平移，视锥中心就可以指向Z轴方向（如下图b所示），这样特征点在X的分布就会简化，学习起来也会容易；同时，物体深度也有很大区别，所以我们进行一个平移，我们基于三维物体分割可以找到分割后物体的中心，这样物体的点会集中在原点附近（下图c所示），便于学习，然后我们再使用一个网络，进一步预测物体真实的中心（下图d所示）

![96](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_112.jpg)

经过这一系列的操作，可以看到精度快速提高，而且这些操作只需要通过矩阵乘法实现

![97](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_113.jpg)

### 效果展示

室外，可以精准分割，哪怕两个物体很近

![99](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_115.jpg)

室内

![102](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_118.jpg)

### 其他应用

AI辅助外形设计（工业4.0）

![106](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_145.jpg)

机器人的应用

![107](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_146.jpg)

蛋白质结构预测功能

![108](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_147.jpg)

甚至用在更一般化的地方，比如说分析图片关系

![109](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointCloudDL_TechBeatTalk_CharlesQi_148.jpg)

### 点云目标检测框架

目前的点云目标检测框架大概有这些

## PointRCNN：点云目标检测算法

### 介绍

PointRCNN是CVPR2019录用的一篇三维目标检测论文，曾在KITTI的3D检测测试中排名第一，文章为[PointRCNN：3D Object Proposal Generation and Detection from Point Cloud][https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.html]，这是一个两阶段的框架，先定位后精细化，与图像的两阶段目标检测算法思想类似

![PointRCNN_1](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointRCNN_1.png)

其两阶段的作用分别是：

- **第一阶段**：

  **目的**：分割前景点，并且生成3D bounding box proposals

  **方法**： 通过使用PointNet++作为骨干网络，提取point-wise feature vector，利用3D bounding box来生成ground-truth分割掩码，第一阶段分割前景点，同时在这些被分割出来的点中生成少些bounding box proposals。

  **核心思想**：学习point-wise 的特征来分割原始点云，在分割好的前景上生成3D 候选框。这样避免了在3D空间中使用大量预定义的3D框，极大地限制了3D建议框的搜索空间，从而减少计算

- **第二阶段**：

  **目的**：对生成的3D box进行微调

  **方法**：在3D proposals生成之后，用一个点云ROI pooling操作来处理第一阶段学习到的点。现存的3D方法都是直接估计全局box坐标，而作者的方法是将被池化的3D点转换到正则坐标，并且和来自阶段1的被池化的点的特征还有分割掩码结合，从而学习相对坐标微调。

  **优势**：充分利用了第一阶段分割和子网络提供的结果信息。

### 动机

基于点云的3D检测的困难主要在于点云的不规则性，常见3D检测的SOTA的方法：

1. 将点云投射到俯视图(AVOD)或者投射到前视图(MV3D)，利用2D检测的框架进行目标检测
2. 变成规则的3D voxels (VoxelNet)。

但这些都不是最佳的，而且投射的方法会遇到在量化过程中信息缺失的问题，并且三维卷积的计算量大。

后来有人提出PointNet，不将点云转换成voxels或其他的规则的数据结构来进行特征学习，而是直接从点云数据来学习3D表示，他们还在3D目标检测中应用了PointNet，从2D RGB检测的结果获得被截取点云，基于这个来估计3D bounding box（F-PointNet，视锥网络），但是这种方法过于依赖2D检测的表现，如果图像漏检或者重叠，都会带来较坏的结果。

### 网络框架

整体的网络架构如下

![PointRCNN_1](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointRCNN_1.png)

这是一个两阶段的目标检测框架，第一阶段生成一些候选区域，第二阶段进行候选框的回归

- stage-1：基于点云分割的3D RPN

  ![3DCVer_AutoDrivePointCloudDetection_L3_11](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/3DCVer_AutoDrivePointCloudDetection_L3_11.png)

  使用PointNet++作为骨干网络（或者说Encoder和Decoder），提取point-wise feature vector，然后区分前景点并且生成候选框

  （1）提取点云特征：使用PointNet++ 作为骨干网络提取特征point-wise feature vector

  （2）前景点分割：每个点的分类问题，由两个卷积层组成。输入是(bs,n,128)的特征（每个batch中，有n个点，每个点有128维特征），输出是(bs, n, 1)的mask（只是区分这个点是不是前景点）。由于背景点的数量远大于前景点数量，正负样本不均衡，所以分类误差损失函数选择使用focal loss

  （3）基于bin区间的3D框生成：也是由两个卷积层组成。输入是(bs, n, 128)的特征，输出是(bs, n, 76)。3d目标检测里的bounding box，需要7个量来表示：[x,y,z,l,w,h,yaw]。这里用基于bin的预测方法使76个维度特征来代表这7个量，然后进行非极大值抑制（NMS），得到一批预选3d区域

  其中基于距离的NMS是这样处理的：

  - 在相机 0～40m距离内的bounding box，先取得分类得分最高的6300个，然后计算bird view IOU，留下IOU大于0.85的，到这里bounding box 又少了一点。然后再取得分最高的210个。
  - 在距离相机40～80m的范围内用同样的方法取90个。这样第一阶段结束的时候只剩下300个bounding box了。
  - 再送入stage-2进行置信度打分和bounding box优化。

- stage-2：基于bin区间的3d框回归

  ![PointRCNN_2](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/PointRCNN_2.png)

  3D box： ( x , y , z , h , w , l , θ ) 

  bin就相当于直尺上的刻度，用绿色大括号表示。这里设物理空间中0.5m是一个bin。通过分类的方式去预测每个点对于bounding box中心点偏移了几个bin（或者说预测中心点落在哪一个bin里面），而不是直接回归具体的位置，其中蓝色的是分割出来的前景点，紫色的是兴趣前景点，以兴趣前景点为例去进行预测

  旋转角也用这种基于bin的方法预测，是把π话分成若干个bin。由于bin是一个整数，还是无法精确定位，所以还需要预测中心点坐标在一个bin中的偏移量。

  此外，76维输出向量是如何对应7个参数的呢

  ![3DCVer_AutoDrivePointCloudDetection_L3_16](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/3DCVer_AutoDrivePointCloudDetection_L3_16.png)

  76维的前12维，x\_bin，是判断X轴的bin区间，判断中心点落在哪一个bin上，12维表示坐标轴正负各有六个bin，然后还有在每个bin上的回归，x\_res，也占了十二个维度，然后在XZ方向上都是如此操作，就占了48个维度

  但是在Y方向（或者说垂直方向）上没有这种划分，因为自动驾驶的目标物体都是在地面上的，所以不会出现太大的出入，只有一个一维回归

  然后YAW角也是基于bin的划分，12个bin区间和12个偏移量

  最后对长宽高也是有回归，占3维

## 3D SSD

### 概述

3D SSD方法，原论文3DSSD: Point-based 3D Single Stage Object Detector，是一种借鉴二维SSD方法的

## CenterPoint：点云目标检测

### 概述

CenterPoint是一个点云目标检测的框架，借鉴了YOLO、SSD等单阶段二维目标检测框架的思想，直接检测隐式的物体中心，不需要候选框