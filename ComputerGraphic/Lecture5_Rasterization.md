# 二维图形

与存储像素点的图像不同，二维图形是一个有数学坐标表达的，由点、线、区域等数学要素所构成的几何对象，只不过由于计算机限制，只能使用有限的点来表示这个连续的空间

![ComputerGraphics_USTC_LiuLigang_L3_20](https://github.com/Michael-Jetson/Images/blob/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_20.png?raw=true)

![ComputerGraphics_USTC_LiuLigang_L3_95](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_95.png)

理论上，图形是分辨率无限大的，只不过计算机无法显示那么多像素，或者说不能表达一个连续空间

![ComputerGraphics_USTC_LiuLigang_L3_91](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_91.png)

![ComputerGraphics_USTC_LiuLigang_L3_92](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_92.png)

这种的应用有需要超大分辨率的图像，比如说高空影像、卫星地图等等

![ComputerGraphics_USTC_LiuLigang_L3_21](https://github.com/Michael-Jetson/Images/blob/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_21.png?raw=true)

当然，我们可以把多个图形混合叠加起来，这样子就可以把一些复杂的图形拆解开来，实现快速绘制

![ComputerGraphics_USTC_LiuLigang_L3_94](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_94.png)

## 应用：互联网地图

实际上图形的一个应用就是地图，在不同的视点和距离上，看到的要素是不一样的

![ComputerGraphics_USTC_LiuLigang_L3_98](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_98.png)

但是这种是需要渲染矢量数据，然后把视图拼接出来，渲染的地点可以是云端或者用户电脑

![ComputerGraphics_USTC_LiuLigang_L3_100](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_100.png)

具体的策略有如下所示

![ComputerGraphics_USTC_LiuLigang_L3_101](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_101.png)

但是因为需要的计算量太大，不可能预先完全渲染，所以只能进行预计算，预先计算一部分，用的时候再用，做一个时间和空间的协调

![ComputerGraphics_USTC_LiuLigang_L3_102](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_102.png)

# 光栅化

## 回顾

透视投影转正交投影保证近和远两个平面不变（或者说变成和近平面一样大），然后我们使用XYZ三轴的覆盖表示空间中立方体的范围，X覆盖左右，Y覆盖上下，Z覆盖远近

在转化的时候，我们怎么定义这个视锥呢？我们从相机出发，假设我们看到的就是近平面，那么可以为其定义宽度和高度还有**宽高比（Aspect Ratio）**，当然还有另一个概念就是Field-of-View（fovY），垂直的可视角度（下图红色虚线所示）就是fovY

![Games101_L5_4](https://github.com/Michael-Jetson/Images/blob/main/UpGit_Auto_UpLoad/Games101_L5_4.png?raw=true)

我们可以将这些概念进行迁移

![Games101_L5_5](https://github.com/Michael-Jetson/Images/blob/main/UpGit_Auto_UpLoad/Games101_L5_5.png?raw=true)

我们可以计算可视角度等

## 屏幕

### 基础概念

我们前面在MVP（Model View Projection）中有一种投影变换挤压为正交投影的方法，并且提到了标准立方体，但是立方体如何显示在屏幕上呢？在实现这个之前，就需要去了解屏幕的概念

图形学中，我们认为屏幕就是一个二维数组，数组中的元素就是一个**像素（Pixels）**，分辨率就是数组的形状（或者说屏幕大小），比如说1920x1080，并且，屏幕是一种典型的光栅成像设备，光栅就是像素的阵列，光栅化就是把事物画在屏幕上，使用离散的像素来表现连续空间中的对象

![Games101_L5_7](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_7.png)

我们这里认为像素就是一个带有颜色的小方块，尽管像素实际上很复杂，颜色是红绿蓝的离散值的混合（RGB像素）

![619f5bc6b147bfcd5e9f868bb5974ce6-6](https://raw.githubusercontent.com/Michael-Jetson/Images/main/619f5bc6b147bfcd5e9f868bb5974ce6-6.png)

### 屏幕空间

屏幕空间就是一个屏幕上建立起来的坐标系，如下图所示，我们认为左下角是屏幕空间的原点，然后定义像素的坐标（或者说下标、位置）是整数的形式，下图中蓝色方块的坐标就是（2，1），实际上我们使用的是像素方块左下角的坐标定义像素坐标，而不是像素方块的中心坐标（2.5，1.5）

![Games101_L5_8](https://github.com/Michael-Jetson/Images/blob/main/UpGit_Auto_UpLoad/Games101_L5_8.png?raw=true)

然后我们在三维空间中有一个立方体，这个立方体如何显示在二维的屏幕上呢？我们先不管Z，先看XY方向，将其从标准正方形变到整个屏幕上

![Games101_L5_9](https://github.com/Michael-Jetson/Images/blob/main/UpGit_Auto_UpLoad/Games101_L5_9.png?raw=true)

![Games101_L5_10](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_10.png)

## 其他成像设备

![Games101_L5_13](https://raw.githubusercontent.com/Michael-Jetson/Images/main/\UpGit_Auto_UpLoad/Games101_L5_13.png)

CNC制造技术就是一种在金属上绘制的技术，或者可以在其他的物理材质上进行显示，比如说打印机

### 阴极射线管

这是老式电视机的成像技术，电子管发射阴极粒子流，然后经过电极偏转，打到显示屏上就可以成像（通过逐行扫描成像），如下图所示

![Games101_L5_19](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_19.png)

有一种压缩成像的方法，就是隔行扫描，但是这会造成画像割裂的情况

彩色成像就是使用三个电子管分别制造不同颜色成像的

![ComputerGraphics_USTC_LiuLigang_L3_9](https://github.com/Michael-Jetson/Images/blob/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_9.png?raw=true)

但是在靠近边缘的地方，电子造成的光斑较大，成像效果较差，在屏幕中间的成像效果较好，这也是CRT方法的一个问题

### 帧缓冲器： 光栅显示的内存

现代的显示器是如何进行显示的呢？在显示器中都有显卡，显卡中有显存，显存中的一个区域，其中的内容映射到屏幕上就完成了显示

现代的显示器通常是LCD和OLED等，比如说计算器的显示屏就是一个分辨率很低的LCD，手机的显示屏就是一个高分辨率的设备，甚至可以超过人视网膜的分辨率（超过的就称为视网膜屏幕）

![Games101_L5_21](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_21.png)

#### LCD：液晶显示器（Liquid Crystal Display）

液晶显示器的工作原理主要基于液晶材料的特性。液晶是一种特殊的物质，它的状态介于固态和液态之间。在电场的作用下，液晶分子可以改变排列方式，从而改变通过它的光线的偏振状态，进而控制单个像素的显示。

![Games101_L5_22](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_22.png)

液晶显示器（Liquid Crystal Display, LCD）使用液晶材料和光的偏振原理来显示图像。这个过程涉及到一些相当复杂的物理和化学知识，但在最基本的层面上，我们可以简化为以下步骤：

1. **背光系统发光**：LCD本身不发光，因此它需要一个背光系统提供光源。这个光源通常来自于LED或者冷阴极荧光灯（CCFL）。
2. **通过第一个偏振片**：背光发出的光首先通过一个偏振片。偏振片只让特定方向的光线通过，例如，只让垂直方向的光线通过。经过偏振片之后，光线变成了线性偏振光。
3. **通过液晶层**：偏振光然后通过液晶层。液晶层中的每个像素都可以通过改变电压来控制液晶分子的排列，从而改变光线的偏振状态。
4. **通过彩色滤镜**：液晶层之后是彩色滤镜，每个像素都有红、绿、蓝三种颜色的滤镜。这些滤镜可以分别对红、绿、蓝光进行过滤，以产生不同的颜色。
5. **通过第二个偏振片**：最后，光线通过第二个偏振片。这个偏振片的偏振方向与第一个偏振片垂直，例如，只让水平方向的光线通过。因此，只有当液晶分子将光线的偏振方向转为水平时，光线才能通过第二个偏振片。否则，光线会被阻止，像素看起来就是黑色的。
6. **形成图像**：通过控制每个像素通过的光线的亮度和颜色，LCD就能显示出各种各样的图像。

原理本质上就是一种材质也就是液晶，可以实现：通过施加电压的大小，可以改变某一颜色透光量的多少，从而模拟RGB不同颜色的组合

液晶显示器的这种工作原理使得它能够在非常薄的层面上显示高清晰度的图像，而且能耗也比许多其他类型的显示器要低。然而，这种工作原理也决定了液晶显示器的一些缺点，例如视角有限、响应时间较长、黑色不够深等。

#### LED（Light emitting diode）：发光二极管

就是发光和不发光两种情况

![Games101_L5_23](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_23.png)

#### 电子墨水屏

通过不同的电压来控制像素的黑白情况，但是缺点是刷新率低

电子墨水屏使用微胶囊或微囊技术。每个微胶囊都像一个微型沙漏，内部填充有两种不同的颗粒。这两种颗粒通常是黑白两色，一种是带正电的白色颗粒，另一种是带负电的黑色颗粒。

当在微胶囊上施加电压时，带电颗粒会根据电场的方向移动。例如，如果上部电极带正电，那么带负电的黑色颗粒就会被吸引到上部，而带正电的白色颗粒则会被排斥到下部，于是从屏幕上看到的就是黑色。反之，如果上部电极带负电，那么就会看到白色。

在电子墨水屏中，每个像素就是一个微胶囊，通过控制每个像素的电场，就可以控制屏幕上的图像。此外，电子墨水屏有一个重要的特性，那就是它是双稳态的。也就是说，当改变了颗粒的位置后，即使去掉电压，颗粒也会保持在新的位置。这就是为什么电子墨水屏在静态显示的情况下几乎不消耗电能的原因。

![Games101_L5_24](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_24.png)

## 光栅化：在光栅显示屏上绘图

我们前面提到了一些成像设备，那么如何在这些成像设备上进行显示呢，同时也提到了三维空间中的点和形状如何投影到二维平面上，就是把多边形拆成像素（也就是光栅化的过程），在这里我们主要研究三角形的光栅化，原因如下

- 三角形是最基础的多边形，可以表示其他的多边形
- 三角形肯定是一个平面并且内外定义清晰
- 只需要定义三角形顶点的不同属性，那么内部就是渐变的，可以得到内部任意点的属性

我们在空间中有一个三角形，那么我们在屏幕空间中就只看XY坐标，然后我们就需要找到一个方法将其变为屏幕上像素显示的图像，这里就有一个光栅化中核心的概念，就是判断光栅中的像素和三角形位置关系，或者说考虑像素的中心点与三角形的位置关系

![Games101_L5_30](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_30.png)

### 线段的光栅化

当然，实际上还有线段的光栅化操作，我们首先考虑端点坐标为整数的线段，然后就可以使用DDA算法找到最接近的整数

![ComputerGraphics_USTC_LiuLigang_L3_36](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_36.png)

当然，考虑到这里有非常多的浮点数计算，考虑使用CPU最擅长的左移右移操作来加速计算

这个算法的思想是基于前一个像素来判断下一个像素，因为线段是连续的，我只需要判断下一个像素相对前一个像素的位置就可以了

![ComputerGraphics_USTC_LiuLigang_L3_38](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_38.png)

但是现在的线段光栅化已经成熟，并且可以使用硬件实现，所以没有进一步研究的价值

### 多边形的光栅化

这是原始的多边形光栅化方法

实际上也是由硬件实现并且很成熟的

### 光滑曲线的光栅化

当然，还有一些由数学公式定义的光滑曲线，我们如果进行光栅化呢？最直接的方法就是光滑曲线转化为局部直线（泰勒展开的第一项就是直线），因为计算机是无法理解连续的，这样才可以进行离散表达，具体的操作入下图所示，使用分段的线性逼近，并且分段越多，误差越小

![ComputerGraphics_USTC_LiuLigang_L3_61](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_61.png)

判断标准是曲线和每段直线的最大距离或者围成面积的和等等

### 隐函数的曲线

对于$f(x,y)=0$​这种的隐函数的曲线，我们怎么去绘制呢？一个自变量值可能对应多个因变量的值

1986年提出的Marcking Cube算法是一个很好的解决这种问题的算法，通过寻找像素点的角点$(x_0,y_0)$对应的值$f(x_0,y_0)$的正负来判断

### 三角形采样方法

这里有一个简单的方法，就是采样，假设有一个连续函数，我们在不同的地方去采集它的值，或者说采样就是将连续函数离散化的过程

![Games101_L5_32](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_32.png)

采样是很重要的概念，我们这里的采样指的是利用像素中心对屏幕空间进行采样，后面还有对其他方面的采样

在三角形中，我们判断像素中心是否在三角形内部，这个也是一个函数，我们定义这个函数为``inside(t,x,y)``

![Games101_L5_34](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_34.png)

这个函数的返回值是1或者0，然后就可以判断一个像素是否一个被填充

![Games101_L5_36](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_36.png)

我们回想如何判断点在三角形内外，答案就是点积或者叉乘

![Games101_L5_40](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_40.png)

不过总有一些特殊情况，比如说点恰好在三角形的边上，甚至在多个三角形的边或者顶点上，那么这样点算在哪个三角形上呢？不过我们在这里选择不做处理

![Games101_L5_41](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_41.png)

不过，我们也不可能对屏幕上每个像素都检查一遍，这样太浪费计算资源了，尤其是一个三角形可能只覆盖屏幕上一小块区域的时候，所以我们有一种节省计算资源的方法，就是设置一个边界框（如下图所示），其范围就是顶点的最大最小值确定的，完全包裹住三角形，这样在光栅化的时候只需要在这个边界框中计算就可以，不再边界框中的点更不可能是三角形的填充像素

![Games101_L5_42](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_42.png)

然后可以进行加速，比如说下图这种方式，每一层都从每一行像素的最左开始和最右结束，可以进一步节省计算资源

![Games101_L5_43](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_43.png)

### 实际屏幕的光栅化

当然前面只是理论，具体的屏幕显示如下

当然仔细分析的话，会发现右边绿色的像素更多，这是因为人眼对绿色的敏感度更高

![Games101_L5_45](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_45.png)

当然在其他的成像设备上可能会有所区别，比如说在彩色打印上，可能会出于节省颜料的目的去减少颜色

![Games101_L5_46](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_46.png)

当然，我们在这里仍然认为每个像素的颜色是一致的（或者说是均匀的）

![Games101_L5_47](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_47.png)

如果我们使用采样的方法进行了显示，那么在细节上会出现下面的效果，这与我们想要的效果相差甚远，这里有很多锯齿，这也是光栅化致力于解决的问题（抗锯齿和反走样），出现的原因就是像素本身有一定的大小，并且采样率对于信号来说是不够高的

![Games101_L5_49](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L5_49.png)

接下来就会去讲解如何完成抗锯齿和反走样问题，并且在信号频率角度去理论分析成因

光线叠加会更加强烈明亮，但是色彩叠加会变暗，所以有聪明的画家发明了点彩法，用很小的紧密的点间隔画上黄色和蓝色，实现高亮饱满的绿色

# 抗锯齿

上面我们可以看到，分辨率低的情况下会有很严重的**锯齿（学名叫Aliasing，走样）**，我们需要进行一些缓解

![ComputerGraphics_USTC_LiuLigang_L3_72](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_72.png)

在数学上是这样的

![ComputerGraphics_USTC_LiuLigang_L3_77](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/ComputerGraphics_USTC_LiuLigang_L3_77.png)

## 采样理论

采样是图形学中一个广泛使用的理论，不仅可以在空间上采样，也可以在时间上进行采样，比如说视频，实际上视频不是连续的，也是一系列在连续时间上采样的图片

![Games_101_L6_12](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_12.png)

同样，采样也会有很多的**错误或者我们不希望看到的结果（英文对应于Artifact）**，比如说锯齿、摩尔纹和车轮效应

![Games_101_L6_14](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_14.png)

摩尔纹（Moiré pattern）是一种在重叠的规则模式中经常出现的视觉现象，当两个相似的图案（例如网格或线条）重叠，但相对于彼此稍微偏离或旋转时，就可能产生摩尔纹。摩尔纹表现为明暗交替的条纹或环形图案。

在计算机图形学中，摩尔纹现象经常出现在纹理贴图（texture mapping）的过程中。当需要将高分辨率的纹理映射到低分辨率的图像上时，如果没有采取适当的采样和滤波策略，就可能产生摩尔纹。因为在这个过程中，原始纹理的高频信息可能超过了目标图像能够表示的频率，从而产生了摩尔纹现象。这就像在数字采样中没有遵循奈奎斯特定理（Nyquist Theorem）所产生的混叠现象。

![Games_101_L6_15](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_15.png)

车辆效应就是旋转物体上可以看到不同花纹

![Games_101_L6_16](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_16.png)

产生这种情况的原因，就是信号变化太快了，采样速度跟不上

## 抗锯齿方法：采样前模糊化（或者预先滤波）

我们对三角形做一个模糊操作，将其变为一个模糊的三角形，然后进行采样，这样，有的采样点就是在模糊的地方采样，是浅红色，有的采样点在三角形中心采样，是红色，这样就可以解决一些锯齿化问题

![Games_101_L6_20](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_20.png)

如下图所示，可以看到，使用这种方法处理之后，锯齿化程度降低了

![Games_101_L6_21](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_21.png)

![Games_101_L6_22](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_22.png)

直接对比也可以看到，效果比较明显

![Games_101_L6_23](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_23.png)

## 模糊操作原理

任何信号都可以分解为若干不同频率的基本的正弦函数和余弦函数的叠加

![Games_101_L6_28](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_28.png)

我们可以使用傅里叶变换获得不同频率的基本信号的系数

可以看下图，有五个不同频率的基础信号，我们使用相同的采样率进行采样，我们发现，我们对于过高频率的信号，采样率低的话就无法恢复信号

![Games_101_L6_31](/home/robot/Project/ML_DL_CV_with_pytorch/ComputerGraphic/assets/Games_101_L6_31.png)

我们可以看下图，信号频率过高，采样率低，就无法恢复，只能恢复为一个低频信号，会出现两种信号采样结果一致，无法区分

![Games_101_L6_32](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_32.png)

滤波，实际上就是得到某个特定频率范围的信号，不是这个频率范围的信号会被剔除

傅里叶变换的作用就是把信号从时域变到频域

下图所示，对左图进行傅里叶变换，将空域变为频域，即右图，右图中心就是最低频的区域，周围就是高频的区域，可以看到，左图的信息集中在低频信息上

![Games_101_L6_34](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_34.png)

假如我们使用滤波器，将低频信号滤除，就会有下图的情况，我们发现，高频信号主要是边界，这是因为高频信号表示图像中变化比较剧烈的地方，也就是边界

![Games_101_L6_35](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_35.png)

如果我们剔除高频信号，保留低频信号， 就会出现这种情况，图像变模糊了，边界消失了

![Games_101_L6_36](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_36.png)

还可以只保留中间频率的信号，就可以提取到一些不怎么明显的边界区域

![Games_101_L6_37](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_37.png)

实际上，卷积就是滤波操作，比如说下图，使用一个这样的卷积核进行卷积，就可以完成一个模糊操作

![Games_101_L6_44](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_44.png)

### 采样就是在重复频域内容

如下图，a是原信号，b是其频域，在a上采样就好比是乘以另外一个冲激函数，值域乘积在频域上就是卷积，也就是b和d的卷积，得到f

![Games_101_L6_49](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_49.png)

如果采样不够快，就会导致频域的混叠

![Games_101_L6_50](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_50.png)

## 反走样方法

1. 增大采样率
   - 这是一个终极解决方法，比如说换一个更高分辨率的显示器
   - 需要更高分辨率，并且计算量更大
   - 这样，频谱上间隔大，就不容易混叠
2. 反走样
   - 先模糊，后采样；先剔除高频，然后采样

![Games_101_L6_53](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_53.png)

在实际操作上，我们使用什么滤波器完成模糊操作呢？答案就是使用一个一定大小的低通滤波器进行卷积操作（或者说平均操作）

卷积的效果如下图所示

![Games_101_L6_58](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_58.png)

当然，也可以通过高采样率解决

### 超分辨率反走样方法

对三角形覆盖的每个像素，我们都求一下其覆盖面的平均就可以完成平均卷积操作，但是怎么把这个覆盖面计算出来也是个问题，人们研究出来一种近似方法，这种方法就是**Multi Sample Antialiasing（MSAA）**，也就是更多的采样点进行反走样，当然这只是反走样的一种近似，不能严格意义上解决反走样问题

对于任何一个像素平面，一个像素就是一个采样点，我们认为一个像素被划分成了若干小像素（如下图所示），然后别去判断小的像素中心是否在三角形内，然后将计算结果平均起来，然后就可以得到三角形对大的像素的覆盖区域的一个近似计算

![Games_101_L6_60](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_60.png)

比如说，原始的像素与三角形如下

![Games_101_L6_61](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_61.png)

然后我们将每个原始像素分为四个小像素（如下图所示），或者说每个像素中间多加一些采样点，然后挨个判断小像素是否在三角形内

![Games_101_L6_62](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_62.png)

我们依次判断，如果一个像素的子像素都不在三角形内，那么覆盖率就是0，如果三个点被覆盖，那么覆盖率就是75%

![Games_101_L6_64](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_64.png)

计算结果如下，覆盖率表示了每个原始像素的颜色深浅

![Games_101_L6_65](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_65.png)

然后就可以得到这样的结果，上面代表模糊操作完成了，然后就是采样操作（如下图所示）

![Games_101_L6_66](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_66.png)

MSAA展示的是对信号的模糊操作，而且MSAA也不是通过提高采样率去解决问题的

当然，MSAA也增大了很多计算量来换取这种效果

当然，还有其他的抗锯齿的方法，如下图所示

![Games_101_L6_69](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games_101_L6_69.png)

Fast Approximate Anti-Aliasing (FXAA) 是一种计算机图形学中常用的抗锯齿技术。抗锯齿技术的目的是为了减少数字图像中的锯齿状边缘，使得边缘更加平滑。

FXAA 的原理是通过对图像的亮度梯度进行分析，找到边缘（通常锯齿出现在这里），并进行平滑处理。这是一种后处理技术，也就是说，它不在3D模型的几何级别处理抗锯齿，而是在最终的2D图像级别进行。

FXAA 的优点是速度快，对硬件要求低，可以在各种性能的设备上运行。而且因为它是后处理技术，所以可以独立于具体的渲染技术或图形API，容易集成到各种渲染流水线中。

然而，FXAA 也有一些缺点。由于它在像素级别工作，所以可能会对图像的细节产生影响，导致某些细节被模糊。此外，它也不能很好地处理子像素级别的锯齿。



Temporal Anti-Aliasing (TAA) 是另一种抗锯齿技术，其在渲染过程中使用了时间维度的信息，因此得名“时间抗锯齿”。TAA 不仅可以减少静态图像中的锯齿，还可以处理动态图像中由于物体或摄像机运动产生的锯齿和闪烁问题。

TAA 的基本思想是对当前帧和前一帧（或多帧）的图像进行某种形式的混合。通常来说，这需要存储前一帧的图像，并将其与当前帧进行比较。此外，由于物体和摄像机可能会移动，因此需要某种形式的运动矢量来确定如何从前一帧的像素映射到当前帧。

举例来说，假设你在看一个移动的物体，那么在连续的两帧中，这个物体的位置是有所不同的。通过计算这个物体在这两帧中的位置差，可以得到一个运动矢量。然后，就可以根据这个运动矢量将前一帧的像素“移动”到当前帧，从而实现抗锯齿。

TAA 的优点是能够很好地处理动态图像，减少运动模糊和闪烁问题，同时也能减少静态图像的锯齿。而且，由于其基于时间的特性，TAA 对图像的采样频率要求较低，因此对性能的影响较小。

然而，TAA 也有一些缺点。由于其基于时间的特性，所以对于高速移动的物体，或者在摄像机移动非常快的情况下，可能会导致图像模糊。此外，由于需要存储和处理前一帧的图像，所以 TAA 对内存的需求较大。