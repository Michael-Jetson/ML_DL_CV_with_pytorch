# 遮挡（Visibility ）

光栅化的作用就是将三角形画在屏幕上，但是面对一系列的三维空间中的三角形，如何将其画在屏幕上并且保持一个遮挡关系呢，解决方法就是深度缓存

## 画家算法

一个三维空间的场景下有很多物体，我们想将这些放在屏幕上，那么必定有一个顺序，或者说存在遮挡的问题，那么如何在屏幕上显示这个效果呢，一个方法就是先显示远处物体，然后将近处的物体覆盖上去，逐渐从远到近完成这个结果，以前的油画就是这样完成的，这种方法也被叫做**画家算法（Painter's Algorithm）**，思想就是新物体可以覆盖旧物体

其名称来源于现实世界中画家绘画的过程，画家在创建一幅画作时，会从后向前绘制，较远的物体会先被画出，然后再画出较近的物体，这样较近的物体就覆盖了较远的物体。

![Games101_L7_4](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_4.png)

画家算法在一些情况下是适用的，但是对于一些复杂的情况，是无法完成正常显示的，比如说下图之中的互相遮挡的情况，画家算法就无法完成排序的显示，所以无法实际应用

![Games101_L7_5](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_5.png)

## 深度缓存（Z-Buffer）

为了解决上面的问题，人们找到了其他的方法，也就是深度缓存（也叫深度缓冲），也就是Z-Buffer，这个算法目前应用广泛

面对空间中的三角形，想对其排序是很困难的，但是对像素进行排序是很容易的，我们可以逐个像素去判断能否看到不同的三角形或者三角形的部分，就可以在这个像素内永远地记录这个像素所表示的几何的最浅的深度

![Games101_L7_6](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_6.png)

我们会渲染最后的成品图像，在生成这个图像的同时，我们也可以去生成另一个图像，这个图像会储存每个像素所看到的最浅的物体的深度信息，这个称为深度缓存

不过，前面我们规定，相机总是从原点往-Z方向看的，所以这里更远的物体Z轴坐标更小，这里为了简化计算，我们认为深度总是正的

我们可以看一下深度缓存的例子（下图所示），左边是需要渲染出的最终图像，右边是深度图，二者是一起生成的

![Games101_L7_7](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_7.png)

每个物体都由许多三角形组成，每个三角形又有许多像素组成，我们盯着一个像素看，比如说地板上的一个小三角形（这个三角形在最后是要被遮挡的），它有可能覆盖要显示的像素，我们就会把地板在这个点对应的深度记录，然后再把物体放上去，并且我们发现在这个像素的位置上物体会覆盖这个地板（或者说深度更小），那么意味着物体会遮挡住地板，就需要更新这个像素的深度



![Games101_L7_8](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_8.png)

深度缓存的算法如上图所示，核心思想就是每个像素内存储最浅深度，那么可以通过两个循环完成操作

首先我们遍历所有的三角形，因为三角形可以光栅化为像素

每个三角形，如果想要画在像素内，那么其深度就应该小于记录的深度

- 初始化所有的深度缓存，认为它们的深度是无限远的
- 对于每个三角形，执行操作
  - 对于三角形中的采样点(x,y,z)或者说像素
    - 如果其深度小于记录的深度
      - 重新记录

如下图所示，我们认为最开始的深度都是无限大的（R字符代表无限远），然后先计算红色三角形的像素，然后更新，更新完红色三角形后，再计算新的三角形，维护逐像素的深度，得到最终的深度图

![Games101_L7_9](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_9.png)

在这里，也可以对两种算法的复杂度进行分析，深度缓存算法实际上只不过是在遍历求最小值，没有一个排序的操作，或者说和顺序无关

![Games101_L7_10](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_10.png)

同时我们假设不会出现两个三角形在同一个像素的深度一样的情况，这个假设是有一定道理的，因为我们使用浮点型计算和表示的，我们很难去比较两个浮点型是否相等，或者说可以认为两个浮点型数值永远不会相等

总的来说，深度缓存是目前最重要的遮挡算法

# 着色（Shading）

## 问题

目前，我们已经学了投影、光栅化和遮挡这些概念了

![Games101_L7_13](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_13.png)

这样可以得到一些显示结果（下图所示），但是也会有一些视觉上的问题，比如说下图中的立方体矩阵，就会在显示上有视觉错觉，让人类大脑无法正确处理

![Games101_L7_14](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_14.png)

我们期望看到的是下图这种情况，每个面的颜色略有不同，显示的更真实，这也是着色问题

![Games101_L7_15](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_15.png)

## 概念

"着色"是一个重要的概念，会引入明暗的不同和颜色的不同，它涉及如何在图形渲染过程中为像素（或更准确地说，为图形的表面）分配颜色。着色过程的目标是创建出逼真的图像，这通常需要模拟真实世界中的光照条件和物体表面的属性。或者说，着色就是一个对物体应用不同材质的过程

![Games101_L7_18](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_18.png)

我们可以看一下一个简单的例子，光线照射到不同的茶杯上，我们可以在某些地方看到镜面高光（Specular highlights）和环境光照（Ambient lighting）

还有的地方，没有接收到光，但是还是可以看到颜色的，因为物理上，物体只有反射或者发出光线才可以被看到，这里可以看到没有接受的光线的地方，是因为除了直接光照外，还有间接光照（比如说桌面反射的光线照射到杯子上）

![Games101_L7_20](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_20.png)

在研究光照之前，需要定义一些基础概念

我们考虑光照，是考虑每个点上的光照，或者说考虑一个**Shading Point（称为着色点）**的光照（如下图所示），每个着色点在物体表面上，尽管不同的物体会有不同的形状，但是我们认为在一个局部的很小的范围内，这个着色点的表面就是平面，然后可以在这里定义法线$\vec{n}$，观测方向$\vec{v}$和光照方向$\vec{l}$，这些方向向量都是单位向量，然后还有一些表面参数，形容物体表面的一些属性（颜色和亮度等）

![Games101_L7_21](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_21.png)

着色不考虑这个点是否在阴影内，我们只看着色点自己，不考虑其他物体的存在，或者说不考虑影子，所以说着色是局部概念，不显示阴影，只考虑明暗

![Games101_L7_22](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_22.png)

## 光照计算——漫反射

我们从漫反射开始考虑，一束光打到一个着色点上，会被往各个方向反射，如下图所示，当然想计算的话没有那么容易

![Games101_L7_23](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_23.png)

当光线入射到物体表面（或者说着色点）的夹角会影响着色点的明暗，或者说光线照射角度不同，物体明暗就不同，接下来就介绍一下其中的原理

如下图所示，当光线垂直照射的时候，有六根光线照到着色点（左图所示），但是当有一点夹角（中间图所示）的时候，我们发现照射到着色点的光线少了，这个时候物体表面就应该暗一些

![Games101_L7_24](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_24.png)

可以使用数学公式来表示，光线方向和着色点的法线向量的夹角表示光线强度一定的条件下，物体的明暗情况

我们注意一下，光是能量，可以说物体表面的明暗取决于收到光线的能量，一个形象的例子就是四季，北半球六月的时候，太阳接近直射，接收到的能量就多，温度就高，十二月的时候，太阳光线与地面法线的夹角就比较大，接受到的能量就少，温度就低

有接受，就有发射，那么光线是如何产生的？光是能量，来自于光源，下图是一个点光源，光线朝四面八方辐射能量，我们这里有一个观测方法，在任意一个时刻，点光源辐射的能量集中在一个球壳上（下图中的圆圈），不断向外扩散，但是辐射能量的功率是有限的，球壳越大，每个位置的能量密度就越低

![Games101_L7_25](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_25.png)

当球壳半径是单位半径的时候，我们设能量密度（或者说光强度，Intensity）是$I$（可以记为单位强度），如果传播到半径$r$的地方，那么光照强度就是$I/r^2$，这也是光强的衰减规律，我们可以根据这个公式，去计算有多少光真正的传播到某点

我们继续研究这个模型，如下图所示，设光源离着色点的距离为$r$，单位光强是$I$，那么传播到着色点的光照强度就是$I/r^2$，然后有多少能量被接受取决于夹角，所以根据余弦定理有$\max (0,\textbf{n}\cdot \textbf{l})$，使用Max函数是因为当夹角大于90度时，着色点不会接收到光线能量，所以这时候就是0

然后着色点会吸收一部分颜色（或者说这部分波长的光线），然后反射另一部分颜色，所以就可以显示不同的颜色，这里我们可以去描述和计算，定义一个系数$k_d$，或者叫漫反射系数，代表吸收能量的能力，表示了明暗，如果我们使用三维向量去表示这个系数，那么就可以代表三通道的明暗

![Games101_L7_26](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_26.png)

如果是漫反射，那么光线会向四面八方反射，那么人眼观察到的光线，就与观测方向无关了，在任何角度观测都是一样的，这时候，反射只与物体和光源有关

当然，漫反射的假设是光线照射到物体表面会被吸收并且均匀分布到各个方向

我们可以看一下当$k_d$变化时候的情况，当系数增大的时候，物体就会变亮（如下图所示）

![Games101_L7_27](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L7_27.png)

当然$k_d$目前是一个相对简单的模型，不太考虑物理真实性，不是一个准确的物理写法，是一个经验模型

## 镜面反射

当然，现实世界中不止有漫反射，还有镜面反射（会产生高光），高光产生的原因就是物体表面比较光滑，并且反射光线方向很接近观测方向，其中Blinn-Phong反射模型是一种常用的局部光照模型，用于计算一个表面在给定光照条件下的颜色。这个模型的反射部分被分解为环境光照、漫反射和镜面反射（即 Specular Term）三个部分。phong高光是用观察方向和反射方向夹角，blinn是中线和法线夹角

Blinn-Phong观察到，当我们的观察方向接近镜面反射方向$R$的时候，就说明法线方向和半程向量很接近，半角向量(Halfway vector)是在Blinn-Phong光照模型中使用的一个向量，用于替代Phong模型中的反射向量，以提高计算效率

![Games101_L8_7](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_7.png)

我们可以求入射方向和观测方向的角平分线方向，也就是半程向量，这种方式很简单，入射方向和观测方向的向量直接相加即可得到方向，然后进行归一化操作就可以得到单位向量

半程向量和法线向量的接近，代表了反射方向和观察方向的接近，或者说，能否观测到高光，只需要看半程向量和法线接近与否就可以，这样更容易计算，因为相对来说，计算半程向量的复杂度，比根据光照方向和法线方向计算反射方向的复杂度低很多

同时，一般认为高光就是白色，所以镜面反射系数$k_s$就是一个白色的标量

不过Blinn-Phong模型还是一种经验模型，主要是为了判断是否可以看到高光

当然，cos还有一个指数项$p$，这是因为向量夹角余弦的确可以体现向量接近程度，但是容忍度太高了，当向量夹角在45度时，高光基本上就应该消失了，但是下图中最左侧可以看到，直接使用cos的话，45度时也会有高光存在，这不符合实际情况，实际情况应该是，当方向很接近的时候，或者说在一个小范围内，高光才会清晰存在，所以就使用指数来压缩这个范围，在Blinn-Phong模型中通常会使用100-200的指数，这样在5度之外就基本上看不到高光了

![Games101_L8_8](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_8.png)

对不同的$k_s$和$p$进行调整，可以看到如下的效果

![Games101_L8_9](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_9.png)

## 环境光照

回想茶杯的明暗，在一些光源无法直射的地方仍然有一定亮度，这是因为光线会进行各种反射最后打到光源无法直射的地方，这些地方也可以接受到光线，当然这种计算也会很复杂，会涉及各种反射的计算，所以我们进行大胆的假设：所有的着色点接受到的环境光照是相同的，强度定义为$I_a$，同时定义一个环境光系数$k_a$

可以看出，环境光是没有入射方向概念的，这是一个四面八方的入射，并且跟观测方向无关，就是一个常数。这样就可以大大简化计算

![Games101_L8_10](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_10.png)

## 着色模型

这样，我们就可以将三种反射模型进行相加，就可以得到想要的反射模型，效果如下图所示，这也就是Blinn-Phong模型，或者说着色模型

![Games101_L8_11](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_11.png)

着色模型是考虑任何一个点，所以应该要对所有的点进行着色操作，实现场景着色

当然，注意一下，着色模型中不考虑观测点到着色点的距离

## 着色频率

### 概念

先看下图中的几个球，这几个球的几何表示是一模一样的，但是边界感明显不同，实际上这几个球的着色模型一样，只不过着色频率不同的

![Games101_L8_14](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_14.png)

如果我们将着色应用在一个面（比如说一个四边形面）上，每个面我们只做一次着色，就得到上图左的效果，看起来很粗糙；如果我们对每个面的顶点进行着色，那就可以得到上图中的效果，看起来更细致；如果我们对三角形内的点进行着色（或者说应用在每一个像素上），那么就可以得到非常细致的效果，如上图右所示，这里就用到一种插值的方法

## 对三角形着色

我们可以对每个三角形进行着色，因为每个三角形都是一个平面，并且很容易求出来其法线方向，然后就可以根据其他的参数计算出来三角形的着色情况

![Games101_L8_15](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_15.png)

但是如何求出三角形顶点的法线方向呢？我们先假设顶点法线方向可以计算，那么三角形内部的着色就可以通过插值的方式计算出来，效果（如下图所示）可以看出来，更加细致，但是当三角形较大的时候，其中的高光也不是很明显

![Games101_L8_16](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_16.png)

如果可以求出顶点法线，并且在每个像素上进行插值计算，那么就可以求出来更为细致的着色效果，不过这里的Phong Shading和Blinn-Phong不是同一个概念，前者是一种着色频率，后者是一种着色模型，只不过是同一个人发明的

![Games101_L8_17](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_17.png)

上面三种着色频率分别是逐三角形、逐顶点和逐像素，三种不同频率

顶点和像素的区别在于，顶点只求顶点的法向量并获得颜色，其余点根据顶点插值颜色；而像素是所有点都插值法向量然后获得颜色；像素的计算量更大

我们可以分析三角形密度和着色频率两个变量，可以发现，三角形密度（或者说几何模型复杂程度）越大，即使使用一些简单的着色模型，也可以达到很光滑的效果，或者说着色频率取决于面、顶点或者像素出现的频率，当出现的频率足够高的话，就不需要特别高的着色频率，如果几何模型足够复杂，三角形密度足够大，那么使用Flat着色一样可以表现出跟Phong着色相等的效果，不过前者的计算量可能会更大（如果三角形数量远超像素数量的话）

![Games101_L8_18](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_18.png)

## 逐顶点着色

前面我们提到了三种着色频率，但是也有一个问题，就是如何去计算顶点的着色呢？或者说如何计算顶点的法线方向呢

先从球面开始考虑，顶点肯定是在球面上的，这样法线就是球心到顶点的单位向量，但是实际上很难碰上一个正好的圆球，所以人们发明了一种方法

![Games101_L8_19](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_19.png)

一个顶点肯定与若干三角形进行关联，那么就可以认为，这个顶点的法线就是周围关联的面的法线的平均，这是一种很好用的方法

当然，考虑到三角形的面积可能是不同的，所以可以使用基于面积加权平均的顶点法线计算方法，比直接平均的效果更好一些

## 逐像素着色

在逐顶点计算之后，怎么进行逐像素的着色呢？

我们已经知道了两个顶点的法线，那么怎么计算内部的一个平滑的法线呢？下图就是计算方法，需要依赖重心坐标

![Games101_L8_20](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_20.png)

当然记得进行归一化

# 图形管线（Pipeline）

接下来我们将尝试将前面提到的所有东西进行应该整合，可以根据不同的三维模型和光照条件，就可以渲染出效果了，这个整合出来的东西就叫做**图形管线（Pipeline）**，或者可以称为**实时渲染（Real-time Rendering）**，第二种称呼更为现代化

实时渲染，实际上就是场景到显示图像的一个完整的过程，中间的这些操作就是一个管线（或者说表示一系列的操作）

![Games101_L8_22](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_22.png)

如上图所示，流程大概如下

1. 输入是一系列的三维空间中的点
1. 然后将这些点投影到屏幕空间上（投影过程中连接关系不变）
1. 然后将屏幕空间上的点连接为三角形
1. 使用光栅化操作进行离散化处理（包括采样和遮挡，找到可以在屏幕上显示的像素）
1. 进行着色操作并且在屏幕上显示

至于这里为什么说是把三维空间中的点投影到屏幕上然后连接为三角形，这是因为我们可以使用三角形来描述三维空间中的物体，对于任意一个三角形，都可以通过定义顶点的方式定义三角形，这样我们定义模型上的顶点，然后定义哪三个顶点可以构成一个三角形

不过着色操作有一些不同，在这里可以看到着色是发生在顶点和像素上，并且都会发生，如果是Gouraud Shading，每个顶点进行着色，那么着色就可以发生在顶点处理上，如果要做Phone Shading，那么将应该在投影之后再进行着色

当然，现在的GPU允许大家自行编程，去处理像素和顶点，也可以使用OpenGL去完成，这样可以设置一个通用的函数，或者称为着色器（Shader），每个顶点和像素都会自动执行

## 展望

或许后面的研究方向就会侧重于实时复杂三维场景的渲染了

![Games101_L8_33](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_33.png)

当然，对于游戏方向，虚幻引擎是一个很好用的工具，可以通过各种接口去实现角色的设计和动作等游戏开发，而不需要过分注重图形学算法实现

## 硬件

GPU是图形管线的硬件实现，便于实现图形学的底层操作，比如说光栅化等，其中着色器是可以编程的，比如说顶点着色器、像素着色器，但是随着技术发展，越来越多的着色器出现，比如说几何着色器（Geometry Shader），可以定义一些几何操作，比如说动态产生三角形，还有Compute Shader，可以完成任意计算操作或者说是一种通用计算器，不再拘泥于图形学的操作

![Games101_L8_34](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_34.png)

GPU就是高度并行化的计算器，并行度远超于CPU，非常时候做图形学的操作，因为渲染操作基本上是相同的

![Games101_L8_35](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_35.png)

# 纹理映射（Texture Mapping）

## 概念

在计算机图形学中，纹理映射是一种技术，用于增加图形渲染的详细程度和复杂性，而无需增加模型的多边形数。它通过将一张图像（纹理）应用到一个简单的形状或模型上，创造出更高级别的细节和复杂性。这种技术常常在三维建模和计算机游戏中使用，以提高图像的逼真度和视觉吸引力。

纹理映射是在干什么事情呢？我们先看下图中的球，我们可以看到不同的地方有不同的颜色，有黄色和蓝色，这是因为漫反射系数发生了改变，所以我们希望有一种方法，可以定义物体上任意一点的属性，比如说木质地板上就存在花纹（这些地方的漫反射系数不同于其他地方），这也就是我们引入纹理映射的思路

![Games101_L8_37](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_37.png)

我们首先确定一下纹理映射的基本概念，首先，纹理映射是定义在物体表面的，因为内部无法被看到，那么我们怎么去理解物体表面？首先我们认为，任何三维物体的表面都二维的，如下图所示，三维的地球仪的表面就可以在二维上表示，我们将地球仪的表面切割下来，就可以在二维上进行表示，或者说可以有一个一一对应的关系，那么我们就可以说，纹理就是一张图，并且具有弹性，纹理映射就是将这张图“套”到三维模型上或者贴到模型上的过程，或者可以这样理解，纹理就是把模型表面的衣服扒下来并且展开成的一个平面

![Games101_L8_38](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_38.png)

下图就是我们将纹理映射到独眼巨人模型上的情况

![Games101_L8_39](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_39.png)

我们可以设想，有一个纹理空间，我们怎么将其映射到模型上？

我们从模型的最基本单元出发，在模型上有一个位置已知的三角形，那么如何从物体上，映射到纹理上，也就是说我们如何确定其在纹理空间中的位置，上图中的任意一个三角形，都可以在纹理空间中找到自己的位置，当然这个会依赖于艺术家的创作

当然，我们还有其他的一些想法，首先就是希望，原来很小的三角形，映射到纹理空间之后也很小，然后希望保证颜色的无缝衔接（这也是一个很重要的研究方向——参数化），也就是在模型上无缝衔接的三角形，在纹理空间中也可以无缝衔接

## 纹理坐标系

纹理映射中有纹理坐标系的概念，这个通常使用$(u,v)$表示，下图是一个简单的颜色纹理

![Games101_L8_40](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_40.png)

此外，通常认为纹理坐标系的范围为0-1，这是为了方便处理，不需要管分辨率等因素

我们看一个纹理的实际应用，下图是一个早期的简单渲染图

![Games101_L8_42](C:\Users\Michael\Downloads\ML_DL_CV_with_pytorch\ComputerGraphic\assets\Games101_L8_42.png)

如果我们将每个点的纹理坐标给显示出来，就是下图中的效果

![Games101_L8_43](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_43.png)

我们可以看到，纹理不断重复，就跟贴瓷砖一样，这样就可以贴满一个模型，当然，上图中的纹理实际上是设计地很好的，在自我复制的时候可以无缝衔接，这种纹理称为tiled texture

![Games101_L8_44](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L8_44.png)

当然，无缝衔接纹理的合成也是一个很重要的方向

下一个问题是很自然的，就是我们知道三角形顶点的纹理坐标，那么我们怎么确定三角形内部任意一个点的纹理坐标（要平滑过渡），这里就需要插值和重心坐标的方法

## 插值与重心坐标

我们有很多操作都在顶点上操作，但是我们希望在得到顶点的属性之后，我们可以计算三角形内部任意一点的属性，实现一个平滑过渡

## 重心坐标

重心坐标是定义在三角形上的，当三角形改变的时候，重心坐标的数值也会相应的改变

![Games101_L9_7](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_7.png)

如上图所示，三角形所在平面上的点的坐标可以任意的表示为三个顶点坐标的线性组合，条件是系数之和为一，如果点在三角形内部，那么三个系数非负

这样我们就可以建立一个新的坐标系，不需要管三角形的具体坐标，任意点都可通过顶点属性的线性组合得到自身的属性，只需要给你三个顶点就可以，这也称为**重心坐标（Barycentric coordinates）**，这样我们就可以通过$(\alpha,\beta,\gamma)$来表示一个点

当然，重心坐标是可以通过面积比求出的，如下图所示，我们将点和三个顶点都进行连线，可以发现，任何一个顶点对面都有一个三角形，定义点A对面的三角形的面积为$A_A$，然后我们就可以得到重心坐标

![Games101_L9_9](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_9.png)

这样，我们就可以使用重心坐标去完成属性计算了，对于任意的属性，我们都可以通过重心坐标进行线性组合得到

![Games101_L9_12](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_12.png)

当然，重心坐标虽然简单，但是在投影下，不能保证重心坐标不会改变，这也是重心坐标的一个缺点

## 渲染应用

我们如何将重心坐标应用在渲染中呢？或者说怎么将纹理应用在渲染当中？

![Games101_L9_14](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_14.png)

对于屏幕上任意一个采样点，其具有一个位置$(x,y)$，对像素来说就是其中心点坐标

然后我们就可以知道其在这个位置上插值出来的纹理坐标$(u,v)$

最后我们在纹理上查询一下这个采样点的属性，就可以得知其纹理（可以认为就是漫反射系数$K_d$）了，然后就可以显示出来高光、哑光、贴图等一系列效果

## 纹理放大

当然，在纹理映射的时候也会出现一些情况，比如说屏幕分辨率很高，但是纹理分辨率很低的情况，或者说纹理太小了，就会产生一些被拉大的情况，比如说下图所示的左边图片就有这种情况，中间图片和右边图片是进行了不同插值方法所展现的效果，中间是双线性插值，右边图是

![Games101_L9_16](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_16.png)

这种情况出现的原因，是纹理太小，导致纹理空间在查询坐标的时候会查询到非整数的值，纹理就会被拉大，就会出现这种像素化、失真的情况（这是因为纹理也是使用像素存储的）

纹理上的像素是有名字的，称为**纹理元素或者纹素（texel）**

因为查询坐标的时候会出现非整数的值，所以我们希望会出现一些插值的方法去进行计算

下图中，红色点是我们希望纹理采样的地方，黑色点是纹理坐标系的像素

![Games101_L9_17](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_17.png)

一个很简单的方法就是直接去寻找最近的点，但是这样是不足的，所以我们希望去寻找相邻的四个点，然后测量红色点到左下角黑色点的水平距离s和垂直距离t，易知其范围都为0-1

![Games101_L9_19](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_19.png)

下一步我们就定义线性插值操作lerp

![Games101_L9_20](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_20.png)

然后通过两次水平的线性插值就得到了红色点上下方的点的值$u_1,u_2$，然后再一次竖直方向插值，就可以得到红色点的值，并且还是综合考虑了周围四个点的值，这也被称为**双线性插值（Bilinear）**，因为是进行了两次线性的插值，实现了一个平滑过渡

![Games101_L9_22](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_22.png)

甚至还可以考虑周围十六个点的属性来进行双线性插值

双线性插值的效果更好，但是也会带来更大的计算开销

另一个问题是纹理过大的话，会有什么问题？纹理过大反而会带来更大的问题，如下图所示，我们有一个向远处无限延伸的平面，上面贴的纹理是格子花纹，并且近处的格子大，远处的格子小，那么我们还是以以前的方法应用纹理并且进行简单操作的话，会得到右边的效果图，包含了摩尔纹和锯齿的走样效果

![Games101_L9_25](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_25.png)

问题是这样出现的，屏幕上同样一个像素，对于不同距离的物体，覆盖的范围不一样，对于远处的物体，一个像素可以覆盖很大一个范围，对于近处的物体，一个像素只能覆盖一个很小的范围，同样的，覆盖的纹理区域的大小也是不一样的

![Games101_L9_26](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_26.png)

这种情况出现的原因就是信号变化过快，采样频率跟不上了，所以一个解决方法就是超采样，但是问题就是计算开销过大

当然，如果采样会带来走样问题，那么我们是否可以不采样？那么如何避免采样呢？我们知道，对于远处的地方，一个像素就可以覆盖很大一个区域，那么我们可以求得这个区域中的平均值，以此代替采样，那么我们怎么去计算任给区域的平均值呢？这个是一个算法问题，在计算几何和数据结构等方面都有应用，也就是**点查询（Point Query）**和**范围查询（Range Query）**两个概念

![Games101_L9_29](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_29.png)

点查询很好理解，比如说给一个纹理坐标，通过插值等方法就可以得到纹理属性

范围查询就是给一个区域，不用采样就立刻得到范围内的平均值，这是一个经典问题，图形学中有一个非常好的近似算法去完成这个，当然，范围查询不止是查询平均值，还有查询最值的范围查询

## Mipmap

这是图形学中的一个重要算法，允许我们去完成范围查询，优点是快，但是是一种近似查询，并且只能查询正方形的范围

mip这个词在拉丁语中是很多小东西的意思，mipmap就是使用一张图生成一系列的图

原始纹理也称为第0层，每增高一层，分辨率就减小一倍，如下图所示

![Games101_L9_32](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_32.png)

在渲染之前，先对纹理进行处理，生成好mipmap，当然，这种方法也被称为图像金字塔（下图所示），同时不会引入太多额外存储，其他的层加起来只有原纹理三分之一的存储量

![Games101_L9_33](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_33.png)

那么我们怎么去计算mipmap呢

![Games101_L9_34](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/Games101_L9_34.png)
