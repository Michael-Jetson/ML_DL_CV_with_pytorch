> 笔者：本笔记涉及GAMES-现代计算机图形学课程（闫令琪主讲，UCSB）中 "Leture 02 Review of Linear Algebra" 这部分的要点笔记。笔记的内容可能并不详尽，但我自问已经足够用心，所有的笔记内容会尽快更新完成。  
> 题外话：目前我对计算机图形学，计算机视觉和人机交互这三个领域很感兴趣，在准备考研&留学的同时学习这三个领域的基本知识。如果您是在学习这些领域内知识的学生，欢迎私信交流学习心得。如果您是一名做这三个领域的老师，正在招收这些方向的研究生，欢迎查看我的[个人主页](https://yulongc.github.io/homepage/)并通过邮箱与我取得联系，虽然我才疏学浅，但希望得到一个机会继续我的学习，非常感谢！

# 1. 开篇：旋转举例 An Example of Rotation

如图，蜗牛的旋转速度在不断的变化 (发的是一张静态图片，动态去尾部Reference的课程链接中找)，并不是以固定速度旋转的。

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p0.png" title="" alt="" data-align="center">

# 2. 向量/矢量 (Vectors)

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p1.png" title="" alt="" data-align="center">

- 向量的书写方式有很多种，可以写成$\vec{a}$，也可以写成$\bold{a}$。除此之外，如果知道起始点(A)坐标和结束点(B)坐标，也可以用两者的坐标差($\vec{AB}=B-A$)来表示。

- 向量最重要的属性是方向和长度。

- 向量在空间中平移后不改变向量本身，因为向量是根据起始点和结束点的相对位置确定的，也即向量没有完全固定的起始点。

- 如何表示向量的长度/大小：$||\vec{a}||$

- 如何表示向量的方向（不关心其长度/大小）：单位向量$\hat{a}=\vec{a}/||\vec{a}||$（方向与原向量方向一致，而长度限定为1）

# 3. 向量/矢量相加 (Vector Addition)

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p2.png" title="" alt="" data-align="center">

- 几何直观上：平行四边形法则和三角形法则（平行四边形法则可以通过平移转化为三角形法则，只需要做到相加的向量首尾相连即可，最终利用完全相连后的初始和结束位置得到向量）。

- 代数上：直接将坐标进行加和即可。

- 向量在坐标系下的表示（笛卡尔坐标系内，在坐标系中表示的原因是方便计算向量长度）

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p3.png" title="" alt="" data-align="center">

- 图中，X和Y分别是横轴和纵轴的单位向量，现在用这两个单位向量来表示向量$\vec{A}$ (如果没有特殊说明，向量默认是列向量表示)
  
  - 列向量表示：$\vec{A}=\left(\begin {array}{c} x \\ y \\ \end{array}\right)$
  
  - 列向量转置：$\vec{A}^T=(x,y)$
  
  - 向量的长度/大小：$||A||=\sqrt{x^2+y^2}$

# 4. 向量乘法 (Vector Multiplication)

## 4.1 点乘 (Dot(scalar) Product)

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p4.png" title="" alt="" data-align="center">

- 计算公式（得到的是一个数值）：$\vec{a}\cdot\vec{b}=||a||||b||\cos\theta$

- 换个思路看这个公式，你会发现点乘公式在图形学中的重要应用就是利用向量去计算它们之间的夹角：$\cos\theta=\frac{\vec{a}\cdot\vec{b}}{||a||||b||}$，如果是单位向量，则向量的点乘就是它们的余弦值大小：$\cos\theta=\hat{a}\cdot\hat{b}$。

- 点乘运算满足的性质
  
  - 交换律：$\vec{a}\cdot\vec{b}=\vec{b}\cdot\vec{a}$
  
  - 分配律：$\vec{a}\cdot(\vec{b}+\vec{c})=\vec{a}\cdot\vec{b}+\vec{a}\cdot\vec{c}$
  
  - $(k\vec{a})\cdot\vec{b}=\vec{a}\cdot(k\vec{b})=k(\vec{a}\cdot\vec{b})$

- 点乘运算在笛卡尔坐标系下的表示
  
  - In 2D
    
    - $\vec{a} \cdot \vec{b}=\left(\begin{array}{l}
      x_{a} \\
      y_{a}
      \end{array}\right) \cdot\left(\begin{array}{l}
      x_{b} \\
      y_{b}
      \end{array}\right)=x_{a} x_{b}+y_{a} y_{b}$
  
  - In 3D
    
    - $\vec{a} \cdot \vec{b}=\left(\begin{array}{l}
      x_{a} \\
      y_{a} \\
      z_{a}
      \end{array}\right) \cdot\left(\begin{array}{l}
      x_{b} \\
      y_{b} \\
      z_{b}
      \end{array}\right)=x_{a} x_{b}+y_{a} y_{b}+z_{a} z_{b}$

- 图形学中点乘的应用1：向量投影，假设一束平行的光向$\vec{a}$照过来，由于$\vec{b}$的遮挡，$\vec{b}$会在$\vec{a}$上形成投影$\vec{b}_\bot$。
  
  - $\vec{b}_\bot$的方向必定与$\vec{a}$一致，即$\vec{b}_\bot=k\hat{a}$
  
  - $k=||\vec{b}_\bot||=||\vec{b}||\cos\theta$
  
  - 取得投影后，就可以将$\vec{b}$进一步分解出一个垂直于$\vec{a}$的法线向量$\vec{b}-\vec{b}_\bot$。

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p5.png" title="" alt="" data-align="center">

- 图形学中点乘的应用2：向量的点乘会告诉向量之间方向的一致性和接近程度，根据$\\cos\theta$来看，如果最终得出来点乘的结果<0，说明两向量方向基本相反（$\vec{a}$和$\vec{c}$）；如果最终的出来点乘的结果>0，说明两向量方向基本一致（$\vec{a}$和$\vec{b}$）；如果最终结果=0，说明两向量垂直。接近程度的话用点乘（先各自转化为方向向量再去点乘）得到的值和1/-1的接近程度就可以表示，这里不做过多描述。

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p6.png" title="" alt="" data-align="center">

## 4.2 叉乘 (Cross(vector) product)

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p7.png" title="" alt="" data-align="center">

- 长度计算公式：$||a×b||=|||a|||b||\sin\phi$

- 叉乘产生的新向量的方向由右手螺旋定则确定，$\vec{a}×\vec{b}$就是从$\vec{a}$的方向向$\vec{b}$的方向螺旋，大拇指的方向就是新向量的方向。

- 叉乘运算满足的性质：
  
  - $\vec{a}×\vec{b}=-\vec{b}×\vec{a}$
  
  - $\vec{a}×\vec{a}=\vec{0}$
  
  - 分配律：$\vec{a}×(\vec{b}+\vec{c})=\vec{a}×\vec{b}+\vec{a}×\vec{c}$
  
  - 结合律：$\vec{a}×(k\vec{b})=k(\vec{a}×\vec{b})$

- 多应用于建立三维空间中的直角坐标系（以x,y,z轴举例）
  
  - $\vec{x}×\vec{y}=+\vec{z}$
  
  - $\vec{y}×\vec{x}=-\vec{z}$
  
  - $\vec{y}×\vec{z}=+\vec{x}$
  
  - $\vec{z}×\vec{y}=-\vec{x}$
  
  - $\vec{z}×\vec{x}=+\vec{y}$
  
  - $\vec{x}×\vec{z}=-\vec{y}$

- 代数表示（笛卡尔坐标系下）：$\vec{a} \times \vec{b}=\left(\begin{array}{l}
  y_{a} z_{b}-y_{b} z_{a} \\
  z_{a} x_{b}-x_{a} z_{b} \\
  x_{a} y_{b}-y_{a} x_{b}
  \end{array}\right)$

- 稍后会讲到的计算形式（将$\vec{a}$假设成矩阵）：$\vec{a} \times \vec{b}=A^{*} b=\left(\begin{array}{ccc}
  0 & -z_{a} & y_{a} \\
  z_{a} & 0 & -x_{a} \\
  -y_{a} & x_{a} & 0
  \end{array}\right)\left(\begin{array}{l}
  x_{b} \\
  y_{b} \\
  z_{b}
  \end{array}\right)$

- 图形学中叉乘的作用1：判断左/右，下图处于一个三维空间中的直角坐标系，$\vec{a}$和$\vec{b}$同在xoy平面内。如果$\vec{a}×\vec{b}$的结果为正，即与z轴正方向相同，说明$\vec{a}$在$\vec{b}$的右边；如果结果为负，即与z轴反方向相同，说明$\vec{a}$在$\vec{b}$的左边。

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p8.png" title="" alt="" data-align="center">

- 图形学中叉乘的作用2：判断内/外，下图中，无论是$\vec{AB}×\vec{AP}$，$\vec{BC}×\vec{BP}$，还是$\vec{CA}×\vec{CP}$，结果都能证明P点位于AB，BC和CA这三条边的左侧，说明P点位于三角形内。（注：无论三角形三个点的顺序如何排列，判断标准都是P点是否在三条边的同一侧，拓展到光栅化，通过计算三角形覆盖了哪些像素点来给像素着色；若计算结果为0，则人为自行决定点的归属）

<img src="https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture02/Lec02_p9.png" title="" alt="" data-align="center">

## 4.3 正交标准坐标系 (orthonmormal frame)

- 用三个单位向量表示正交标准坐标系，对这三个向量的要求如下：
  
  - $||\vec{u}||$ = $||\vec{v}||$ = $||\vec{w}||$ = 1
  
  - $\vec{u}\cdot\vec{v}$ = $\vec{v}\cdot\vec{w}$ = $\vec{u}\cdot\vec{w}$ = 0
  
  - $\vec{w}$ = $\vec{u}×\vec{v}$  (right-haned)

- 坐标系中的任意一个向量都可以用这三个向量表示，表示的方法采用向量在这三个单位向量上的投影分别乘上这三个单位向量后相加：$\vec{p}=(\vec{p} \cdot \vec{u}) \vec{u}+(\vec{p} \cdot \vec{v}) \vec{v}+(\vec{p} \cdot \vec{w}) \vec{w}$

# 5. 矩阵

- 简单定义：多行或多列向量组成的平面结构

- 矩阵和单个数据元素的乘积等于矩阵的每个元素都乘以这个数据元素(element by element)得到的新矩阵。

- 矩阵与矩阵乘积
  
  - 前提条件：两个矩阵能够相乘，必须保证第一个矩阵的列数等于第二个矩阵的行数。举个例子，A是一个(M×N)的矩阵，B是一个(N×P)的矩阵，两者乘积得到的矩阵形状为(M×P)。
  
  - 乘积得到新矩阵的元素：求新矩阵第x行第y列的元素值，就去将第一个矩阵的第x行向量与第二个矩阵的第y列向量进行点积即可。举个例子，26是新矩阵的第2行第4列的元素，去第一个矩阵找到第2行向量(5 2)，再去第二个矩阵找到第4列向量(4 3)$^T$，再将两者进行点积即可得到26。

$\left(\begin{array}{ll}
1 & 3 \\
5 & 2 \\
0 & 4
\end{array}\right)\left(\begin{array}{llll}
3 & 6 & 9 & 4 \\
2 & 7 & 8 & 3
\end{array}\right)=\left(\begin{array}{cccc}
9 & ? & 33 & 13 \\
19 & 44 & 61 & 26 \\
8 & 28 & 32 & ?
\end{array}\right)$

- 矩阵运算满足的性质：
  
  - 注：矩阵运算并不满足交换律(Non-commutative)，也即AB和BA并不是相同的运算。
  
  - 结合律：(AB)C = A(BC)
  
  - 分配律：A(B+C) = AB + AC
  
  - 矩阵和向量相乘的时候永远把向量视作一个列向量。举个例子，二维形状按y轴进行对称操作（保持y不变，x变为其相反数）：
  
  $\left(\begin{array}{cc}
  -1 & 0 \\
  0 & 1
  \end{array}\right)\left(\begin{array}{l}
  x \\
  y
  \end{array}\right)=\left(\begin{array}{c}
  -x \\
  y
  \end{array}\right)$
  
  - 矩阵转置：
    
    - 单个矩阵的转置就是行列互换
    
    - 两个矩阵相乘后转置：$(AB)^T=B^TA^T$

- 单位矩阵与逆(Identity Matrix and Inverses)
  
  - 单位矩阵举例：$I_{3 \times 3}=\left(\begin{array}{ccc}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{array}\right)$
  
  - 逆：$\begin{array}{l}A A^{-1}=A^{-1} A=I \\\end{array}$
  
  - 逆运算：$\begin{array}{l}
    (A B)^{-1}=B^{-1} A^{-1}
    \end{array}$

- 向量乘积的矩阵形式(Vector multiplication in Matrix form)
  
  - 点乘与叉乘：

$\begin{aligned}
& \vec{a} \cdot \vec{b}=\vec{a}^{T} \vec{b} \\
= & \left(\begin{array}{lll}
x_{a} & y_{a} & z_{a}
\end{array}\right)\left(\begin{array}{l}
x_{b} \\
y_{b} \\
z_{b}
\end{array}\right)=\left(x_{a} x_{b}+y_{a} y_{b}+z_{a} z_{b}\right)
\end{aligned}$

$\vec{a} \times \vec{b}=A^{*} b=\left(\begin{array}{ccc}
0 & -z_{a} & y_{a} \\
z_{a} & 0 & -x_{a} \\
-y_{a} & x_{a} & 0
\end{array}\right)\left(\begin{array}{l}
x_{b} \\
y_{b} \\
z_{b}
\end{array}\right)$

# Reference

[GAMES101: 现代计算机图形学入门](https://link.zhihu.com/?target=https%3A//sites.cs.ucsb.edu/~lingqi/teaching/games101.html)
