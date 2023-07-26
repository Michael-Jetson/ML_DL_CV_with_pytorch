> 笔者：本笔记涉及GAMES-现代计算机图形学课程（闫令琪主讲，UCSB）中 "Leture 01 Overview of ComputerGraphics" 这部分的要点笔记。笔记的内容可能并不详尽，但我自问已经足够用心，所有的笔记内容会尽快更新完成。
> 
> 题外话：目前我对计算机图形学，计算机视觉和人机交互这三个领域很感兴趣，在准备考研&留学的同时学习这三个领域的基本知识。如果您是在学习这些领域内知识的学生，欢迎私信交流学习心得。如果您是一名做这三个领域的老师，正在招收这些方向的研究生，欢迎查看我的[个人主页](https://yulongc.github.io/homepage/)并通过邮箱与我取得联系，虽然我才疏学浅，但希望得到一个机会继续我的学习，非常感谢！

导言：简单直观的看出什么样的画面是好的画面，亮度高的画面是好的画面，说明画面的全局光照设计得很好。

# 1. 图形学应用：

- Video Games(只狼, 无主之地, 艾尔登法环等)
- Movies(黑客帝国, 阿凡达等)
- Animations(疯狂动物城, 冰雪奇缘等)
- Design(Autodesk Gallary, 装修设计等)
- 可视化(Science，engineering, medicine，journalism等)
- VR(Oculus VR, Microsoft Hololens等)
- Digital lllustration
- Simulation(The Dust Bowl phenomena 沙尘暴, Black hole from Interstellar 黑洞等)
- Graphical User Interfaces
- Typography

# 2. Why we Study?

- Fundamental Intellectual Challenges:
  
  - 深入理解真实世界，去创造更加真实的虚拟世界
  
  - 了解新的计算和显示的方法与技术

- 技术挑战(Technical Challenges):
  
  - 各种各样的数学方法，包括曲线，曲面的数学基础等
  
  - 物理基础，光照，阴影等
  
  - 描述和操作三维形体
  
  - 制作动画/仿真

# 3. 课程内容?

## 3.1 光栅化 (Rasterization)

作用是如何将三维空间的几何形体(3D triangles/polygons)显示在屏幕上。实时（每秒生成约30幅画面，也即30帧，否则成为离线）的图形学会广泛应用光栅化的方法，主要是进行投影。

![](https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture01/Lec01_p0.png)

## 3.2 几何知识 (Curves and Meshes)

如何表示一条光滑的曲线与曲面，如何用简单的曲面通过细分的方法得到更复杂的曲面，形状变化的时候如何保持物体的拓扑结构等等......

![](https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture01/Lec01_p1.png)

## 3.3 光线追踪（Ray Tracing）

动画，电影和游戏普遍用到的技术，RTX ON！在实时光线追踪之前，光线追踪可以生成质量更高的画面，但是速度较慢，实时光线追踪为了trade off两者（质量与速度）而诞生。

![](https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture01/Lec01_p2.png)

## 3.4 动画/仿真(Animation / Simulation)

![](https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture01/Lec01_p3.png)

## 3.5 碎碎念

- 这门课不会详细的教你上手OpenGL/DirectX/Vulcan等API工具，老师希望学生理解这些API的内部原理，比如如何从三维到二维做投影，在学习原理之后可以快速上手使用这些API。(We learn Graphics, not Graphics APIs！)
- 这门课不会详细的讲解CV和DL的详细内容，至于如何区分CV和CG，下面的图是老师的理解，我结合老师的课件和课上的内容的理解是：需要猜测的内容是CV，需要表达的内容是CG。

![](https://github.com/Michael-Jetson/ML_DL_CV_with_pytorch/blob/main/GAMES101Notes/Lecture01/Lec01_p4.png)

# Reference

[GAMES101: 现代计算机图形学入门 (ucsb.edu)](https://link.zhihu.com/?target=https%3A//sites.cs.ucsb.edu/~lingqi/teaching/games101.html)
