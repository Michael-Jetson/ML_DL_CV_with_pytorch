# 概述

## 高精地图

什么是高精地图（HD map）？

- 数据精度为分米甚至厘米级，包含空间信息、语义信息和时间信息的数据体。
- 空间信息—点云地图
- 语义信息—车道线、停止线、转向路标、速度标识、人行横道、路牙
- 时间信息—红绿灯信息，早晚可变车道信息，就是带时间的语义信息单独抽离

高精地图有Vector map、lanelet2、opendrive、Nds等格式，不同格式间的高精地图可以相互转化的

## 经典建图算法概述

### 点云地图创建

输入点云数据一般基于激光雷达坐标系，激光雷达坐标系和车身坐标系为刚性连接。如果将车辆起始位置当
成地图坐标系的原点，那么在之后的运动过程中的某些间隔均匀的时刻，如果能够准确的获取车辆的位姿变
化，就能将原本基于激光雷达坐标系的点云信息转换到地图坐标系下，进而就构成了点云地图的一部分。以
此往复，就会构成一张庞大的点云地图。

那么如何能够准确的获取车辆的位姿变化？有以下方法

- RTK差分定位（成本较高，需要搭建基站，不能够有遮挡）
- 激光slam匹配算法 （存在累积误差，长时间使用会造成漂移）
- 轮速计定位 （存在累积误差，长时间使用会造成漂移）
- 实际使用时，根据不同场景混用

工具可以在点云地图上标注语义信息的工具有Autoware tools、Unity插件版、VTD、RoadRunner等（前两款免费，但是Autoware Tools不太好用）

### 高精地图生产流程

大概这样的流程

![Autoware.ai_L2_5](/home/pengfei/文档/ML_DL_CV_with_pytorch/AutoDriving/assets/Autoware.ai_L2_5.png)

### 经典SLAM算法介绍

经典激光slam算法概述

- 视觉slam（VIO）：orbslam、vins、svo、dso
- 激光slam（2d）：gmapping、hector、Karto、cartographer2d
- 激光slam（3d）：loam系、cartographer3d、Ndt

本课程中我们仅仅关注建立3d点云地图的激光slam算法，尤其是Ndt算法

在前面我们有讲到，建图的关键在于位姿变化的准确估计，对于slam算法而言，位姿变化的计算是通过
点云特征匹配优化后得出的。
根据特征匹配形式的分类

- Scan to Scan：loam系
  Loam会将输入scan中的点云根据曲率大小分为平面点和边缘点，之后的匹配优化过程也是针对当前输入scan和上一scan的平面点和边缘点来研究进行的。根据边缘点的距离优化公式和平面点的距离优化公式来构造优化方程求解位姿变化量。Lego-loam、lio-sam等都是基于这一原理来进行位姿优化求解的、只不过他们引入了更多传感器并加入了回环检测。

- Scan to Map ：Cartographer、Ndt
  二者都是通过当前scan同已经建好的map（或者submap）来进行特征匹配的，和loam提取有曲率特征的点云不同，Cartographer将当前scan通过hit的方式来和上一次建好的submap来进匹配优化；而Ndt则是将map网格化后计算每个网格的均值方差，并通过当前scan中的每个点落在map网格中的正太分布概率来进行匹配优化的。

# map_file模块解析

解析流程如下

![Autoware.ai_L2_10](/home/pengfei/文档/ML_DL_CV_with_pytorch/AutoDriving/assets/Autoware.ai_L2_10.png)

map_file是一个ros package，有两个节点，分别负责读取pcd点云文件和csv语义地图文件，两个节点有三个输出的话题，如下图所示

![Autoware.ai_L2_11](/home/pengfei/文档/ML_DL_CV_with_pytorch/AutoDriving/assets/Autoware.ai_L2_11.png)