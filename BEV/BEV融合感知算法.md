# 融合背景

我们之前提到，BEV感知是一个建立在众多子任务上的一个概念，本身就很综合，输入宽泛（从主动传感器到GNNS/GPS等），我们可以把BEV算法进行一个分类

其中，以相机为输入的算法为BEV Camera，代表作品有BEV Former、BEVDet，以点云为输入的算法称为BEV LiDAR，代表作品有PV-RCNN系列，以图像和点云混合输入的算法称为融合感知方法，代表作有BEV Fusion

那么为什么要融合？我们希望融合可以达成什么目的？实际上，融合意味着模态信息互补，稳定，而且单模态有无法弥补的劣势，比如说相机主要是在前视图比较低的位置捕获，并且在复杂场景存在遮挡问题，存在信息丢失的情况，激光雷达受限于机械结构，在不同距离有不同分辨率，意味着采样点的数目会随着距离变化而变化，而且容易受到异常天气干扰 

![AutoDriveHeart_BEV_L3_6](/home/pengfei/文档/ML_DL_CV_with_pytorch/BEV/assets/AutoDriveHeart_BEV_L3_6.png)

## 融合思路介绍

我们的思路主要是在什么阶段进行融合

- 前融合（数据级融合）：直接融合不同模态的原始数据，比如说点云投影到图像的二维空间，生成伪2D图，或者生成伪点云的方法
- 深度融合（特征级融合）：点云和图像各自通过网络得到特征，然后在特征空间上融合
- 后融合（目标级融合）：偏向于后处理的方法，使用NMS等方法进行融合，是决策级的融合

![AutoDriveHeart_BEV_L3_8](/home/pengfei/文档/ML_DL_CV_with_pytorch/BEV/assets/AutoDriveHeart_BEV_L3_8.png)

我们可以更深入一些，划分的更详细一些

如下图所示，上面是图像分支，可以输入RGB图和灰度图，然后就是特征，特征的范围很广，有图像特征、深度图、分割，还有物体层面的检测结果

下面是点云分支，情况也类似，不过具体的种类有区别，数据可以是伪点云、点云、体素、二维雷达图像

![AutoDriveHeart_BEV_L3_9](/home/pengfei/文档/ML_DL_CV_with_pytorch/BEV/assets/AutoDriveHeart_BEV_L3_9.png)

点云和图像各有三个模块，可以两两之间融合，但是只是同层级的融合，那么我们能不能做不同层级的融合呢？实际上是可以的，这种方法称为非对称融合

如下图所示，第一种（左上）是点云和图像语义特征融合，我们将图像经过分割处理得到的前景点像素

![AutoDriveHeart_BEV_L3_10](/home/pengfei/文档/ML_DL_CV_with_pytorch/BEV/assets/AutoDriveHeart_BEV_L3_10.png)