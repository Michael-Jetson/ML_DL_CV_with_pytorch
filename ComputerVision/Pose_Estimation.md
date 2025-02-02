# 人体姿态估计

人体姿态估计是计算机视觉领域的一个方向，任务是识别出图像中人体的姿态，主流方式是确定人体上若干个关键点（比如说关节、眼睛等）的位置并且进行相连，对人体姿态的估计有利于完成一系列更高级别的任务，如动作识别

## 应用

人体姿态估计的一个应用就是人机交互，让计算机可以理解人在做什么动作并且做出相关的响应，比如说识别舞蹈动作，然后就是视频监控领域，可以识别行为并且做出判断，比如说检测老人跌倒的情况

## 挑战

人体姿态估计的一些主要问题如下

1. 关节或者说关键点之间具有强烈的连接关系
2. 关键点（关节）存在不可见（或者遮挡）和形状微小的问题
3. 需要从上下文中获取某些信息
4. 人的尺度不一样，一个图片中可能存在多种大小的人

## 数据集

目前主要的人体姿态估计数据集有下面几种，他们所拥有的关键点数量不同

![JulyEdu_KeyPointDetection_PangYan_L2_5](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_5.png)

以MSCOCO为例，这个数据集有十七个关键点

![JulyEdu_KeyPointDetection_PangYan_L2_6](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_6.png)

MSCOCO数据集网页，中间是图片数据集，右边是标注文件

![JulyEdu_KeyPointDetection_PangYan_L2_7](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_7.png)

标注文件的格式是JSON，可以直接在网页中打开或者使用Python读取，其中分为五个部分

1. "info"：提供一些数据集的信息，比如说年份、版本、描述等
2. "licenses"：授权信息
3. "images"：图像
4. "annotations"：标注
5. "categories"：类别

![JulyEdu_KeyPointDetection_PangYan_L2_8](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_8.png)

下面是标注文件前三个信息的情况

![JulyEdu_KeyPointDetection_PangYan_L2_9](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_9.png)

主要要注意"categories"中的信息

- supercategory：类别的超类别，因为是人体姿态估计数据集，所以是person类
- id：区分类别的ID
- name：类别名
- keypoints：有哪些关键点的名称
- skeleton：代表骨骼，表示哪两个关键点可以表示一个骨骼，可以进行相连，比如说[16,14]表示第16个关键点和14个关键点可以相连

![JulyEdu_KeyPointDetection_PangYan_L2_10](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_10.png)

然后就是标注信息

`annotations`：包含每个图像标注的信息，每个标注的信息都是一个字典，包含以下字段：

- `id`：标注的唯一标识符。
- `image_id`：与此标注关联的图像的ID。
- `category_id`：与此标注关联的类别的ID（在人体姿态估计任务中，类别通常为“人”）。
- `segmentation`：对象的边缘分割信息，用一系列的连续x，y坐标的点对表示。
- `area`：标注区域的面积（像素）。
- `bbox`：标注对象的边界框，表示为[x，y，width，height]，其中x和y是边界框左上角的坐标。
- `iscrowd`：表示该标注是否来自一群对象或者说一群人。如果是，那么`iscrowd=1`，否则`iscrowd=0`。
- `keypoints`：表示人体关键点的数组，该数组大小为51，每三个数字表示一个关键点，分别为[x坐标，y坐标，可见性]。可见性有三个可能的值：0表示关键点不可见且没有被标注，1表示关键点不可见但已被标注，2表示关键点可见且已被标注。在COCO数据集中，通常有17个关键点，包括眼睛、耳朵、鼻子、肩膀、肘部、手腕、臀部、膝盖和脚踝。
- `num_keypoints`：图像中被标注的关键点数量。

![JulyEdu_KeyPointDetection_PangYan_L2_11](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_11.png)

当然

![JulyEdu_KeyPointDetection_PangYan_L2_23](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_23.png)

头部的点范围最小，比如说眼睛耳朵，身体上的点就更大，尤其是臀部

## 历史

最早的人体姿态估计的，可以追溯到计算机视觉的早期，一种经典的方法就是**Pictorial Strictures (PSs，形变部件模型)**，早在1973年就由Fischler和Elshlager提出，对人体姿态估计领域影响深远，并且被Felzenszwalb和Huttenlocher改造的更加易用（通过距离变换方式）。它的基本思想是将目标对象（如人体）模型化为一个多部件的结构，并使用图形模型（通常是树形图）来表示部件之间的关系。

当然，这种方法也有限制，比如说需要手工设计特征等，所以很难大规模应用，在这个时期，人们主要研究的重点是保持易处理性并且丰富模型的表征能力，比如说一些可以表示复杂关节点关系的模型

深度学习背景的人体姿态方法估计在2014年被提出，即DeepPose: Human Pose Estimation via Deep Neural Networks这篇文章，其中首次提出了使用深度神经网络去判断人体姿态的方案，不过在这里的方案是基于回归实现的，作者提出的方法是对关键点进行回归，输入信号是整张图片，这样就可以解决上下文问题，同时不需要显式地注意模型拓扑和关节之间的交互

## 主流方法

目前，二维单人姿态估计有两种主流方法，一种是基于直接回归坐标的方式（DeepPose），包括CNN多阶段回归模型和CNN多阶段反馈回归模型，一种是基于热力图回归坐标的（CPM或者说姿态机和Hourlgass等），比如说CNN+图模型（使用图模型或者树模型进行精细化等）、CNN多阶段回归和检测加回归（这是一种混合模型）等，还有基于GNN图模型进行估计

多人姿态估计的方法有自上而下（先具体到每个人，然后对单人进行姿态估计，即Top-down，先找人后找点）和自底向上（先识别关节点，然后将其组合成人体，Bottom-up）两种方式或者说两种分支

![JulyEdu_KeyPointDetection_PangYan_L2_15](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_15.png)

### 多人姿态估计Top-Down方式

Top-down是先检测出每个人，然后对每个人进行关键点检测（这里的点是有语义的），但是对于人数较多的情况下，计算量线性增长，实时性不足

![JulyEdu_KeyPointDetection_PangYan_L2_27](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_27.png)

在这种方式上，有一个里程碑式的工作，即Mask R-CNN，当然并不是标准的Mask R-CNN，而是增加了其他功能的Mask R-CNN

Mask R-CNN实现了联合学习，通过添加不同的head或者说分支实现不同的功能，并且有不同的损失函数进行叠加

![JulyEdu_KeyPointDetection_PangYan_L2_30](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_30.png)

但是随着卷积层数的增加，高级语义的信息越来越丰富，但是分辨率逐渐降低，导致难以识别小目标物体，一个解决方法就是**FPN（Feature Pyramid Network，也叫特征金字塔网络）**，这种带带在每一层上进行一个预测，然后将损失叠加起来，这样就可以完成对小目标的识别

FPN主要接近物体检测中的多尺度物体，通过简单的网络连接改变，在基本不增加原有模型计算量的情况下，大幅度提高检测小目标的性能

![JulyEdu_KeyPointDetection_PangYan_L2_33](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_33.png)

Bottom-up对所有的关节点进行检测，然后将关键点组合起来，组合为人体骨架，这里模型会学习到人体骨架的关系，判断哪些关键点属于同一个人，适用于人数非常多的情况，但是精度稍低

## 评价标准

人体姿态估计就是检测出关键点并且进行连接，所以评价标准与一般的目标检测不同，如下图所示，蓝色为真实标签，绿色为预测标签

![JulyEdu_KeyPointDetection_PangYan_L2_17](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/JulyEdu_KeyPointDetection_PangYan_L2_17.png)

我们评价关键点检测的标准是这样的：因为没办法算重合面积，所以主要是计算预测关键点和真实关键点之间的距离或者误差

这里列举了一些常见的评估标准：

1. 均方根误差（RMSE）：均方根误差是一种常见的评估指标，用于测量模型预测的关键点和实际关键点之间的距离。计算方法是取所有关键点的预测误差的平方的平均值，然后再取平方根。RMSE值越小，说明模型的预测性能越好。
2. 平均准确率（PCK, Percentage of Correct Keypoints）：这是人体姿态估计中常用的一个评价指标。对于每一个关键点，如果预测位置与真实位置的欧氏距离小于某个阈值（通常是人体某部位长度的一定比例，如头部长度的50%），那么就认为这个关键点预测是正确的。然后计算所有关键点预测正确的比例，即为PCK。
3. 对象关键点相似性（OKS, Object Keypoint Similarity）：在COCO数据集中，通常使用的是OKS-based mAP（mean Average Precision）。OKS考虑了人体关键点预测的位置误差，也考虑了人的尺度因素。mAP评价指标在此基础上，进一步考虑了预测的准确性和召回率，并综合了所有阈值下的结果。
4. 平均端到端误差（Mean Per Joint Position Error, MPJPE）：在3D人体姿态估计任务中，MPJPE是一个常用的评价指标。它计算的是预测的3D关键点位置与真实3D关键点位置之间的欧氏距离的平均值。通常在计算MPJPE之前，会先进行一次无尺度的刚体对齐（去除旋转和平移）。

## DeepPose

DeepPose是最早的使用神经网络来进行人体姿态估计的方法

这种方法，将k个人体关节视为k个连续标签，然后使用神经网络和深度学习方法进行回归，同时，因为坐标都属于图像坐标系并且有范围，故可以进行归一化处理，将一张图片归一化为224大小，同时坐标也进行归一化处理

DeepPose的主要步骤包括：

1. 提出一个初始的关节点估计：首先，DeepPose会使用一个卷积神经网络（CNN）对每个关节点进行一个粗略的估计。这个网络的输入是整个图像，输出是所有关节点的坐标，这样可以捕获关节点的所有上下文信息。
2. 迭代优化关节点估计：然后，DeepPose会使用一个序列的卷积神经网络来迭代优化关节点的估计。每个网络的输入是上一步估计的关节点周围的图像块，输出是对关节点坐标的更新（或者说学习到的偏移量）。通过这种方式，DeepPose能够逐步提高关节点估计的精度。

这种方法简单易用，没有太多复杂的地方，实际上CNN分类网络的一种迁移应用，将其应用在人体姿态估计问题上，使用回归方式和L2距离完成关节点的估计，但是不同关节点之间没有直接的交互

当然，这种方法还有其他问题，因为其基于CNN架构，所以对输入图片的形状也有要求，即固定大小224x224（原文中为220），图片相对粗糙，这样也存在了无法精确定位人体关节点的问题

当然，DeepPose为了更好的去估计位姿，使用了级联的位姿回归方式，其中分为两个阶段

1. 第一阶段为初始阶段，使用一个CNN进行回归
2. 第二阶段，训练额外的神经网络回归器，输入是上一阶段所预测关节点周围的图像区域，这样分辨率更高，回归器可以看到更清晰的图像细节，对上一阶段预测的位姿来进行进一步的细化

这种类别的模型也称为迭代误差反馈模型，是一个多阶段的反馈模型

## 迭代误差反馈模型

这种算法的动机是希望网络学习到一个多阶段反馈的模型，论文题目为[Human pose estimation with iterative error feedback](http://openaccess.thecvf.com/content_cvpr_2016/html/Carreira_Human_Pose_Estimation_CVPR_2016_paper.html)

这种方法的思路就是，每轮产生一个关键点预测，然后下一轮将关键点预测图和原始图片一起输入进行纠正

## Convolutional Pose Machines（卷积姿态机）

卷积姿态机是CMU的一个里程碑式的工作，使用多阶段有先验的方式去实现一个单人位姿估计的任务，并且使用了残差连接的原理，使得网络更容易训练

卷积姿态机（CPMs）继承了姿态机的特点

卷积姿态机是由一系列的多分类器组成，每个分类器可记为$g(\cdot)$，每个分类器用来预测不同层次中不同关键点的位置，每个分类器都被记为一个**阶段（stage）**，从第二个阶段开始，每个阶段接受前一阶段提取的信息作为先验信息

## 基于先验的模型

这种模型的动机就是给网络添加一些先验知识，也就是双源CNN的思想

这种网络的工作方式，与R-CNN系列有相通之处，也是基于区域提议，然后将区域（一个图像块）提议当做先验输入网络，然后区域的地方产生一个Mask，一起输入到网络中，这样就可以产生一个分类之后的区域

这样就可以促使神经网络学到更多的东西

## 基于热力图的方法

这种方法的动机是这样，人体的尺度不一样，那就想办法让网络克服这个问题，学习到关节之间的关系，并且使用了CNN+图模型方法

论文：Joint Training of a Convolutional Network and a Graphical Model for Human Pose Estimation

## 基于树状结构

动机：之前的建模是基于关节之间的关系，那么是否可以对所有关节形成的树状结构进行建模，也就是CNN+树状结构图模型

论文：http://openaccess.thecvf.com/content_cvpr_2016/html/Yang_End-To-End_Learning_of_CVPR_2016_paper.html

但是是一个全连接图，计算量较大，效率较低

## 多阶段回归

动机：图模型的计算效率低，尝试抛弃图模型，使用多阶段回归来保证精确度

方式：使用卷积姿态机，并且使用大卷积核提高感受野，实现多阶段回归（CMU的论文）

## 关键点检测方法——热力图回归

这是检测人体关键点的一个主要方法

