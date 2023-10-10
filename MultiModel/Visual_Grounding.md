# Visual Grounding

这是一个多模态的任务，主要目的是将自然语言描述与图像中的特定区域或者说对象相对应。简单来说，它是关于如何将语言描述与图像中的具体对象或区域“锚定”或“定位”的（定位形式等同目标检测）。此方向还有另一个名称，Referring expression comprehension (REC)，这里referring expression与Image Caption不同，此概念的有一个性能度量标准：如果认为一个referring expression是好的，那么它在上下文中唯一描述了相关对象或区域

例如，给定一张包含多个对象的图像和一个描述，如“红色的苹果”，Visual Grounding 的任务是在图像中找到与描述相匹配的对象或区域。

与此相对的是referring expression generation（REG）任务，这是REC的逆任务，对图像中的特定区域或者对象进行语言描述

## 分类

按照方法分类，目前有这些

![Referring_Expression_Comprehension_A_Survey_of_Methods_and_Datasets_2](.\assets\Referring_Expression_Comprehension_A_Survey_of_Methods_and_Datasets_2.png)

- joint embedding
- modular-based
- graph-based
- using external parsers
- weakly supervised
- one-stage
- vision-language pre-training

## Joint Embedding

这是一种视觉和语言联合嵌入的方法，核心思想是将视觉和语言信息编码到同一个向量空间，然后学习如何进行对齐，其中CNN可对图像进行encode，LSTM可对文本进行encode

具体的操作流程，是使用CNN-LSTM架构，CNN可以是预训练的模型，在生成式任务中，CNN提取特征并且传入LSTM中进行文本生成，在解析任务中，LSTM接受region-level的CNN特征和词向量作为输入，然后在每一个时间步最大化给定参考区域的表达式的可能性（如下图所示[^1]），最后LSTM输出一个向量，代表了文本描述在图像上的位置或区域。这种框架有效地结合了图像和文本的信息，以实现精确的对象定位

![Referring_Expression_Comprehension_A_Survey_of_Methods_and_Datasets_3](.\assets\Referring_Expression_Comprehension_A_Survey_of_Methods_and_Datasets_3.png)

### Generation and Comprehension of Unambiguous Object Descriptions

这篇工作，应该是最早在REG（referring expression generation and comprehension）领域提出CNN-LSTM架构的工作，或者说这是第一个基于深度学习的REC工作，在此文章中，CNN是在ImageNet上预训练的，并且解决了REC的逆问题，同时基于COCO数据集提出了用于REC和REG训练的数据集Google Refexp (G-Ref)，有超过三十万张图像，在此之前相关的数据集规模都较小

此工作的架构很接近Image Caption的模型，使用CNN表示图像，然后跟着LSTM生成文本，不过区别就是除去位置信息，还使用感兴趣区域的CNN特征来增强CNN表示（下图所示）

![Generation_and_Comprehension_of_Unambiguous_Object_Descriptions_5](.\assets\Generation_and_Comprehension_of_Unambiguous_Object_Descriptions_5.png)

### Modeling context in referring expressions

与上一个文章类似，但是引入了视觉对比法（visual comparative method，Visdif），将目标物体与周围物体分开（尤其是相似物体），便于聚焦于目标物体和表达式之间的关系上，因为同一个类型的周围物体不一定会提供有用的信息，所以此工作提出了一种更明确的编码，用来描述同一个类别的物体之间的视觉差异，这种编码方式会计算两类视觉特征进行视觉比较，然后生成指代表达

在此工作中，第一种是目标物体与图像中其他同类物体在视觉外观上的异同，对CNN特征进行池化，作者尝试了最大/最小/平均三种方式的池化，并且认为平均池化的性能最好，并且计算视觉差异的公式为
$$
\delta _{v_i}=\frac{1}{n}\sum_{j\neq i}\frac{o_i-o_j}{\Vert o_i-o_j\Vert}
$$
$n$是选中用于比较的物体数量，$o_i$是目标物体表征，是CNN网络的线性层输出，在原文中使用的是VGG的FC7层输出，此外还有$g_i$是全局上下文表征，是全部图像输入VGG从而在FC7层得到是表征

第二种计算方式为计算相对位置与大小，计算对象也是目标物体和其他同类物体，这是因为人们经常在指代表达中使用关于位置和大小的信息，并且考虑目标物体周围同类物体的数量不定，使用只选择五个最近的同类物体，如果数量不够则使用0填充

位置/尺度表征记为$l_i$，为五维向量，包括左上角、右下角坐标和面积大小（这些都是相对图像而言的），分别为$\frac{x_{tl}}{W},\frac{y_{tl}}{H},\frac{x_{br}}{W},\frac{y_{br}}{H},\frac{w\cdot h}{W\cdot H}$

位置差异表示记为$\delta l_i$，计算公式为
$$
\delta l_{ij}=[\frac{[\Delta x_{tl}]_{ij}}{}]
$$


### Modeling context between objects for referring expression understanding

考虑到训练过程中的监督只包括所提及对象的注释，这是一种有限的监督，很难对对象之间的关系进行建模，所以拓展了Visdif机制，提出了multiple-instance learning（MIL）方法，在此工作中，输入特征包括region和context region，尝试将referring expression映射到区域及其支持上下文区域（region and its supporting context region）中，比如说下图中，表达式“The plant on the right side of the TV”中，图片中的植物是referred object，TV是context object 

![Modeling_Context_Between_Objects_for_Referring_Expression_Understanding_1](.\assets\Modeling_Context_Between_Objects_for_Referring_Expression_Understanding_1.png)

但是上下文对象的边界框标注无法用于训练，所以这个工作尝试使用弱监督进行关系的学习；其中两个训练目标函数的公式类似于mi-SVM（Support Vector Machines for Multi ple-Instance Learning ）

下面是模型的示意图，将指令转化为词嵌入向量，然后区域由CNN提取特征，一同输入LSTM中，并且使用最大池化的方式选出目标

![Modeling_Context_Between_Objects_for_Referring_Expression_Understanding_2](.\assets\Modeling_Context_Between_Objects_for_Referring_Expression_Understanding_2.png)

### Natural language object retrieval

其中提出了Spatial Context Recurrent ConvNet (SCRC)模型作为目标检测的评分函数，输入为{region, context region}，并且图像本身也被看做一个上下文区域

模型包括三个LSTM，分别记为$LSTM_{language}$、$LSTM_{local}$、$LSTM_{global}$以及两个CNN（分别负责local和global）以及词嵌入层和词预测层

## Dataset

目前，REC领域的数据集有很多，最早的一个就是ReferItGame，这个数据集有十几万个描述、九万多个物体和近两万个自然场景的图片，基于ImageCLEF IAPR数据集扩展而来

然后就是Google Refexp (G-Ref)数据集，由COCO数据集扩展而来，包括超过三十万数量的八十个物体类别的图片，并且带有实例级的分割信息，平均每张图片有多个expression

## Scanrefer: 3d object localization in rgb-d scans using natural language.

这个应该是第一篇3D Visual Grounding方面的工作，工作基于点云输入，并且是第一个3D visual grounding方面的带有3D语义和自由语言描述的数据集，包括一万多物体的五万多人工描述，之前并没有真正意义上的3D数据集

模型在架构上包括两个主要模块：检测/编码模块和融合定位模块，其中视觉部分是基于候选区域法，使用PointNet++提取点云特征并且用于检测和融合，语言部分使用GloVE+GRU方式获得特征向量

![ScanRefer_6](.\assets\ScanRefer_6.png)

## BUTD-DETR

这是一个使用Transformer手段并且基于图像与点云的工作，一个改进是解决了基于预训练目标检测器方法中对小目标和被遮挡目标无法识别的问题，此工作基于MDETR，同样使用预训练目标检测器来得到box proposal

# WildRefer

## 介绍

在这里，我们提出了一个新的任务：3D Visual Grounding in Wild，简称3DVGW，目标是通过实时的视觉数据和面向对象的语言描述来定位现实世界中的目标对象，这个任务在自动驾驶、辅助机器人等领域都有应用

## Related Work

目前，二维的Visual Grounding已经取得了很大的进步，但是这些方法局限于图像数据，而难以应用于三维，在一些三维的Visual Grounding的数据集被提出之后，使用网络去捕捉、学习三维数据成为了可能，这对机器人操作等方面是有很大意义的，数据集包括ScanRefer（RGBD）、ScanNet

不过这些基于RGBD的数据集都有一个问题：数据集的场景都是室内场景，因为RGBD的工作范围不是很大，无法在大场景下很好的采集数据，所以后面就使用了LIDAR作为视觉传感器采集数据，补充RGBD的不足，这里的数据集有nuScenes、SemanticKITTI

## Difference

不过3DVGW不同于传统的RGBD visual grounding，3DVGW是在动态场景下工作，并且是多个模态的视觉数据融合

在此之前，Visual Grounding的工作一般以提前扫描的静态室外场景作为数据集，没有时间标签

## Mechanism

### Dynamic Visual Encoder

基于Transformer，提取动作特征

### Triple-modal Feature Interaction

在这里进行三种模态的交互，包括语言的语义信息、点云的几何信息和图像的表示信息，以此来增强特征

### Decoder

这里使用DETR形式的decoder，通过语言的上下文特征和视觉token来预测bounding box

## Dataset

提出了STRefer和LifeRefer两个Dataset，都是128线LIDAR和工业相机采集的以人为中心的大场景数据，前者是相对拥挤场景下的，后者是日常生活场景下的

注释是自由形式的语言描述和3D Bounding Box，语言描述部分由多人组织，标签经过校验

## Architecture

### Overview

Input为点云和图像以及语言描述，输出就是目标物体的Bounding Box，整体框架为一阶段框架

![WildRefer_3](.\assets\WildRefer_3.png)

在这里使用PointNet++作为点云的特征提取器，使用ResNet34作为图像的特征提取器，使用BERT来提取文本特征，三种模态的特征分别记为$F_{P_t}$，$F_{I_t}$，$F_L$

并且为了充分挖掘序列数据中的时间信息并且与动作相关的描述进行匹配，我们设计了一个动态视觉特征编码器

同时，为了让视觉的模态信息更好的与语言中的语义特征对齐，我们提出了一种三模态交互策略来进行特征增强

### Dynamic Visual Encoder（DVE）

因为3DVGW需要去识别动作，所以需要一个可以动态识别的部分，也就是这个DVE，也就是动态视觉编码器，这里会接受CNN和PointNet++提取的特征序列（因为CNN和PointNet++一次会输入一连串的序列，也就是连续的视频帧），并且进行处理

### Triple-modal Feature Interaction (TFI)

我们有了三种特征$F_{P_t}$，$F_{I_t}$，$F_L$，



[^1]:[Referring Expression Comprehension: A Survey of Methods and Datasets](https://ieeexplore.ieee.org/document/9285213)

