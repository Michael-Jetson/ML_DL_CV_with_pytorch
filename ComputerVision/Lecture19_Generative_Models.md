# 概述

在课程中，大多数时间都是在讨论监督学习模型，这种模型实际上就是在学习一个映射函数，如何把输入的图像映射到一个标签，要注意的是，作为输入数据的图像需要人为进行准确的标注，否则就无法准确进行

![EECS498_L19_8](./assets/EECS498_L19_8.png)

实际上无监督学习是一个有些模糊的概念，一般思想就是获取我们不知道的大量数据，以某种方式构建一个模型，揭示数据中某种隐式结构

一个例子就是聚类，聚类任务中输入一大堆样本，目标就是将样本分成不同的簇，只不过这是没有标签的样本

![EECS498_L19_9](./assets/EECS498_L19_9.png)

另一个例子就是降维，比如说主成分分析方法，这些方法可以把高维数据投影到较低维空间，而这种降维任务的目标是这样的，我们有的大量数据是在一个高维空间中的，而我们希望在高维空间中找到一些非常低维的子空间，它可以捕获原始输入数据集的大部分结构，然后我们就可以基于这种低维数据子流形，不需要依靠标签就可以进行预测

这也是自动编码器的思想，这种特殊的神经网络会学习输入的潜在结构并且重建输入，这个过程是无需标签的

![EECS498_L19_15](./assets/EECS498_L19_15.png)

然后我们可以对比一下判别模型和生成模型，这两种模型实际上的区别就是试图学习的潜在概率结构的类型不同，判别模型实际上想学习一个概率分布，这个概率分布可以在输入图像的时候，以此为条件预测标签的概率，而生成模型则是尝试学习图像上的概率分布，还有一种结合体——条件生成模型，其会尝试学习以标签为条件的图像上的概率分布

然后概率分布函数是一个关键点是其是归一化的，这意味着概率分布中不同的元素需要相互竞争，也就是如果想为一个元素分配更高的概率，那么其他元素的概率就会更低

首先如下图所示，假设一个分类模型在猫狗图像数据集上进行了训练，那么当为其输入一个猫或者狗的图像的时候，其输出的概率必然是高低不同的分布，这也是竞争的体现，不过竞争是不同标签之间的竞争，而不是不同图像之间的竞争

![EECS498_L19_17](./assets/EECS498_L19_17.png)

但是如果输入一张其他的图像，比如说一个猴子或者一个胡乱绘制的图像，那么分类器是无法分辨这个不合理的图像的，模型还是会按照归一化的方式输出概率分布，也就是说，当对监督学习模型进行**对抗性攻击（adversarial attack）**的时候，或者说合成了一个不合理的图像让模型进行判别的时候，其仍然被迫在标签空间上输出归一化的分布，也就是判别式模型存在着一种缺点，当输入是不合理的时候，是无法告诉我们输入是不合理的

![EECS498_L19_19](./assets/EECS498_L19_19.png)

而相比之下，生成模型是学习可能存在于所有可能图像的概率分布，也就是说，对于输入到系统中的每个可能图像，模型都需要分配一个数字来表示该图像存在的可能性有多大，当然要实现这个功能是很困难的，需要很强大的视觉理解，如下图所示，模型认为狗图像存在的可能性最大，比猫图像更大，因为可以认为在户外看到狗或者说成年狗的可能性较大，但是在户外草地看到小猫的可能性不是那么大

![EECS498_L19_20](./assets/EECS498_L19_20.png)

这样模型就可以通过给图像设置一个较低的可能性的方式，拒绝不合理的输入

当然还有一种条件生成模型，就是在某种条件下，去判断不同图像的可能性是多少，如下图所示，这样可以构建一个条件分类器，如果在某种条件下图像的概率低于阈值，那么就认为这是一个不合理的输入

![EECS498_L19_23](./assets/EECS498_L19_23.png)

不过这些模型实际上并不是完全不同，可以参考贝叶斯公式进行条件概率的翻转，也就是我们可以根据现有条件构建一个条件生成模型，通过生成模型、判别模型和预定义标签来得到条件概率模型的输出，这是不同类型的概率模型之间的一种很好的联系，并且可以结合判别式模型和生成式模型两种方法来构建条件概率模型

![EECS498_L19_25](./assets/EECS498_L19_25.png)

那么具体应该怎么去构建呢？比如说判别式模型可以为新数据分配标签，这样可以用于监督特征学习，用法可以是使用一个在ImageNet上训练好的网络，然后除去分类层，然后使用卷积层来作为特征提取器，从图像中提取有用的有意义的语义特征；对生成模型，可以检测异常值，告诉我们，基于训练的数据，这种输入图像不太可能出现

![EECS498_L19_28](./assets/EECS498_L19_28.png)

而想使用生成式模型合成新数据，实际上就是在生成式模型中进行采样来合成匹配的新数据，因为生成式模型实际上是学习图像的分布

![Stanford_CS236_L2_3](./assets/Stanford_CS236_L2_3.png)

## 生成式模型分类

从大类别上来说，整个生成式模型有显式密度函数模型和隐式密度函数模型两种，因为生成式模型实际上就是把密度函数分配给图像

- 对于隐式密度生成模型，实际上是无法计算概率密度的，只能在模型中进行采样计算（或者说从潜在密度中进行采样）
- 对于显式密度模型，可以直接计算输出概率密度

![EECS498_L19_34](./assets/EECS498_L19_34.png)

对于显式密度模型，也有两个小类别，一种是易于处理的密度模型，输入测试图像之后可以接受图像上密度函数的近似值，一种是近似密度模型，这些模型无法直接输出实际值，但是可以计算密度函数的某种近似值，计算方法有变分方法或者蒙特卡洛方法

对于隐式模型，则可以考虑直接采样和蒙特卡洛方法等进行采样得到输出

这里主要是讨论自动回归模型、变分自动编码器和GAN三种

## 生成式模型应用

当然，实际上生成式模型不是一种模型，是一类模型，实际上有很多应用可以使用生成式模型来完成，除去基础的图像生成和语言生成，还有文生图、视频生成、医学图像重建、图像超分辨率、AI4S（如生成分子结构）、机器人决策生成

# Autoregressive Model：自动回归模型

## 基本概念

自动回归模型，是生成式模型的一种，本质上是建立在使用链式法则的思想上的，使用模型基于模型的过去数据点来预测新数据点，有易于处理的显式密度，如下图所示，其输入是一组数据 $x$ 和一个个可学习的权重矩阵 $W$，输出是该图像的概率密度值，然后我们要做的事情就是获取一些数据样本来训练这个模型，并且尝试最大化密度函数的值，也就是最大化观察到的数据的可能性，并且由于归一化约束，这种设计迫使模型降低不在训练数据集中的事物的权重

![EECS498_L19_39](./assets/EECS498_L19_39.png)

某种意义上讲，自回归模型本质上做的事情就是，对于一些可以预测输出的神经网络模块，将其组合起来，每个模块视为一个单独的组件，而链式法则提供了一种在给定先前组件的情况下预测各个组件的方法，也就是说，把概率上的联合概率，拆解成条件概率累乘的形式，每个条件概率都是使用一个神经网络模块完成的，比如说对于图像，最简单的一种思路就是，给定前一个像素的情况下，预测下一个像素，也就是逐像素生成

![Stanford_CS236_L3_3](./assets/Stanford_CS236_L3_3.png)

## AR模型示例

当然，神经网络会引入非线性因素，所以实际上的计算会复杂很多，一种基于这种方法的网络就是Fully Visible Sigmoid Belief Net（FVSBN），这种网络中所有的变量都是显式可见的，并且使用Sigmoid函数来表示条件概率分布（也就是逻辑斯蒂回归分类器）

![Stanford_CS236_L3_7](./assets/Stanford_CS236_L3_7.png)

那么这种模型如何从输入中进行采样呢？实际上就是从第一个像素开始采样或者说给定一个值，然后在给定条件分布的情况下预测第二个像素，然后进行迭代，直到计算完成整个图像的所有像素

但是这种网络的问题是参数量过大（平方增长），并且方法非常简单，无法实现很强的效果

所以还有一种Neural Autoregressive Density Estimation（NADE）模型，这种方法的思路是引入线性层进行处理，然后使用激活函数，也就是使用单层的网络来代替逻辑回归分类器，而且使用了共享权重矩阵，降低了参数量，并且降低了过拟合的可能性

![Stanford_CS236_L3_10](./assets/Stanford_CS236_L3_10.png)

这种网络相对上一个网络有了较大的性能提高，对于黑白图像的生成效果有较大加强，而且参数量也变线性增长

当然，上面都是基于黑白图像进行二元预测的，对于彩色图或者灰度图来说，每个像素有数百离散值，所以需要进行一些修改，比如说使用softmax进行分类

![Stanford_CS236_L3_12](./assets/Stanford_CS236_L3_12.png)

如果想对连续变量进行建模，就可以使用连续变量分布方法来进行预测，比如说高斯分布，把高斯的参数进行建模

![Stanford_CS236_L3_13](./assets/Stanford_CS236_L3_13.png)

表面上，这两种模型看起来很像自动编码器，在自动编码器中，编码器部分获取数据点并且将其映射到某种隐式的数据表示（数据压缩过程），然后解码器尝试获取编码器的输出并且将其映射为某种输出（数据重构）

![Stanford_CS236_L3_15](./assets/Stanford_CS236_L3_15.png)

然后就是Masked Autoencoder for Distribution Estimation，也就是MADE模型，这种模型可以对一些信息进行屏蔽，比如说每个输出只能看到之前的输入，不能依赖之后的输出，但是基于这种方法实现的网络还需要考虑如何进行排序操作，当然，实际上也可以通过RNN这种模型来进行顺序学习并且只能依赖过去的输入

![Stanford_CS236_L3_17](./assets/Stanford_CS236_L3_17.png)

当然，RNN也有一系列的问题，比如说无法并行化，只能逐个计算，并且有梯度消失和爆炸的问题

![Stanford_CS236_L3_25](./assets/Stanford_CS236_L3_25.png)

当然，后面的Attention-based模型做了一些改进，结合RNN模型，可以做到查看所有时间步上的特征，然后基于整个序列的信息进行最终输出

![Stanford_CS236_L3_26](./assets/Stanford_CS236_L3_26.png)

基于这种思想的经典作品就是PixelRNN，这是16年的工作，可以逐像素生成彩色图像

![Stanford_CS236_L3_28](./assets/Stanford_CS236_L3_28.png)

其生成效果如下图所示，这是在ImageNet上训练的结果，输入的图像会进行Mask，可以看到，可以实现不错的生成效果

![Stanford_CS236_L3_29](./assets/Stanford_CS236_L3_29.png)

还有一种PixelCNN，这种方法基于卷积操作来提取待预测像素周围像素的特征，然后逐步对不同像素进行预测

![Stanford_CS236_L3_31](./assets/Stanford_CS236_L3_31.png)

这种方法的性能与PixelRNN接近，但是会快很多

然后就是深度学习中的对抗攻击问题了，对于一个图像加入一些噪声，哪怕整体上这个图像看起来没有大改变，但是模型仍然会以一种很高的置信度将其分类为完全错误的类别，也就是模型会对这些可能是精心设计的噪声非常敏感

![Stanford_CS236_L3_33](./assets/Stanford_CS236_L3_33.png)

## 最大似然估计学习

实际上自动回归模型就是一系列的基本分类器，然后只需要以正常方式训练所有分类器就可以了，但是如何对这些分类器进行训练呢？

这里需要找到一个相似性或者距离的概念，来判断或者评估联合概率分布是否彼此接近，然后通过对比这种距离去对模型进行优化

![Stanford_CS236_L4_6](./assets/Stanford_CS236_L4_6.png)

那么如何去评估这种相似性或者距离的概念呢，这里就需要KL散度的定义了，这种指标可以对比两个分布之间的差异，衡量两个分布之间的相似性

![Stanford_CS236_L4_7](./assets/Stanford_CS236_L4_7.png)

当KL散度为零的时候，就意味着两个概率分布完全相等，也就是说，我们只要让模型输出的概率与标签的KL散度为零或者说尽可能小，那么就可以让模型进行学习，这实际上就是进行最大似然估计的过程

![Stanford_CS236_L4_10](./assets/Stanford_CS236_L4_10.png)

当然，这个过程也是通过随机梯度下降来完成的

![Stanford_CS236_L4_18](./assets/Stanford_CS236_L4_18.png)

## PixelRNN 与 PixelCNN

PixelRNN 和 PixelCNN 都是序列化生成图像的工作，核心思想在前一个点或者前一块区域的基础上生成下一个像素点，这里使用了神经网络来代替条件概率完成建模，具体的，PixelRNN 是通过 RNN 来实现序列化生成，PixelCNN 是通过 CNN 卷积预测下一个像素，这种过程可以使用前文提到的链式法则来描述，似然函数可以精确计算，并且可以有效评估性能，不过这种工作的缺点就是，必须序列化的来逐个生成像素，学习和生成的会非常慢

# VAE：变分自编码器

这是一种隐式密度估计的方法，

## Base

自编码器是无监督中的一种方法，核心就是通过无标签的数据找到一个有效的低维特征提取器，或者说把一个高维的向量压缩成一个低维特征向量，具体的压缩方法有全连接层、卷积层等等

然后可以通过解码器，对特征进行解码，重构出来输入的数据

训练的过程实际上是监督学习损失函数完成的，把重构出来的数据和原始输入对比，然后使用 L2 或者说均方差损失进行监督学习，这个过程中实际上不需要任何的标签，实际上就是无监督学习

应用也有很多，训练完成之后可以移除解码器，单独使用编码器，因为训练好的编码器的输出肯定是有意义的，所以编码器可以作为有监督学习的特征提取网络

# GAN：对抗生成网络

## 概述

GAN 是生成式网络中的代表作，其核心思想是使用两个网络进行对抗式学习如何生成高质量的图像，两个网络就好比造假钞和审查员，一个负责制造出足以以假乱真的假钞让审查员辨别不清真假，一个负责评判钞票是否是真钞

G 模型去拟合数据的分布，这是很重要的一步，因为生成式模型就需要尽可能模拟原始数据的分布；D 模型是一个判别模型，用来估计样本是从真正的数据里面筛选的还是生成模型 G 生成的。其中 G 模型的作用就是尽可能让 D 模型犯错，思想是当 G 模型已经抓住了数据的分布生成了一个真假难辨的样本之后，就说明训练效果已经比较好了，可以理解为 G 是造假人员，D 是审查员，当 G 造出一个物品的时候 D 会判断真假，当 G 生成了一个真假难辨的图片的时候（也就是 D 模型给出了真假各一半的概率的时候），模型就已经达到很好的效果了

实际上 GAN 更像是一种框架而不是一个具体的网络，其中的生成模型可以是 MLP 或者 CNN 或者 RNN，判别器也可以是各种具体网络，只要在做的是生成式任务就可以运用对抗学习的思想进行优化

当然一个缺点就是，GAN 本身是一个无控制的生成式模型，没办法根据人的想法来有条件的生成特定的图像，所以后面有了大量的进一步改进的工作

## 相关工作

在 GAN 之前的生成式相关的工作，主要显式密度估计方法，这种方法是想构造一个分布函数出来，然后对其中的一些参数进行学习，然后这些参数通过最大化对数似然函数来进行优化，坏处就是高维时计算困难，这种方法的核心就是学习出来原始的分布并且明确其中的各种参数

另外的一些相关工作是隐式的密度估计方法，使用一个模型来近似一个分布，这样的话学习起来容易很多，但是坏处就是不清楚最终的分布具体是什么样子，此外这里作者发现，对函数期望的求导等价于对函数自身的求导，这也是 GAN 可以进行反向传播的基础

实际上在这之前，已经有了相关的使用辨别模型来辅助生成模型的工作了，比如说 NCE

## 对抗网络

原始论文中的网络结构是 DNN，为了让生成模型更好的学习数据分布，人为的给出了一个先验噪声分布 $p_z(z)$，在代码中可以直接给出一个随机数的序列，我们希望给出了先验，生成器就可以输出一个 $p_g$ 来刻画数据分布或者说学习到数据的分布

然后就是同样是 DNN 结构的判别网络，负责判断生成模型输出的结果是否足够真实

我们希望的是把判别器训练到，可以准确判别输入是生成的还是真实的，也就是最大化概率 $\log (1-D(G(z)))$，同时希望把生成器训练到可以生成真假难辨的样本，也就是最大化概率 $\log D(x)$，最终达到一种均衡状态，二者都不能进一步优化

![GAN_2](./assets/GAN_2.png)

实际上 GAN 的流程如上图所示，x 是数据分布中采样得到的标量，z 是从均匀分布中采样得到的标量，而绿线表示的是 z 的分布，黑点表示 x 的分布（这里是高斯分布），蓝点表示判别器的概率输出

1. 图一，生成器把 z 映射成一个高斯分布，判别器训练开始
2. 图二，映射的分布开始拟合数据分布，判别器可以分辨真假
3. 图三，继续更新生成器，继续逼近原始的数据分布
4. 最终，生成器完全逼近真实分布，判别器无法分辨，输出的概率恒定为一半

当然，在损失函数处实际上有一个问题，就是在早期的时候生成器相对弱，很容易训练出来一个强大的判别模型导致生成器的梯度变成 0，所以

原作者使用了 MNIST 数据集进行了生成式的训练实验

## 代码

### 算法流程

算法上并不是很难，在每一轮训练中，会先采样一批的噪声分布样本（实际上就是一个随机数列）和一批真实的数据样本，然后送入生成网络和判别网络中并且进行训练

![GAN_1](./assets/GAN_1.png)

### 代码

这里基于 MNIST 进行实验，所以结构和代码会很简单

生成器代码

```python
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
		#使用MLP进行网络搭建，最后使用Sigmoid进行归一化处理，因为黑白图像的像素数值归一化之后就是0-1
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),

            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),
            #  nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # shape of z: [batchsize, latent_dim]
		# 实际上输出的是一个长向量，向量长度是图片像素数，所以要reshape成二维图像
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)

        return image
```

判别器代码

```python
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
		# 实际上也是一个MLP，输出是一个基于Sigmoid的概率
        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size, dtype=np.int32), 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        # shape of image: [batchsize, 1, 28, 28]
 		# 把图像变成向量然后判别
        prob = self.model(image.reshape(image.shape[0], -1))

        return prob
```

损失函数的话使用 BCE 损失也就是Binary Cross-Entropy Loss，二进制交叉熵损失

```python
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        gt_images, _ = mini_batch

        z = torch.randn(batch_size, latent_dim)

        pred_images = generator(z)
        g_optimizer.zero_grad()

        recons_loss = torch.abs(pred_images-gt_images).mean()

        g_loss = recons_loss*0.05 + loss_fn(discriminator(pred_images), torch.ones(batch_size, 1))#希望生成器可以学习如何生成以假乱真的图片

        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()

        real_loss = loss_fn(discriminator(gt_images), torch.ones(batch_size, 1))
        fake_loss = loss_fn(discriminator(pred_images.detach()), torch.zeros(batch_size, 1))
        d_loss = (real_loss + fake_loss)

        # 观察real_loss与fake_loss，同时下降同时达到最小值，并且差不多大，说明D已经稳定了

        d_loss.backward()
        d_optimizer.step()
```

不论是使用什么样的网络来构建生成器和判别器，都基于这个流程实现即可

实际上这是一个无监督学习的过程，因为并没有使用了标号，然后使用了有监督学习的损失函数来进行无监督学习，因为实际上是把图片真伪作为一个标号输入了进去，所以在训练上就高效了很多，这也是后面很多自监督学习方法比如说 BERT 的灵感来源



这是 GAN 领域中很重要的两个基础模型，[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)  和 [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)

## cGAN

GAN 的生成器有一个问题，就是生成过程只是从噪声中进行采样，无法指定生成的图片样式，所以 cGAN 的一个思想就是给一个额外先验的类别信息，让生成器知道应该生成什么类别的图片，甚至可以让生成器生成的图像更加真实

添加先验的方式也很简单，把类别先验做成 one-hot 向量然后拼接到生成器和判别器的输入上，做一个 Embeddding 即可，然后判别器还需要判断生成的图片是否属于某一个类别，而非仅仅判断图片的真伪

![cGAN_1](./assets/cGAN_1.png)

## LSGAN

GAN 使用的是使用 Sigmoid 进行生成，同时使用交叉熵损失函数来进行一个监督训练，但是这种函数容易出现梯度消失的问题，所以 LSGAN 使用了最小二乘损失函数来进行监督训练，或者说使用了均方差损失，使得任务变成了回归任务 

代码操作上与 GAN 一致，只是损失函数使用了 MSE 损失

## StackGAN

[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_StackGAN_Text_to_ICCV_2017_paper.html) 也是一个非常经典的论文，其核心思想是通过多阶段网络的堆叠，生成高质量的图像

实际上就是先生成低分辨率的图像，然后每个阶段都在前一个阶段的基础上生成更高分辨率的图像

## Text2Image

通过输入文本，从而产生对应的图片，开山之作是 [Generative Adversarial Text to Image Synthesis](https://proceedings.mlr.press/v48/reed16.pdf)

想训练类似的模型，就需要大量的图像搭配对应的文本描述，这篇工作的核心就是使用 RNN 编码文本为特征向量，然后结合 cGAN 架构，以此生成匹配图像

# Image2Image

[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) 是图生图工作的代表作，图生图主要是应用于一些使用简单图像生成复杂图像的任务，比如说从草图到完成图等等

# Diffusion Model

奠基论文有 [DDPM](https://arxiv.org/abs/2006.11239) 和 [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)

扩散模型本身是生成式模型，所有的图像生成式任务实际上都可以通过扩散模型完成

## Base

参数重整化：如果我们希望可以通过从