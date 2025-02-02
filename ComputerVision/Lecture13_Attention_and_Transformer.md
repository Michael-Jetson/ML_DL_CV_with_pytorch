# 从RNN中得到启发

我们使用递归神经网络进行序列到序列的预测，接受一些输入序列然后输出一些序列（比如说翻译，或者视频字幕），序列的长度是不定的，那么我们的模型应该怎么处理呢

## 输入的表示

这里可以使用独热向量的编码方式（如下图所示），但是这种方式认为所有的向量是相互独立且没有关系的，没有任何语义的信息，如果在NLP中使用这种方法编码语言，会导致向量过于稀疏和计算复杂等问题，所以会使用一种**词嵌入（Word Embedding）**的方法，使得输入向量具有语义信息，并且是稠密的

![lhy_attention_2](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_3.png?)

Word Embedding是自然语言处理（NLP）中的一种表示方法，它可以将词语或文本转换为具有固定长度的向量。这些向量捕获了词语之间的语义和语法关系，使机器可以更好地理解人类的语言。这种方法通常用于深度学习模型中，因为深度学习模型通常不能直接处理文本数据，而需要将文本转换为数值型数据。

Word Embedding 的基本思想是：在语义空间中，相似的词语应该有相似的向量表示。这就是著名的“分布式假设”（distributional hypothesis）：一个词的意义主要由它周围的词决定。

目前有很多生成词嵌入向量的方法，包括 Word2Vec, GloVe (Global Vectors for Word Representation), FastText 等。

在语音信号处理过程中，通常会使用向量化的方式来记录信号，一般采样率为16kHz，每25ms的片段整合为一个向量

![lhy_attention_3](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_4.png?)

当然，一些图的节点也可以作为向量进行处理

![lhy_attention_4](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_5.png?)

![lhy_attention_5](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_6.png?)

## 输出的表示

输入就是一组向量，那么输出是什么形式呢？可以是对每个输入进行预测（输入输出的数量一样），比如说对句子的每个单词进行类别判断，也可以对整个序列进行预测，比如说对句子的情感分析，还有一种就是由模型决定输出序列的长度

![lhy_attention_6](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_7.png?)

![lhy_attention_8](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_9.png?)

## 尝试使用全连接层

我们想对序列进行分类，第一个想到的就是全连接层方式，但是全连接层是不考虑上下文的，所以我们必须考虑使用一个滑动的窗口来让其考虑上下文信息，但是对于不定长度的输入序列来说，这是难以实现的，因为如果我们选择了一个很长的窗口，那么参数就会非常多，计算量很大，甚至在某些情况下，这是无法办到的，如果窗口很小，就无法有效考虑上下文

![lhy_attention_9](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_10.png?)

## 编码器-解码器模型的限制

这种模型的工作方式如下：一个称为编码器的循环神经网络接受并且处理原始输入向量，然后产生隐藏状态序列，并且在最后总结输入，产生两个输出向量（隐藏状态和上下文向量），上下文向量是要传送给解码器的每一个时间步的，这样，理论上模型就可以充分考虑上下文信息，每一个输出都是在结合所有输入的基础上得到的

![EECS498_L13_11](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_11.png??raw=true)

在解码器的开始，同时接受隐藏状态和上下文向量，然后用开始标记作为输入

在这里，这个上下文向量式一个在编码解码序列之间传递信息的向量，所以现在这个上下文向量应该以某种方式总结解码器生成其句子所需的所有信息，然后进入解码器的每个时间步

但是，这里有一个问题，如果我们希望使用序列架构的RNN来处理非常长的序列，比如说我们想翻译一本书，那么这种架构是难以完成的，因为我们难以将一个很长序列的内容打包到一个上下文向量中（或者说，一本书的内容难以打包为一页纸），我们希望有一些机制，不要强迫模型把所有的信息都集中在一个单一的向量上

那么，我们可以在编码器的每个时间步上计算一个上下文向量，然后解码器可以选择上下文向量，或者说关注上下文向量的不同部分（也可以理解为关注输入序列的不同），我们称为注意力机制

# 注意力机制下的RNN

## 注意力机制理解

我们面临传统编码器-解码器模型的限制，我们使用一种 Self-Attention 的方式或者说注意力机制来解决问题

![lhy_attention_10](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_11.png?)

我们输入一组不定长的序列（上图中无黑框的颜色方块），那么Self-Attention会直接得到一整个序列的信息，然后对每一个输入向量都计算一个带有上下文信息的输出（上图中带有黑框的颜色方块），每一个都是考虑了整个序列所得到的向量，然后我们使用一个全连接层来进行预测，这样每个决策就可以得到考虑全局信息的输出了

当然我们也可以使用多层Self-Attention，全连接层负责处理单个位置的数据，Self-Attention负责考虑全局信息，交替使用

![lhy_attention_11](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_12.png?)

当然，Attention is all you need并不是这种方式的创始论文，这种方式在之前的论文就出现了，只不过是其他的叫法或者类似的架构，只不过这篇论文将其发扬光大

## Seq2Seq模型中的注意力结构

我们有编码器，将输入序列（这个输入序列可能是整个网络的输入或者上一层输出的隐藏状态）编码为一系列隐藏状态，然后使用解码器进行预测，然后使用一些对齐函数（可以理解成一个很小的全连接网络），这个对齐函数的输入是两个向量：当前编码器的隐藏状态，解码器的隐藏状态（公式在下图右上角）
$$
e_{t,i}=f_{att}(s_{t-1},h_i)\\
f_{att}\quad is \quad an\quad MLP(多层感知机)
$$
这些对齐函数会输出一个分数，这个分数表示，考虑到解码器的当前隐藏状态，我们应该对编码器的每个隐藏状态关注多少，不过最开始，这些分数也是随机的，因为是直接从前馈函数中产生的，没有经过反向传播的学习

![EECS498_L13_15](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_15.png??raw=true)

我们会根据这个分数，以某种方式在解码器的每一步构建一个新的上下文向量，在使用比对分数的时候，我们将会使用这个对齐函数来比较$s_0$和$h_i$，比对分数代表模型认为隐藏状态对于预测输出隐藏状态$s_0$之后的单词所必须的信息

然后我们就得到了每个隐藏状态的对其分数编码序列，即
$$
\vec{e_{i}}=(e_{i,1},e_{i,2},\cdots,e_{i,n})
$$
然后我们经过一个Softmax操作，将这些分数转化为概率分布，这个分布说明，我们对于编码器的每个隐藏状态上放置多少权重（或者说关注度多少），这些分布也被称为注意力权重
$$
\vec{a_i}=softmax\left(\vec{e_i}\right)
$$
然后我们对隐藏状态进行线性加权求和，得到上下文向量

也就是说，这个网络可以自行预测我们要对输入序列的每个隐藏状态施加多少权重，并且我们可以为解码器网络的每个时间步动态改变该权重

然后，我们就可以使用我们计算出的上下文向量和输入来进行预测了（如下图所示）

![EECS498_L13_17](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_17.png?)

我们输入上下文向量和第一个输入单词，然后我们得到的输出词可能对应于输入词中的一个或者多个单词，但是我们这里的上下文向量是动态生成的，它允许解码器专注于编码器网络不同的输入

这里给出了这种直觉：如果解码器网络想生成输出序列的特定单词，那么将专注于这个单词所需的输入序列中的重要部分

然后我们在使用解码器中上一步的隐藏状态来计算这一步的上下文向量（重复之前的对齐、概率分布计算等操作），来继续生成新的文本

![EECS498_L13_20](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_20.png?)

## 注意力机制结构

当然，上述的注意力机制是在 RNN 模型中使用的，除去注意力机制的部分还有 RNN 和语言生成的部分，这里开始介绍纯粹的注意力机制，这种机制不同的地方在于，编码器每一步都会计算一个新的上下文向量，或者说每一步都会打包上下文信息

下图是李宏毅老师的图，向量 $a^i,i=1,\cdots,n$ 是网络的输入或者上一层的输出，然后每个输入向量经过一个 Self-Attention 层，得到了一个考虑了上下文信息的输出向量$b^i,i=1,\cdots,n$

![lhy_attention_12](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_13.png?)

那么具体是如何产生的带有上下文信息的输出呢？首先我们先只考虑$b^1$，我们要找到序列中跟$a^1$相关的其他向量，这也是这种机制的目的：找出与之相关的重要部分，而不是把所有的信息都打包，或者说找到对于这部分决策最有用的信息，抛弃那些无用的信息。在这里，我们使用一个数值$\alpha$来表示两个向量关联程度，那么我们怎么计算这个关联程度呢，我们需要一个计算的模组

![lhy_attention_13](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_14.png?)

我们可以使用点积操作（Dot-product）来作为判断关联程度的模块（当然还有其他模块，但是点积法最广泛，因为有计算效率高和易于优化等优点），输入两个向量然后进行相关的点积操作，就可以计算其关联程度，此外还有加性方法（Additive）

![lhy_attention_14](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_15.png?)

在这里，$q$被称为查询（Query）向量，$k$被称为键（Key）向量，它们是输入向量的映射，映射的权重矩阵$W^q$和$W^k$是可以进行学习的

在自注意力机制中，查询向量、键向量和值向量的含义如下：

1. 查询向量（Query）: 它代表当前我们关注的信息或者说当前的上下文。它将与所有的键向量进行匹配，以决定给予各个位置多大的注意力。
2. 键向量（Key）: 键向量可视为一种索引，用于与查询向量进行匹配（或者说被匹配），得到一个得分，反映当前位置与查询的相关性或匹配程度。
3. 值向量（Value）: 值向量代表输入向量在该位置的实际信息或内容。在得到每个位置的注意力权重后，这些权重将应用于对应位置的值向量，然后将所有位置的加权值向量求和，得到最终的输出。

然后我们将这种方法应用在整个序列上，就可以完成关联程度的计算了，我们计算得到了$q^i$和$q^j$的关联性程度$\alpha_{i,j}$，或者是**注意力得分（Attention Score）**，我们对某个向量和序列中所有向量进行计算（包括自己和自己计算关联性），就可以得到

![lhy_attention_15](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_16.png?)

然后对整个$\alpha$作Soft-max操作（ReLU也可以），然后得到一个归一化的关联程度，然后我们就可以根据这个去抽取重要的信息（我们根据这个知道了哪些向量跟此向量的关联性最大）

![lhy_attention_16](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_17.png?)

然后我们再使用一个值矩阵计算出值（Value）向量$v$，然后将其乘以注意力分数$\alpha$，然后进行加权组合，就可以得到结果向量$b$，然后就可以看到，哪个向量更重要（或者关联性很强），那么就会抽出更多信息，那结果向量$b$就会更接近哪个向量

![lhy_attention_17](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_18.png?)

当然，对于一个序列来说，计算是并行的，会同时计算出一系列的向量$b$

![lhy_attention_18](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_19.png?)

同时，对于一个Self-Attention层来说，查询向量、键向量和值向量的权重矩阵是共享的。所以我们也可以这样理解，将一系列向量组合为一个矩阵，比如说查询矩阵$V$

![lhy_attention_20](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_21.png?)

当然，注意力分数也可以这样进行计算，甚至我们可以进一步集成到两个矩阵中，也就是我们的注意力矩阵$A$就等于$K^TQ$，如下图所示

![lhy_attention_22](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_23.png?)

然后输出向量也可以使用矩阵乘法计算

![lhy_attention_23](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_24.png?)

我们可以进一步集成，将输入序列记为$I$，然后直接使用不同的可学习的权重矩阵（也就是$W^q$、$W^k$、$W^v$）计算输出

![lhy_attention_24](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_25.png?)

## 意义

我们进行端到端的训练的时候，我们没有告诉模型它应该关注哪些部分，但是经过训练，模型可以在生成输出序列时选择关注序列的不同部分

这样，就可以解决之前的瓶颈：无法将所有信息塞入一个上下文向量，然后解码器一直使用这一个向量。现在，注意力机制为解码器提供了生成新序列的灵活性 

在处理非常长的序列的时候，那么这将允许模型进行某种程度的转移注意力，将注意力集中在输入的不同部分，以此产生输出

![lhy_attention_37](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_38.png?)

同时，RNN是没办法并行计算的，这对模型的学习速度是一个很大的限制，但是Self-Attention是可以使用GPU进行并行计算的，这样就比RNN更容易训练，所以，现在RNN逐步被Self-Attention取代，甚至完全替代

# 带注意力机制的CNN

因为注意力机制，实际上并不关心输入是一个序列这一个事实，只不过它可以用在RNN机器翻译上（这个任务的输入输出是序列），但出于这种注意机制的目的，它实际上并没有使用输入向量是一个序列的事实

所以原则上，我们可以使用完全相同的注意机制来构建可以处理不是序列的其他类型数据的模型，比如说CNN

不过注意一下，因为 EECS498 课程是19年的版本，而真正意义上的 Vision Transformer 是在20年出现的，所以这里的 CNN + 注意力机制作为一个了解即可

## 对比注意力机制与CNN

一张彩色图片就是一个三维张量，我们可以将其考虑为一个序列，每个位置的RGB信息就组成了一个三维向量

![lhy_attention_32](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_33.png?)

但是，这样就与CNN有了很大不同，因为CNN是逐步看到全局，而注意力机制则是直接计算与全局的关联

![lhy_attention_34](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_35.png?)

或者可以将CNN看做一种带有感受野的注意力，是一种简化版或者受限制版，注意力机制是一种带有可学习感受野的CNN，是一种复杂化的CNN

![lhy_attention_35](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_36.png?)

我们知道，相对简单的模型，对数据集的要求没有那么高，但是复杂的模型则有更高要求，如果数据集不够，就容易造成欠拟合，所以CNN对数据集的要求比Self-Attention低

如下图所示，在小数据集上，CNN的表现更好，但是随着数据集的增大，Self-Attention的效果超过了CNN（前者弹性更大），所以需要根据具体情况去判断要使用哪种模型

![lhy_attention_36](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_37.png?)

## 图像标题生成

我们可以使用注意力机制，来生成图像的说明

我们使用CNN来计算图像的特征，然后使用注意力机制来生成字幕

模型通过对输入图像的特征网格进行加权重组来生成自己的新上下文向量

![EECS498_L13_36](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_36.png??raw)

## 人眼视觉：焦点

人眼的视觉，是很奇特的

我们知道人眼是一个球体，然后你在一侧有一个晶状体，相当于一个透镜，然后在你眼睛的后部有一个叫做视网膜的区域，光线通过晶状体进入眼球，投射到视网膜上

视网膜上包括实际检测光的感光细胞，它们感光之后，产生信号并且发送给大脑，然后大脑进行处理并理解为我们现在看到的东西

但是，视网膜的所有部分不是平等的，实际上视网膜正中央的一个特定区域，学名为“fovea”，中文名"黄斑窝"或者"中央凹"。这是视网膜中央部位的一个小凹陷，直径约为1.5毫米，它的特点是细胞密度最高，主要由视锥细胞构成，负责我们的中心视力和色彩视觉。当我们集中注意力看某个物体时，我们的眼睛会自动调整，使得光线尽可能地落在黄斑窝上，这样可以得到最清晰的视觉效果。

![EECS498_L13_40](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_41.png??raw)

从上图也可以看到，视网膜不同区域的敏感度不一样（或者可以表现为视力），越是边缘区，视力越差，在中央凹这里，视力达到最高

# 注意力机制下的图神经网络（GNN）

当然，注意力机制还可以使用在图神经网络上

在前面，我们使用注意力机制去寻找不同向量之间的关联性，但是在图网络这里，节点之间的关联性是直接给你的，所以在这里只需要去计算有边相连的节点就可以

![lhy_attention_39](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_40.png?)

# 注意力机制的通用化：注意力层

## 通用的注意力层

我们尝试将注意力机制抽象并且概括，将其用在更多类型的任务中，尝试得到一个通用的层，然后直接插入到RNN和CNN中进行使用，这种方法可以很好地对注意力机制进行重建

![EECS498_L13_45](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_45.png?)

我们的输入是一个查询向量（Query Vector，即$q$），这个将取代隐藏状态向量，查询向量的作用是帮助模型确定应该关注的输入数据的哪些部分，从而提高模型的性能和准确性。它是实现注意力机制的关键组件，使模型能够更好地处理复杂和大规模的数据。

我们也有一个输入向量（即$X$），来代替我们想关注的这组隐藏向量

然后还有一些相似函数，来对比查询向量和数据库中的每个输入向量，当然，实际上主要是点积作为相似函数

然后就是使用softmax，对相似函数的输出进行处理，得到注意力权重，然后对输入向量进行加权来计算输出向量

![EECS498_L13_46](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_46.png?)

实际上，上面这种注意力机制中的相似函数$f_{att}$，是早期的工作，现在使用了更新的方法，也就是使用向量之间的简单点积，这样更有效而且性能更好

![EECS498_L13_47](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_47.png?)

同时为了限制相似性函数计算结果过大过小，我们使用缩放点积（如上图所示），来使得相似性有一个限定范围，这样就可以限制梯度不会过大，导致梯度爆炸（因为这个相似性会进入softmax进行计算的，这个值太大，也会导致反向的梯度很大，同时如果向量e中一个元素远大于其他元素，也会导致一个非常高的softmax峰值分布，导致梯度几乎在任何地方接近消失）

此外，如果向量很长，那么就更容易产生非常大的数值，然后我们除以平方根，就可以消除这种影响

然后，我们想要允许多个查询向量，所以以前我们总是在每个时间步都有一个查询向量，在解码器中，我们使用该查询向量在所有输入向量上生成一个概率分布

现在我们想概括注意力机制，并且有一组查询向量 $Q$ 和输入向量 $X$（以矩阵形式出现的），对于每一个查询向量，我们想在每个输入向量上生成一个概率分布，以便于计算相似度

我们想计算每个查询向量和每个输入向量之间的相似度，我们就使用比例点积来作为计算相似度的函数（这里改进为矩阵乘法），然后我们计算概率分布的时候，就可以使用针对单个维度的softmax，在这里我们是在第一维上计算

然后我们使用 softmax 就可以得到注意力权重了，然后使用注意力权重矩阵来对输入矩阵进行加权求和，得到输出矩阵 $Y$

这里实际上输入向量有两个功能，一个是计算注意力权重，另一个是计算输出向量，但是我们可以将这个输入向量转换为两个向量：键向量和值向量

这里我们使用两个可学习的矩阵完成这个操作，一个是键矩阵$W_k$，另一个是值矩阵$W_v$

![EECS498_L13_50](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_50.png?)

我们这里使用查询向量和键矩阵来计算相似性，使用注意力权重矩阵和值向量来计算输出，这使得模型在如何正确使用其输入数据方面具有更大的灵活性

下图中，底部我们有四个查询向量，然后左侧是一组输入向量

首先我们使用键矩阵来计算输入向量的键矩阵，然后将键矩阵与查询向量进行对比，得到非标准化相似性分数矩阵

然后进行softmax操作，得到对齐分数，然后继续计算

![55](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_56.png?)

最后输出向量 $Y$，其中就包括了更多的信息，而且其中更重要的信息占比更大（不过也有一点不重要的信息）

## 自注意力层

在之前的情况下，我们都是有两组输入：查询向量和输入向量，但是实际上我们只有一组向量作为输入，并且我们想将输入集中的每个向量与其他向量进行比较，使用整个输入来找到关键信息，所以我们需要将输入进行转换，我们再使用一个可学习的查询矩阵 $W_Q$，来获得查询向量

![EECS498_L13_63](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_63.png?)

然后类似地，我们将使用键矩阵来获得键向量，然后计算查询向量和键向量之间的相似性矩阵$E$​（也就是查询得到的重要性信息），进行放缩然后使用softmax得到概率分布，加上值矩阵得到注意力权重矩阵，加权求和之后得到输出向量（实际上就是进行信息汇总）

当然，这里实际上是通过点乘的方法计算 $Q$ 和 $K$ 中每一个事物的相似度，进而找到对于 $Q$ 更重要的那些（通过权重表示重要程度），而这里的 $K$ 和 $V$​ 实际上是一样的或者说是有联系的，这样子 QK 点乘才可以指导哪些信息重要

## 多头注意力（Multi-head Self-Attention）

这是一种注意力机制的改进，可以计算不同类型的关联程度，使用更为广泛，比如说在文本翻译、语音辨识等方面这样可以得到更好的效果，至于要多少个 head，则又是一个超参数

在普通自注意力机制下，我们每个类型只使用一个可学习的权重矩阵来判断关联性，但是我们可以使用多个权重矩阵表示不同类型的关联程度，如下图所示，我们使用了两个查询矩阵$W^{q,1}$，$W^{q,2}$，这样就可以计算更多种的相关形式，每个查询矩阵负责计算一种相关性，不会去计算其他相关性中的数值，可以类比卷积层中的不同卷积核

![lhy_attention_25](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_26.png??raw=)

多头注意力的具体结构是，使用一些更小的权重矩阵，分别计算 QKV 向量，然后，$a^i$的查询向量的分量 $q^{i,1}$只跟向量的$k^{i,1}$​等进行计算，或者使用标准的权重矩阵，得到向量之后进行拆分，然后使用各个子向量进行注意力计算，这两种方式本质上是一致的，并且最后都会将所得到的向量进行拼接，恢复长度

![](https://miro.medium.com/v2/resize:fit:1400/1*PiZyU-_J_nWixsTjXOUP7Q.png?)

使用多头注意力，可以增强模型的建模能力，表征空间更加丰富，实际上每个头对应一个低维的子空间特征，也就是每个头对应特征不同的子空间。整体的参数量其实是没有发生改变的

## 位置编码（Positional Encoding）

当然，我们可以发现，注意力机制是没有空间信息的，对向量的操作是一样的，所有的向量之间的距离是一样的，这在文本翻译等领域是有不足的，这种方式不会考虑词语之间的顺序关系，因为我们知道动词往往出现在句首，解决方法就是增加一个包含位置信息的向量，将位置信息塞入，不同的位置就有不同的向量

当然这里要注意一下，一般来说位置编码是与内容无关的，也就是只要位置固定了，不论这个位置的内容是什么，编码都是固定的

![lhy_attention_28](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_29.png??raw=true)

第一种设置位置向量的方式就是人工设置，每个位置的位置向量$e^i$是唯一的，但是它可能无法完全适应所有的任务和数据，而且也需要一定的领域知识和经验才能设计出有效的编码方案，在Attention is all you need中是使用三角函数来编码的，基于正弦和余弦函数的位置编码方法，就是一种典型的"hand-crafted"编码方式。在这种方法中，每个位置的编码是通过正弦和余弦函数在不同频率上的值得到的。

第二种方法就是通过深度学习的方法，学习出一个合适的位置向量，有下面几种方法

![lhy_attention_29](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_30.png??raw=true)

常用的一种编码就是正弦余弦编码，这

## 掩码自注意力机制（Masked Self-Attention）

掩码自注意力机制（Masked Self-Attention Mechanism）是自注意力机制的一个变体，主要用于处理序列数据，特别是在生成任务中，如语言模型和序列到序列（Seq2Seq）模型中。掩码自注意力机制通过使用一个掩码（mask）来阻止模型在计算注意力得分时访问未来的信息，从而保证模型只能基于当前和之前的信息做出预测。

如下图所示，每个输出的向量只考虑在它之前的输入向量。这样的话，在训练的时候，每个词语只能获取其之前的词语的信息

![](https://peterbloem.nl/files/transformers/masked-attention.svg)

其数学表达式如下
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$
$M$ 是一个掩码矩阵，其元素值为$- \infty$​（对应于被掩盖的位置）或 0（对应于未被掩盖的位置）。在应用 softmax 函数之前，被掩盖的位置的得分会变为负无穷，经过 softmax 后，这些位置的权重接近于零，从而有效地阻止了模型访问未来的信息

在 PyTorch 中，我们可以构建一个上三角矩阵，然后填充负无穷来构建掩码矩阵

```python
mask = torch.triu(torch.ones(dim, dim), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))
```

使用了 `torch.triu` 函数来创建一个上三角矩阵，其中对角线以上的元素都设置为1，然后使用 `masked_fill` 函数将这些1替换为负无穷大

## 归一化：LayerNorm

在注意力机制中的归一化，并不是批量归一化操作，是层归一化，这是一种可以进行学习的归一化操作，其参数计算公式如下，其中 $x_i$ 为输入的分量
$$
\begin{align*}
\mu &= \frac{1}{d} \sum_{i=1}^{d} x_i, \\
\sigma^2 &= \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
\end{align*}
$$
对输入进行归一化的数学公式如下，其中，$\epsilon$ 是一个很小的正数，用来防止除以零的情况发生。
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$
对归一化后的向量 \(\hat{x}\) 进行缩放和平移变换：
$$
y_i = \gamma \hat{x}_i + \beta
$$
其中，$\gamma$ 和 $\beta$ 是可学习的参数，分别用于缩放和平移归一化后的特征，而且是在整个训练过程中基于所有批次和所有样本的反馈进行学习的，而不是针对单个批次或样本进行学习。这样做是为了确保这些参数能够捕捉到整个训练集的统计特性

使用这种层归一化的原因是这样的

1. **独立于批次大小**：层归一化对每个样本进行独立归一化，这意味着它不依赖于批次大小。这在处理可变大小的输入或在推理时使用不同的批次大小时非常有用。相比之下，批量归一化在计算均值和方差时依赖于整个批次，这可能在小批次或不一致的批次大小下导致不稳定性。
2. **顺序数据的适应性**：Transformer模型通常用于处理顺序数据（如文本或时间序列）。在这种情况下，层归一化可以更有效地处理序列中的每个元素，因为它对每个元素进行独立归一化，而不是依赖于整个批次中的所有元素。
3. **计算效率**：在某些情况下，层归一化可以比批量归一化更高效，特别是在处理长序列或使用小批次时。因为层归一化不需要在批次维度上计算均值和方差，这可以减少计算量。
4. **兼容性**：层归一化更容易与自注意力机制结合使用，这是Transformer模型的核心部分。由于自注意力机制处理的是序列中的每个元素，层归一化的逐元素归一化策略与之更加兼容。

## 例子：带有自注意力层的CNN

我们想在视觉任务中使用注意力机制

![EECS498_L13_81](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_81.png?)

## Transformer的变形

Transformer有很多变形，下图中的论文就总结了一些，而且因为Transformer计算量大，所以未来一个研究方向就是减小其计算量，提高计算速度

![lhy_attention_41](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_42.png??raw)

# 三种处理序列方法总结

## 循环神经网络

最明显的就是RNN，给定输入向量的序列，产生输出向量的序列

优点是非常擅长处理长序列，擅长在非常长的序列上传递信息，所以最终隐藏状态和输出，取决于整个输入序列

但是问题是，它并没有很好的并行处理，它非常依赖于数据的顺序，但是我们知道，我们构建大型神经网络都是使用GPU和张量来并行处理的，这就无法很好的利用算力，所以难以构建大型RNN

## 一维卷积

我们可以使用一维卷积的方式进行计算，因为序列中的输出元素可以独立于其他输出元素进行计算，可以打破循环神经网络的顺序依赖性

但是问题是难以处理长序列，除非堆积很多一维卷积层，因为输出序列中的每个点只能看到几个点，只有多层卷积之后才可以一个点看到整个序列

## 自注意力

可以克服前面两种方法的缺点，很适合在GPU上并行计算，如果给定一组向量，它将每个向量与每个其他向量进行比较，这类似于RNN，每个输出都取决于所有输入

但是缺点是，会占用大量内存

不过，结果证明，如果我们想构建一个处理序列类型的神经网络，我们可以只使用自注意力机制来完成，这种工作就是transformer

## Transformer

transformer就是将自注意力作为比较输入向量的唯一机制，工作方式就是首先接受一个输入序列，然后将其通过一个自注意力层，同时增加一个残差连接来完成归一化操作（对于序列模型也是有用的）

但是，归一化层不涉及不同向量之间的交互，仅仅是对单个向量完成归一化

![EECS498_L13_92](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/EECS498_L13_92.png??raw)

在这个归一化层之后，我们得到的还是一组向量，就是应该前馈的多层感知机，这是一个全连接的神经网络，会独立地对每个向量进行处理

然后我们继续增加一个残差连接进行归一化处理，然后输出向量Y

我们一般将这些层封装为一个块，称为Transformer，它可以是处理向量序列的大型模型的基本构建块

输入是一组向量 x，输出也是一组向量 y，不会改变输入输出的数量，但是维度可能改变，向量之间的唯一交互发生在自注意力层内部，因为归一化和前馈感知机都是在单个向量上进行独立处理的

因为这些良好特性，所以这种模块非常适合GPU上进行并行计算

选择Transformer我们需要设置几个超参数，一个是模型深度

事实证明，我们可以使用这些模型为许多序列预测或语言预测任务实现非常相似的效果，或许这就是自然语言处理的通用范式





# 注意力机制下的语音识别

我们之前是将每10ms的声音信号片段转化为一个向量，那么1秒的信号就转化为一百个向量的集合，可以看到，随便一句话所代表的序列长度是非常长的，因为注意力矩阵的大小与序列长度有关，这样注意力矩阵就会非常庞大，会占用庞大的空间

![lhy_attention_31](https://raw.githubusercontent.com/Michael-Jetson/Images/main/UpGit_Auto_UpLoad/lhy_attention_32.png??raw=true)

所以就有了一种Truncated Self-Attention的方式，为了解决上面这个问题，一种方法是限制自注意力机制只关注输入序列的一部分，而不是全部。这就是所谓的Truncated Self-Attention。在这种机制中，对于序列中的每一个位置，我们只计算它与前面k个位置的注意力权重，而不是与所有位置的注意力权重。这里的k是一个预设的参数，表示注意力的范围或者窗口大小。

这样，就只考虑一个小范围的向量就可以，这是因为在语音识别中，一个音素不需要考虑很远的其他音素，很远的因素影响很小，只需要考虑附近音素就可以

# Transformer

## 标准架构

Transformer是一种使用了注意力机制的 Seq2Seq 的模型，主要特点就是它全面使用了注意力机制，而不是传统的循环神经网络（RNN）或卷积神经网络（CNN）结构，来处理序列中的依赖关系

在Transformer模型中，注意力机制被用于所有的输入和输出之间，这允许模型在任何两点之间都可以直接交互，从而处理复杂的依赖关系。此外，Transformer模型使用了多头注意力（Multi-Head Attention）结构，允许模型从不同的表示子空间同时学习输入之间的交互，进一步增强了模型的表达能力。

Transformer 的整体架构如下图所示，编码器进行理解，解码器进行生成

![](https://camo.githubusercontent.com/278c42ceeaf0ec778864a7f9b772914bce779f158ed87796c41eea388cf842b2/68747470733a2f2f696d676d642e6f73732d636e2d7368616e676861692e616c6979756e63732e636f6d2f424552545f494d472f74662d2545362539352542342545342542442539332545362541312538362545362539452542362e6a7067?)

Transformer 的 Encoder 部分的结构，大概就是多层 Self-Attention 层的堆叠，产出处理之后的特征向量，每一个 Self-Attention 层都有两个模块，首先进行一次 Multi-Head Attention 加残差连接和归一化，然后进行一次 Feed Forward进行非线性变换

Decoder 部分的功能就是产生最终输出，其结构是多个层的堆叠，并且每个层都会接受编码器的输入进行 Cross-Attention 操作，对特征向量或者词向量进行进一步增强，用于最后的生成任务

![](https://camo.githubusercontent.com/8e84e1e9c3d790d2444fdeca08fc0227ff569e6d22c458ab81537a2dc974a1ff/68747470733a2f2f696d676d642e6f73732d636e2d7368616e676861692e616c6979756e63732e636f6d2f424552545f494d472f74662d65642d2545352541342538442545362539442538322e6a7067?)

## NLP 中 Transformer 的处理过程

在 NLP 任务，比如说机器翻译中，词是一个个生成的，首先在输入部分，一次性输入所有的需要翻译的词汇，然后进行计算，最终编码器的输出作为 K V 值输入解码器，而上一个输出作为 Q 进行查询

![](https://camo.githubusercontent.com/0c744d5a452a00ef4bbaead312179c646a5db61d60ae58390ac151a205e9b9a9/68747470733a2f2f696d676d642e6f73732d636e2d7368616e676861692e616c6979756e63732e636f6d2f424552545f494d472f74662d2545352538412541382545362538302538312545372539342539462545362538382539302e676966?)

Q 是查询变量，是已经生成的词，来源解码器

K=V 是源语句，来源于编码器

当我们生成这个词的时候，通过已经生成的词和源语句做自注意力，就是确定源语句中哪些词对接下来的词的生成更有作用，首先他就能找到当前生成词

通过部分（生成的词）去全部（源语句）的里面挑重点

Q 是源语句，K，V 是已经生成的词，源语句去已经生成的词里找重点 ，找信息，已经生成的词里面压根就没有下一个词

## 位置编码

在原始论文中，是通过三角函数实现了固定表征的，这样子每个单词的位置就是确定的（哪怕在不同的句子中也要一样，比如说两个不同长度句子中的第二个单词的位置相等），对于不同的句子，相同位置的距离是一致的（间隔是不会随着句子长度而改变的），并且可以推广到更长的没见过的句子，或者说可以认为是见过的句子的线性组合

位置编码的特征向量然后会加到嵌入向量上，这样子就可以解决了 Transformer 对顺序不敏感的问题，并且使用残差连接，使得深层网络也可以获得位置信息

# Vision Transformer（ViT）

## Vision Transformer

第一篇基于 Transformer 的视觉论文是 **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

这个工作在 Patch 中进行注意力机制，避免了在像素级别上进行注意力计算，节省了计算资源，同时仅仅使用了 Encoder 结构，并没有使用 Decoder 部分

![WZ_VisionTransformer_5](./assets/WZ_VisionTransformer_5.png?)

并且作者发现，位置编码的形式并不是很重要，一维、二维或者其他编码对精度的影响微乎其微，但是是否有位置编码对结果的影响是较大的，如下图所示，有编码的精度相对无编码高出三个点，而不同编码的差距则很小

![WZ_VisionTransformer_6](C:/Users/pengf/Desktop/ML_DL_CV_with_pytorch/ComputerVision/assets/WZ_VisionTransformer_6.png?)

## Swin Transformer

这个工作使用 Transformer 实现了类卷积结构，优化了分类任务的结果
