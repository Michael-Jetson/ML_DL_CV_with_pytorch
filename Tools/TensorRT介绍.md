# TensorRT介绍

目前主流部署平台有NVIDIA GPU，Intel CPU，ARM和FPGA

深度学习目前主要是在GPU上进行训练和推理，在这个过程中就需要一些软件的框架来进行深度学习加速

TensorRT三一种用来进行高性能深度学习推理的SDK，包含了推理优化器和运行时环境，可以为深度学习推理应用提供低延迟和高吞吐量，在推理过程中，基于TensorRT的应用程序执行速度可以比CPU平台快四十倍

TensorRT是以英伟达的并行编程模型CUDA为基础构建的，针对多种深度学习推理应用的生产部署提供INT8和FP16优化，可以显著减少应用延时，并且支持在多个深度学习框架中将已训练好的模型导入TensorRT

应用优化后，TensorRT可以选择平台特定的内核，在英伟达的GPU和嵌入式平台上更大限度提高深度学习推理性能

## TensorRT安装

### Zip或者tar安装

- 优点是可以自行指定安装目录，可以同时安装多个版本的SDK
- 缺点是使用时必须指定LD_LIBRARY_PATH

### Debian安装

- 装入系统目录中，只能安装一个版本
- 环境变量中包括目录，不需要单独引用

# 构建TensorRT引擎

## TensorRT编程模型

TensorRT三分两个阶段进行操作的，构建阶段和运行时阶段

- 构建阶段

  - 通常离线执行，需要需要给TensorRT 提供一个模型定义，然后 TensorRT 会针对目标 GPU 对其进行优化
  - TensorRT构建阶段的最高级别接口是Builder(C++)。构建器负责优化一个模型，并产生一个引擎
  - 为了建立一个引擎，需要完成以下步骤
    1. 创建一个网络定义
    2. 为构建器指定一个配置
    3. 调用构建器来创建引擎

- 运行时阶段

  使用优化的模型来运行推理，TensorRT 执行阶段的最高级别接口是Runtime(C++)，使用时有以下步骤

  1. 反序列化plan以创建引擎
  2. 从引擎创建可执行上下文
  3. 然后，反复执行：填充输入缓冲区以进行推理以及在执行上下文上调用两个函数 enqueue() 或 execute() 以运行推理

大概的流程图如下

![TensorRT_L2_](/home/pengfei/下载/TensorRT_L2_.png)

使用一些深度学习框架做网络定义文件，然后使用Builder构建器做Engine，主流可以序列化为字节数据，字节数据也可以反序列化为Engine

## 网络定义（ONNX导入）

ONNX是一个框架无关的选项，可用于TensorFlow、PyTorch等中的模型。TensorRT支持使用TensorRT API或trtexec从ONNX文件自动转换。
ONNX转换是全有或全无，这意味着模型中的所有操作都必须由TensorRT支持（也就是说有一个操作是TensorRT不支持的，就无法使用，或者必须为不受支持的操作提供自定义插件），ONNX转换的最终结果是一个单一的TensorRT引擎
使用ONNX文件导入模型，有两种方法：

- 使用nvonnxparser直接解析ONNX文件。这种方法TensorRT应该是在内部做了某种缓存，系统启动之后部署程序第一次启动时，整个加载过程会显得略长，但是后面的加载过程会比较顺利。但是系统一旦重启，仍然会经历上述过程。相对不推荐
- 使用trtexec或者TensorRT API将ONNX转换为TensorRT engine。优点是每次加载速度都比较快，缺点是换台电脑就需要重新转换，因为TensorRT engine依赖于特定的GPU和TensorRT，GPU不一致就无法运行（这个容易发现问题）。当然只要保证GPU和TensorRT一致，engine可以多机使用

下面是一段程序

```C++
nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
//IBulider是最高级的定义，这里创建一个IBuilder实例，用于构建优化后的推理引擎
nvinfer1::INetworkDefinition* network = builder->createNetwork();
//定义模型的网络架构
nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
//做一些配置，设置构建过程中的参数和优化配置
// 1. create network use C++ API from scratch
// 2. use onnxparser
nvonnxparser::IParser* parser =
nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
//如果使用ONNX模型，则创建一个nvonnxparser::IParser实例来解析ONNX文件，并将其加载到定义的网络中。
parser->parseFromFile(xxx);
//添加一个优化配置文件（OptimizationProfile），这对于动态形状的输入特别重要
config->addOptimizationProfile(profile);
config->setMaxWorkspaceSize(1024 << 30);
config->clearFlag(BuilderFlag::kTF32);
config->setFlag(BuilderFlag::kFP16); // config->setFlag(BuilderFlag::kINT8);
// config->setInt8Calibrator();
config->setDefaultDeviceType(DeviceType::kDLA);
config->setDLACore(0);
config->setFlag(BuilderFlag::kGPU_FALLBACK);
nvinfer1::ICudaEngine* mEngine = builder->buildEngineWithConfig(*network, *config);
```

NetworkDefinition接口(C++)被用来定义模型。将模型转移到TensorRT的最常见路径是以ONNX格式从框架中导出，并使用TensorRT的ONNX解析器来填充网络定义。
也可以使用TensorRT的Layer (C++) and Tensor (C++) 接口一步一步地构建定义。
无论你选择哪种方式，必须定义哪些张量是网络的输入和输出。没有被标记为输出的张量被认为是瞬时值，可以被builder优化掉
输入和输出张量必须被命名，以便在运行时，TensorRT知道如何将输入和输出缓冲区绑定到模型上。

### TensorFlow-ONNX构建网络

ONNX的github上提供了工具tf2onnx可以很方便的把保存的tf模型转换
为onnx。工具地址为：https://github.com/onnx/tensorflow-onnx 。
tf2onnx 支持tf-1.x, tf-2, keras, tflite模型。可以使用`pip install -U
tf2onnx `安装该工具。
转换则可以使用以下命令进行

```shell
python -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx
```

tf2onnx默认使用opset版本为9，可以使用--opset=xx指定其他版本的
opset。

### PyTorch-ONNX构建网络

PyTorch本身就有导出onnx的函数

```python
import torch
import torch.nn as nn
import torchvision

# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(10, 50) # 第一层全连接层，输入特征10，输出特征50
        self.relu = nn.ReLU()        # 激活函数
        self.fc2 = nn.Linear(50, 1)  # 第二层全连接层，输入特征50，输出特征1

    def forward(self, x):
        # 定义前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleModel()
#也可以SimpleModel().cuda()
# 设置模型为评估模式
model.eval()


# 创建一个随机数据作为模型输入
# 假设我们的输入有10个特征
x = torch.randn(1, 10)

# 导出模型
# 你需要指定一个样例输入，这样ONNX导出器可以推断所有操作的形状
torch.onnx.export(model,               # 运行的模型
                  x,                   # 模型输入的样例数据（或者是一个元组，如果模型有多个输入的话）
                  "simple_model.onnx", # 输出文件名
                  export_params=True,  # 是否导出模型参数权重
                  opset_version=10,    # ONNX版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names = ['modelInput'],   # 输入层的名字
                  output_names = ['modelOutput'], # 输出层的名字
                  dynamic_axes={'modelInput' : {0 : 'batch_size'},    # 指定输入数据的动态轴
                                'modelOutput' : {0 : 'batch_size'}})
# 提供输入和输出名称是为模型图中的值设置显示名称。设置这些名称并不改变图的语义，它只是为了方便阅读
```



注意，在PyTorch中，将模型导出为ONNX格式需要提供一个样本输入`x`，因为ONNX导出过程需要执行一次模型的前向传递。这个前向传递是为了：

1. **推断图结构**：PyTorch是一个动态图框架，这意味着计算图在每次执行时都会重新构建。与此相反，ONNX需要一个静态图表示。通过执行前向传递，PyTorch可以确定在给定输入下的确切计算图。
2. **推断张量的形状**：在动态图框架中，张量的形状是在运行时确定的。为了导出为静态图，ONNX需要知道图中每个操作的输入和输出张量的形状。提供样本输入`x`允许ONNX在导出过程中推断出这些形状。
3. **推断操作属性**：某些PyTorch操作可能根据输入数据的具体形状和类型在内部做出不同的决策。样本输入确保这些决策可以在导出过程中正确地反映出来。
4. **优化图结构**：有时在导出过程中会执行图优化，比如常量折叠。这需要实际的数据通过图运行，以确定哪些操作可以被优化。

总的来说，样本输入`x`对于ONNX导出器来说是必要的，因为它依赖于具体的数据流来构建一个准确、优化的模型表示。在没有这个具体的数据流的情况下，导出器无法准确地捕捉到模型的行为和结构。

### ONNX查看

可以通过网站Netron打开生成的ONNX文件，查看网络结构。
可以的使用onnx.checker校验生成的文件

```python
import onnx
# Load the ONNX model
model = onnx.load("alexnet.onnx")
# Check that the model is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
```

可以使用ONNX runtime来加载运行导出的ONNX文件

```python
import onnxruntime as ort
ort_session = ort.InferenceSession("alexnet.onnx")
outputs = ort_session.run(
None,
{"actual_input_1": np.random.randn(10, 3, 224,224).astype(np.float32)},
)
print(outputs[0])
```

还可以进行网络简化

![28](/home/pengfei/图片/自动驾驶深度学习部署/01_TensorRT/28.png)

## 使用TensorRT Layer and Tensor API构建网络

这个的问题是当网络结构更改的时候，必须重新使用C++编写或者修改代码，所以不推荐使用，了解有这种方法即可

下面是程序

```C++
// 创建builder，network:
IBuilder* builder = createInferBuilder(gLogger);
INetworkDefinition* network = builder->createNetworkV2(
1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
//添加Input layer, 并设定输入尺寸(此处使用动态batch size).
//当有多个输入时，和一个输入是一样一样的
auto data = network -> addInput(INPUT_BLOB_NAME, dt, Dims3{-1, 1, INPUT_H, INPUT_W});
// 添加卷积层，并设定输入为data, 设置strides和权重
auto conv1 = network -> addConvolution(*data->getOutput(0), 20, DimsHW{5, 5},
weightMap["conv1filter"], weightMap["conv1bias"]);
conv1->setStride(DimsHW{1, 1}); // Note: Weights passed to TensorRT layers are in host memory.​
//添加Pooling layer:​
auto pool1 = network -> addPooling(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
pool1->setStride(DimsHW{2, 2});
//添加 FullyConnected 和Activation layers:​
auto ip1 = network -> addFullyConnected(*pool1->getOutput(0), 500, weightMap["ip1filter"],
weightMap["ip1bias"]);
auto relu1 = network -> addActivation(*ip1->getOutput(0), ActivationType::kRELU);
// Add the SoftMax layer to calculate the final probabilities and set it as the output
auto prob = network -> addSoftMax(*relu1->getOutput(0));
prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//标记输出
network->markOutput(*prob->getOutput(0));
```

