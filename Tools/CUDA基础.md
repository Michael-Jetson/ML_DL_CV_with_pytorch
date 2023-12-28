# 第一个CUDA程序

我们在目录下新建一个文件，命名为hello.cu，cu表示这是一个CUDA程序

```c++
#include<stdio.h>
int main()
{

        printf("helloword\n");
        return 0;
}
```

然后在此目录下打开终端，进行编译并且执行

```shell
nvcc hello.cu -o hello
./hello
```

然后就可以输出helloword了

这里的nvcc就是专属cuda程序的编译器，如果使用gcc等编译器直接编译cuda文件的话会报错，实际上nvcc与g++的内容大多数是共用的，只不过nvcc有一些专门编译cuda函数的内容

当然我们可以使用CMake构建项目，而不是使用makefile，我们配置的CMakeLists.txt是这样的

```cmake
cmake_minimum_required(VERSION 3.8)
project(CUDA_TEST)
 
find_package(CUDA REQUIRED)
 
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
 
cuda_add_executable(cuda_test src/introduction.cu)
target_link_libraries(cuda_test ${CUDA_LIBRARIES})
```

使用CUDA package，include CUDA的路径，然后使用cuda_add_executable编译cu文件

当然，我们可以直接在LANGUAGES后面加CUDA，即可使用add_executable，否则无法cmake

```cmake
project(CUDA_TEST LANGUAGES CXX CUDA)
add_executable(cuda_test src/introduction.cu)
```



# CUDA中的线程与线程束

## 概念

gpu不是单独的在计算机中完成任务，而是通过协助cpu和整个系统完成计算机任务，把一部分代码和更多的计算任务放到gpu上处理，逻辑控制、变量处理以及数据预处理等等放在cpu上处理，所以在这里会使用host代表CPU和内存，使用device来代表GPU和显存 。一个程序很可能是CPU开始执行，然后在某些时刻将某些计算放在GPU上计算，这个时候就需要用到kernel（核函数），这个kernel是一个以线程为单位来计算的函数，在这里的作用相当于一个入口

如下图所示，一个kernel会分配一个grid，一个grid是层级架构，一个grid里面就有多个block，一个block里面有多个thread，这里注意一下，grid和block是逻辑概念，实际上GPU中不存在物理意义上的grid和block，在编程中使用是为了方便

总的来说，grid是一组被一起启动的CUDA核心的集合，用于定义整个GPU上的 并行作业，grid中的block是一个小工作单元，包括了一组可写作的thread，提供了一种小范围内的数据共享和协作模式，每个block可以在GPU的一个多处理器上执行，grid和block都可以是一维、二维或者三维的，便于灵活组织数据

thread是基本执行单元，在同一个block和grid中的thread会执行相同的核函数

![AutoDriverHeart_TensorRT_L2_7](/home/pengfei/文档/ML_DL_CV_with_pytorch/Tools/assets/AutoDriverHeart_TensorRT_L2_7.png)

在grid中，每个block都有一个shared memory（block级别的共享内存），并且还有多级的memory，同时每个thread都直接连接一个register（寄存器）和一个local memory
## block和thread的遍历

一个kernel中有这么多的子单元，我们自然想的是如何进行遍历

如下图所示，我们一共有八个thread，那么就设置inputSize为8，每个block有四个thread，我们就设置blockDim也就是维度为4，然后gridDim自然就是2，然后我们进行传参

在 CUDA 编程中，`dim3` 是一个用来定义三维空间尺寸的struct数据类型，包含三个无符号整数成员：x, y, z。这些成员分别代表三维空间中的尺寸，dim3在指定核心（kernel）启动时的 grid 和 block 的尺寸时非常有用。在 CUDA 编程中，`dim3` 通常用于定义 grid 和 block 的尺寸，比如例程

```c++
dim3 blockDim(16, 16); // 每个 block 有 16x16 个 threads
dim3 gridDim(10, 10);  // grid 尺寸为 10x10 个 blocks
kernel<<<gridDim, blockDim>>>(...); // 启动核心
```

`dim3` 类型是 CUDA 特有的，主要用于简化并行计算任务中对多维数据的管理

如果不指定 `dim3` 的某个维度，该维度的默认值为 1。例如，`dim3 dim(16)` 相当于 `dim3 dim(16, 1, 1)`。

![AutoDriverHeart_TensorRT_L2_9](/home/pengfei/文档/ML_DL_CV_with_pytorch/Tools/assets/AutoDriverHeart_TensorRT_L2_9.png)

可以进行遍历，每个thread和block都有一个索引值，也就是threadIdx和blockIdx

![AutoDriverHeart_TensorRT_L2_10](/home/pengfei/文档/ML_DL_CV_with_pytorch/Tools/assets/AutoDriverHeart_TensorRT_L2_10.png)

核函数一定要加`__global__`来表明其为一个核函数，这里有个遍历用的核函数

```c++
__global__ void print_idx_kernel(){
    printf("block idx:(%3d,%3d,%3d),thread idx:(%3d,%3d,%3d)\n",
          blockIdx.z,blockIdx.y,blockIdx.x,
          threadIdx.z,threadIdx.y,threadIdx.x)
} 
```

不过需要注意一下遍历顺序，是先遍历z，后y，最后x

![AutoDriverHeart_TensorRT_L2_11](/home/pengfei/文档/ML_DL_CV_with_pytorch/Tools/assets/AutoDriverHeart_TensorRT_L2_11.png)

在核函数的调用格式上与普通C++的调用不同，调用核函数的函数名和（）之间有一对三括号，里面有逗号隔开的两个数字。因为一个GPU中有很多计算核心，可以支持很多个线程。主机在调用一个核函数时，必须指明需要在设备中指派多少个线程，否则设备不知道怎么工作。三括号里面的数就是用来指明核函数中的线程数以及排列情况的。核函数中的线程常组织为若干线程块（thread block）。
三括号中的第一个数时线程块的个数，第二个数可以看作每个线程中的线程数。一个核函数的全部线程块构成一个网格，而线程块的个数记为网格大小，每个线程块中含有同样数目的线程，该数目称为线程块大小。所以核函数中的总的线程就等与网格大小乘以线程块大小，即<<<网格大小，线程块大小 >>>
核函数中的printf函数的使用方法和C++库中的printf函数的使用方法基本上是一样的，而在核函数中使用printf函数时也需要包含头文件<stdio.h>,核函数中不支持C++的iostream。
cudaDeviceSynchronize();这条语句调用了CUDA运行时的API函数，去掉这个函数就打印不出字符了。因为cuda调用输出函数时，输出流是先放在缓存区的，而这个缓存区不会核会自动刷新，只有程序遇到某种同步操作时缓存区才会刷新。这个函数的作用就是同步主机与设备，所以能够促进缓存区刷新。

# CUDA的矩阵乘法计算实现

## 流程概述

首先，矩阵计算肯定是基于CPU和GPU联合处理的，也就是需要host端到device端的数据传输

这里需要在两端进行内存空间的分配，这一操作是在host上执行的，还有配置核函数参数、数据传输等，总的来说GPU只需要启动核函数进行计算

![AutoDriverHeart_TensorRT_L2_26](/home/pengfei/文档/ML_DL_CV_with_pytorch/Tools/assets/AutoDriverHeart_TensorRT_L2_26.png)

程序端可以调用三种层级的CUDA API，但是基本上只调用相对顶层的API（除去Driver这个很底层的）

## 计算过程

计算过程示意图如下，blockSize为1

![AutoDriverHeart_TensorRT_L2_29](/home/pengfei/文档/ML_DL_CV_with_pytorch/Tools/assets/AutoDriverHeart_TensorRT_L2_29.png)

一个FMA就是一个加法乘法混合运算

![AutoDriverHeart_TensorRT_L2_30](/home/pengfei/文档/ML_DL_CV_with_pytorch/Tools/assets/AutoDriverHeart_TensorRT_L2_30.png)

![AutoDriverHeart_TensorRT_L2_32](/home/pengfei/文档/ML_DL_CV_with_pytorch/Tools/assets/AutoDriverHeart_TensorRT_L2_32.png)

我们可以基于Grid和Block的逻辑进行切分计算，这样可以大大加快计算的效率

![AutoDriverHeart_TensorRT_L2_34](/home/pengfei/文档/ML_DL_CV_with_pytorch/Tools/assets/AutoDriverHeart_TensorRT_L2_34.png)

CUDA中有个规定，就是一个block中可以分配的thread的数量最大是1,024个线程。如果大于1,024会显示配置错误

GPU的warmup：GPU在启动的时候是有一个延时的，会干扰对算法执行时间的测量，所以可以先启动GPU让其完成一点任务，然后再测量

## 程序

定义在GPU上的函数有几种

- `__global__`：定义核函数，在GPU上执行，从CPU端通过三重括号的语法调用，可以有参数，不可以有返回值
- `__device__`：定义设备函数，在GPU上调用也在GPU上执行，不需要三重括号，与普通函数一样，可以有参数，有返回值
- 总的来说，host可以调用global，global可以调用device，device也可以调用device