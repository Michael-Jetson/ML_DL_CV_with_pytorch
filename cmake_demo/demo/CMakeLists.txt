#CMakeLists.txt
# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)

project(Hello)#先声明项目的名称为HELLO

MESSAGE(STATUS "This is BINARY dir " ${HELLO_BINARY_DIR})
#在命令行输出信息

MESSAGE(STATUS "This is SOURCE dir "${HELLO_SOURCE_DIR})

ADD_EXECUTABLE(hello main.cpp)#使用指定的源文件创建出一个可执行文件