#cmake最低版本需求，不加入此行会受到警告信息
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(HELLO) #项目名称
MESSAGE(STATUS "This is my first cmake project")
#把当前目录(.)下所有源代码文件和头文件加入变量SRC_LIST
AUX_SOURCE_DIRECTORY(. SRC_LIST)
#添加头文件路径
include_directories(./include)
#指明本项目包含一个子目录 src
ADD_SUBDIRECTORY(./src)
#生成应用程序 hello
ADD_EXECUTABLE(hello ${SRC_LIST})
#添加链接库
TARGET_LINK_LIBRARIES(hello test_h1)