# 介绍

主要是介绍从基础到实践中各种数据集和数据格式以及使用方法

最基础的数据集格式就是文本分类数据集，我们可以在一个txt文本中保存以下内容：

```
hello 1
bad 0
sad 0
best 1
```

然后就构成了一个非常简单的文本分类数据集文件了，接下来就可以读取这个文件

```python
f=open("./data.txt",encoding="utf-8")
all_d=f.read()
```

注意一下不要写成转义符`\n`这种

上述代码可以读取txt文件并且转化为字符串格式，得到所有的数据也就是`all_d`变量

```python
f.readline()
f.readlines()
```

这样可以读取文本内容，前者可以按行读取，如果读完文件就会返回空字符串（可以使用`f.seek()`跳转读取位置或者说行指针，这个适合超大数据集文件的处理，不需要一次性处理完）；后者会把所有的内容一次性读取并且返回一个列表

当然，上面可以得到一行行的字符串，但是可以进一步细分，得到文本和标签两个部分，比如说：

```python
a="hello 1"
text,label=a.split(" ")
```

对于包含空格的文本行，可以使用`split`分割（当然也可以使用其他的标识符分割）

# Debug

点击红点可以添加断点，调试的时候会停止在断点之前

然后可以查看中间状态和变量，并且执行一些代码（手动额外输入）

这样子可以通过调试，找到一些可能存在的问题

然后就是try-except方法，可以在安全情况下进行尝试并且捕获异常

```python
try:   # 尝试以下代码，如果出错了，就执行 except 中的代码， 如果不出错，就执行 else 中的代码
    label = int(label)  # 类型转化
    if label not in ground_true:
        # continue
        raise Exception#报一个什么样的错误
        '''raise [exceptionName [(reason)]]
        可以使用字符串描述原因
        TypeError 类型错'''
except :
    print("标签报错啦！")
else:
    all_text.append(text)
    all_label.append(label)
```

主函数：程序的入口

在主函数之前的就是全局变量