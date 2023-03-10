# 一、下载clash

clash是一个平台，可以提供科学上网，当然这需要你有订阅链接，有点类似于迅雷——你有一个下载链接，需要迅雷提供下载，链接是独特的，迅雷是通用的

https://github.com/Dreamacro/clash/releases

![2023-01-30 08-26-16 的屏幕截图](./image/2023-01-30 08-26-16 的屏幕截图.png)

选择Linux-amd64的即可，v1/v3貌似没有区别

下载文件是一个.gz格式压缩包，解压之后就是一个可执行文件，我们单独创建一个clash文件夹存放这个可执行文件，并且将这个可执行文件改名为clash

![](./image/2023-01-30 08-30-56 的屏幕截图.png)

# 二、下载配置文件

在clash文件夹下，打开终端，下载配置文件

```shell
sudo wget -O config.yaml [订阅链接]
sudo wget -O Country.mmdb https://www.sub-speeder.com/client-download/Country.mmdb
```

我们用自己的订阅链接替换掉第一行的[订阅链接]，就像这样

![](/run/user/1000/doc/9a0cb1c0/2023-01-30 08-36-34 的屏幕截图.png)

然后回车，就可以根据订阅链接下载配置文件了，可以发现文件夹下多了一个config.yaml文件

然后输入第二个指令，下载第二个配置文件，如果这里下载不成功，可以删除文件夹下的Country.mmdb文件，在后面加载程序的时候会自动下载

然后打开config.yaml文件，找到secret，这里表示密钥，初试状态为“”，将其改为“123456”（后面要用，也可以是其他的密码，这里为了方便记忆）

![](/run/user/1000/doc/5b2aaaaf/2023-01-30 08-41-05 的屏幕截图.png)

# 三、启动clash

在clash文件夹中启动终端，输入

```shell
chmod +x clash
./clash -d .
```

第一个指令是赋予clash文件可执行权限，第二个是启动clash文件并且告诉它，它所需的配置文件在当前目录下，这里注意一下./clash -d后还有一个点