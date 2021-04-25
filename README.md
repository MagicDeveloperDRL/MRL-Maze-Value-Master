# MRL-Maze-Value-Master
# AI走方格迷宫

## ——基于Value的RL算法 

本项目是作者（MRL Liu）的一个阶段性总结，重构了之前学习时的一些基于Value算法，将它们的例如经验池的对象等抽象出来为一个对象，便于整理知识网络。

该项目的原始算法代码使用的是莫烦Python的相关实现，在此向莫烦老师表示感谢。

作者对算法的代码就行了整理和重构，该项目目前包含了以下RL算法：

1、基于Q-Table的Q-Learning、Sarsa、Sarsa-Lambda

2、基于ReplayBuffer的Nature DQN、Double DQN

DQN算法系列使用了基于TensorFlow框架训练的全连接网络的作为函数拟合器。

本项目的特色是：

是作者一个阶段总结性项目。基于MNIST的手写数字识别项目已是深度学习入门的必备项目，但区别于其他，本项目的特色是添加了模型的保存与加载功能、TensorFlow可视化功能、指数衰减法的学习率、滑动平均模型技术、L2正则化等常见技术和可视化手写数字图片功能等。整个项目基于良好的面向对象思想，方法定义层层推进，可以说是非常好的总结性学习材料。例如：

![image-20210419215548673](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210419215548673.png)

本项目的所有代码共分为2个模块：

| **步骤** | **模块名称**         | **主要任务**                           |
| -------- | -------------------- | -------------------------------------- |
| 一       | datahelper.py        | 提供可视化读取后的图片文件数据的方法   |
| 二       | Model_Constructor.py | 提供定义模型、训练模型、评估模型的方法 |

# **（1）读取的MNIST图片数据可视化效果**

![image-20210419211623293](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210419211623293.png)

# **（2）训练模型的TensorBoard效果**

定义的计算图结构在TensorBoard中的可视化效果：

![image-20210419212906164](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210419212906164.png)

定义的loss和accuracy在TensorBoard中的可视化效果：

![image-20210419213108485](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210419213108485.png)

![image-20210419213133427](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210419213133427.png)

定义的模型变量在TensorBoard中的可视化效果：

![image-20210419213201903](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210419213201903.png)

# **（3）测试模型的可视化效果**

在随机抽取图片的可视化预测效果：

![image-20210419211812899](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210419211812899.png)

在训练3000次后模型在验证集中可以达到的准确率：0.9838

![image-20210419211831764](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210419211831764.png)

# **（4）本项目开源地址等附加信息**

本项目使用的一些其他参考信息：

| **条目**             | **说明**                                                     |
| -------------------- | ------------------------------------------------------------ |
| 本项目GitHub开源地址 | https://github.com/MagicDeveloperDRL/MRL-Mnist-Number-Master |
| 本项目作者博客地址   | https://blog.csdn.net/qq_41959920/article/details/115875588  |
| 本项目用到的第三方库 | Numpy,TensorFlow1.14.1,matplotlib,                           |
| 主要参考书籍         | 《TensorFlow实战Google深度学习框架》（第2版）                |
| 数据集来源           | http://yann.lecun.com/exdb/mnist（本项目原工程中包含有数据集及保存的训练数据） |

本项目包含的文件目录结构如下：

![image-20210419212503003](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210419212503003.png)

如果读者不想下载多余的网络模型和TesorBoard文件，可以只下载代码datahelper.py和net_model.py和数据集mnist，然后仿照上述目录新建logs和models文件夹即可运行生成新的网络模型和TesorBoard文件。