# MRL-Maze-Value-Master
# 使用多种AI算法玩方格迷宫

## ——基于Value的RL算法 

# 前言

本项目是作者（MRL Liu）使用AI算法的强化学习方法玩迷宫游戏的一个阶段性总结，本项目的迷宫游戏是简单的方格迷宫，其状态空间和动作空间都足够简单，是作者整理的手中的第1个RL项目。

该项目重构了作者之前学习时的一些基于Value的RL算法，将它们的例如经验回放池的对象等抽象出来为一个对象，便于整理知识网络。该项目的原始算法代码使用的是莫烦Python的相关实现，在此向莫烦老师表示感谢。

本项目的特色是：

1、使用了统一范式的代码来定义基于Value的算法系列的实现，封装了Q-Table和ReplayBuffer对象

2、添加了网络模型的保存与加载功能、TensorFlow可视化功能、经验池保存和加载等。

3、整个项目基于良好的面向对象思想，方法定义层层推进。

本项目自定义的2个方格环境如下：

| line_env                                                     | maze_env                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210425222905119.png) | ![image-20210426135130790](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210426135130790.png) |

作者对算法的代码就行了整理和重构，该项目目前包含了以下两类RL算法：

1、基于Q-Table的Q-Learning、Sarsa、Sarsa-Lambda

2、基于ReplayBuffer的Nature DQN、double DQN和dueling DQN等

DQN算法系列使用了基于TensorFlow框架训练的全连接网络的作为函数拟合器。

本项目的主要运行代码共分为3个模块：

| **模块** | **模块名称** | **主要任务**                             |
| -------- | ------------ | ---------------------------------------- |
| 一       | main.py      | Maze项目的启动器，负责切换不同的方格环境 |
| 二       | trainer.py   | Maze项目的训练器，负责切换不同的训练流程 |
| 二       | brain.py     | Maze项目的AI大脑，负责切换不同的决策算法 |

# **（1）自定义数据可视化训练变化**

![image-20210426162057305](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210426162057305.png)

# **（2）训练模型的TensorBoard效果**

定义的计算图结构在TensorBoard中的可视化效果（Nature DQN算法）：

![image-20210426161658841](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210426161658841.png)

定义的loss在TensorBoard中的可视化效果：

![image-20210426162256336](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210426162256336.png)

定义的模型变量在TensorBoard中的可视化效果：

![image-20210426162345606](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210426162345606.png)

# **（3）训练过程的打印日志**

![image-20210426162510208](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210426162510208.png)

![image-20210426162545878](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210426162545878.png)

# **（4）本项目开源地址等附加信息**

本项目使用的一些其他参考信息：

| **条目**             | **说明**                                                    |
| -------------------- | ----------------------------------------------------------- |
| 本项目GitHub开源地址 | https://github.com/MagicDeveloperDRL/MRL-Maze-Value-Master  |
| 本项目作者博客地址   | https://blog.csdn.net/qq_41959920/article/details/115875588 |
| 本项目用到的第三方库 | Numpy,TensorFlow1.14.1,matplotlib,                          |
| 主要参考教程         | https://morvanzhou.github.io/tutorials/                     |

本项目包含的文件目录结构如下：

![image-20210426161839741](C:\Users\Lab\AppData\Roaming\Typora\typora-user-images\image-20210426161839741.png)

该项目中包含的AI算法的实现流程将会逐渐推出博客进行解析，若觉得对读者有帮助，欢迎继续关注。