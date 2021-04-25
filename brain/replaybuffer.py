'''''''''
@file: ReplayBuffer.py
@author: MRL Liu
@time: 2021/4/20 17:08
@env: Python,Numpy
@desc: 经验回访池
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import json
import os
import time
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity,state_dims):
        self.capacity = capacity # 经验池容量大小
        self.data = np.zeros((capacity, state_dims* 2+2))  # 经验池存放的经验数据
        self.pointer = 0 # 当前指针

    def store_transition(self, s, a, r, s_):
        # 检查是否存在
        if not hasattr(self, 'pointer'):
            self.pointer = 0
        # 存储数据
        transition = np.hstack((s, [a,r], s_))  # 按行连接
        index = self.pointer % self.capacity  # 如果超过该容量则自动从头开始
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, batch_size):
        if self.capacity < self.pointer:
            batch_indexs = np.random.choice(self.capacity, size=batch_size)
        else:
            batch_indexs = np.random.choice(self.pointer, size=batch_size)
            #assert (self.pointer >= self.capacity, '经验回放池还没有被装满')
            print('经验回放池还没有被装满就开始采样')
        return self.data[batch_indexs, :]  # 获取n个采样

    def save_memory_json(self, filename='./data/memory.json'):
        """
        # 保存训练检测数据
        :param filename: '../config/record.json'
        :return:
        """
        # 检查是否存在文件夹
        if not os.path.exists("data"):
            os.makedirs("data")
        # 记录下保存的时间
        save_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        data = {"time:": save_time, "trans:": [trans.tolist() for trans in self.data]}
        # 打开文件
        f = open(filename, "w")
        json.dump(data, f)  # 将Python数据结构编码为JSON格式并且保存至文件中
        f.close()  # 关闭文件
        print("训练检测数据成功保存至{}文件".format(filename))

    def load_memory_json(self, filename):
        """
        # 读取训练检测数据
        """
        f = open(filename, "r")
        data = json.load(f)  # 将文件中的JSON格式解码为Python数据结构
        f.close()
        self.data = [np.array(trans) for trans in data["trans"]]

        print("训练检测数据已成功读取到经验池中...")
        return