'''''''''
@file: q_learning.py
@author: MRL Liu
@time: 2021/4/25 15:30
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
from brain.q_table import Q_Table


class Q_Learning(object):
    def __init__(self,n_action ,
                 epsilon=0.9,# 贪心系数
                 gamma=0.9,# 折扣因子
                 learning_rate=0.01 # 学习率
                 ):
        self.action_space = range(n_action)  # 'up', 'down', 'left', 'right'
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        # 初始化Q表
        self.q_table = Q_Table(n_action)

    "策略函数，返回采取的动作"
    def choose_action(self, state):
        # 有1-epsilon的概率随机选择动作
        if np.random.uniform() > self.epsilon:  # act non-greedy or state-action have no value
            action = np.random.choice(self.action_space)
        # 有epsilon的概率贪心选取回报值最多的动作时
        else:
            state_actions = self.q_table.get_actions(state)  # 获取当前状态可采取的动作及其价值
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        return action

    "更新函数，值迭代"
    def update(self, s, a, r, s_,done):
        # 获取Q预测
        q_predict = self.q_table.get_q_value(s, a)
        # 计算Q目标
        if done != True:
            actions = self.q_table.get_actions(s_)
            q_target = r + self.gamma * actions.max()
        else:
            q_target = r
        # 更新Q表
        q_predict += self.learning_rate * (q_target - q_predict)
        self.q_table.update_q_value(s, a, q_predict)