'''''''''
@file: sarsa_lambda.py
@author: MRL Liu
@time: 2021/4/25 16:17
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
from brain.q_table import Q_Table

class Sarsa_Lambda(object):
    def __init__(self, n_action,
                 epsilon=0.9,  # 贪心系数
                 gamma=0.9,  # 折扣因子
                 learning_rate=0.01  # 学习率
                 ):
        self.action_space = range(n_action)  # 'up', 'down', 'left', 'right'
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lambda_ = 0.9  # trace_decay
        # 初始化Q表
        self.q_table = Q_Table(n_action)
        self.eligibility_trace = Q_Table(n_action)

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

    "学习函数，值迭代"
    def update(self, s, a, r, s_,a_):
        # 获取Q预测
        q_predict = self.q_table.get_q_value(s, a)
        # 计算Q目标
        if s_ != 'terminal':
            next_q_predict = self.q_table.get_q_value(s_, a_)
            q_target = r + self.gamma * next_q_predict
        else:
            q_target = r
        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1
        # Method 2:
        self.eligibility_trace.clear_s_q_value(s)
        self.eligibility_trace.update_q_value(s,a,1)

        # 更新Q表
        self.q_table += self.learning_rate * (q_target - q_predict) * self.eligibility_trace
        # self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)
        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_

