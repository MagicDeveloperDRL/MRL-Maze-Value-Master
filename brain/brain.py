'''''''''
@file: brain.py
@author: MRL Liu
@time: 2021/4/25 14:16
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''

from brain.q_learning import Q_Learning
from brain.sarsa import Sarsa
from brain.sarsa_lambda import Sarsa_Lambda
from brain.dqn import DQN
class Brain(object):
    def __init__(self,n_features,n_action):
        # 初始化智能体
        #self.agent = Q_Learning(n_action)
        #self.agent = Sarsa(n_action)
        #self.agent = Sarsa_Lambda(n_action)
        self.agent = DQN(n_features,n_action)


    def choose_action(self, state):
        action = self.agent.choose_action(state)
        #print(action)
        return action


    def update(self, *args):
        self.agent.update(*args)



