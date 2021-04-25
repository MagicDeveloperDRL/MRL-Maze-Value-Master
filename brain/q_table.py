'''''''''
@file: q_table.py
@author: MRL Liu
@time: 2021/4/25 14:54
@env: Python,Numpy
@desc: 基于pd.DataFrame的Q-Table对象
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import pandas as pd

class Q_Table(object):
    def __init__(self,n_action):
        """创建一个空的Q_table"""
        self.n_action=n_action
        self.q_table = pd.DataFrame(
            data=None,  # q_table initial values
            index=None,  # 行名为空
            columns=range(n_action),  # 列名为actions's name
            dtype=np.float64
        )
        #print(self.q_table)    # show table

    def get_actions(self,state):
        """获取某个状态的所有动作"""
        self.check_state_exist(state)# 检查是否需要添加该状态
        state_actions = self.q_table.loc[state, :]  # 获取当前状态可采取的动作及其价值
        return state_actions

    def get_q_value(self,state,action):
        """获取某个状态-动作的Q值"""
        self.check_state_exist(state)  # 检查是否需要添加该状态
        get_q_value = self.q_table.loc[state, action] # 获取Q表中对应的（s,a）的值
        return get_q_value
    def update_q_value(self,state,action,value):
        """更新某个状态-动作的Q值"""
        self.check_state_exist(state)  # 检查是否需要添加该状态
        self.q_table.loc[state, action]=value # 获取Q表中对应的（s,a）的值
    def clear_s_q_value(self,state):
        """清空某个状态的所有动作的Q值"""
        self.check_state_exist(state)  # 检查是否需要添加该状态
        self.q_table.loc[state, :]*=0 # 获取Q表中对应的（s,a）的值
    def check_state_exist(self, state):
        """检查状态是否存在"""
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*self.n_action,
                    index=self.q_table.columns,
                    name=state,
                )
            )

if __name__=='__main__':
    # 创建一个空的Q_table
    q_table = Q_Table(3)
    print(q_table.q_table)
    # 获取某个状态的所有动作
    actions = q_table.get_actions(0)
    print(q_table.q_table)
    print(actions)
    # 更新某个状态-动作的Q值
    q_table.update_q_value(5, 1, 0.6)
    print(q_table.q_table)
    # 获取某个状态-动作的Q值
    q_value = q_table.get_q_value(5,1)
    print(q_table.q_table)
    print(q_value)
    # 清空某个状态的所有动作的Q值
    q_table.clear_s_q_value(5)
    print(q_table.q_table)
