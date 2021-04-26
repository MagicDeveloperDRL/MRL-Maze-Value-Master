'''''''''
@file: maze_env.py
@author: MRL Liu
@time: 2021/2/15 15:42
@env: Python,Numpy
@desc: 二维的方格环境
@ref: https://morvanzhou.github.io/tutorials/
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 80   # 单位长度
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

class Maze_Env(tk.Tk,object):
    def __init__(self):
        super(Maze_Env,self).__init__() #继承自tk.TK
        # 初始化动作空间
        self.action_space = ['up','down','left','right']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        # 初始化迷宫配置
        self._init_maze()


    def _init_maze(self):
        """初始化迷宫配置"""
        self.title('走迷宫项目')  # 设置窗口标题
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))  # 设置窗口大小
        self.origin = np.array([UNIT / 2, UNIT / 2])  # 起点位置的二维坐标
        # 创建画布
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        # 使用画布绘制网格
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        # 使用画布绘制方块Agent(self.rect )
        self.rect =self._create_rectangle(self.origin,'red')
        # 使用画布绘制方块陷阱1
        hell1_center = self.origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self._create_rectangle(hell1_center,'black')
        # 使用画布绘制方块陷阱2
        hell2_center = self.origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self._create_rectangle(hell2_center,'black')

        # 使用画布绘制方块终点(self.oval)
        oval_center = self.origin + UNIT * 2
        self.oval = self._create_rectangle(oval_center,'yellow')

        self.canvas.pack()  # 放置以上所有组件，否则不显示

    def _create_rectangle(self,center,color):
        """使用画布创建方块"""
        rect = self.canvas.create_rectangle(
            center[0] - 15, center[1] - 15,
            center[0] + 15, center[1] + 15,
            fill=color)
        return rect

    def reset(self):
        """重置环境，返回初始状态"""
        self.update() # 更换窗口
        time.sleep(0.5)
        self.canvas.delete(self.rect) # 删除Agent
        self.rect = self._create_rectangle(self.origin,'red')
        #s = self.canvas.coords(self.rect) # 获取方格的左上角坐标和右下角坐标
        s = (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)
        #print(s)
        return s

    def step(self,state,action):
        """更新环境，输入动作，返回输出位置"""
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0]) # 移动的举例
        if action == 0:  # 向上移动一个单位长度
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 向下移动一个单位长度
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 向右移动一个单位长度
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # 向左移动一个单位长度
            if s[0] > UNIT:
                base_action[0] -= UNIT
        else:
            print('无法识别的动作')

        self.canvas.move(self.rect, base_action[0], base_action[1])  # 移动 agent

        s_ = self.canvas.coords(self.rect)  # 获取此时的状态为后继状态
        # 奖励函数
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            #s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            #s_ = 'terminal'
        else:
            reward = 0
            done = False
        s_ = (np.array(s_[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)
        #print(s_)
        return s_, reward, done

    def render(self):
        """更新窗口"""
        time.sleep(0.1)
        self.update()

def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 2
            s, r, done = env.step(s,a)
            if done:
                break

if __name__ == '__main__':
    env = Maze_Env() # 创建环境
    env.reset()
    env.after(100, update) # 在窗口主循环中添加方法
    env.mainloop() # 调用主循环显示窗口