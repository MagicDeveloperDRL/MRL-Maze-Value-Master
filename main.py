'''''''''
@file: main.py
@author: MRL Liu
@time: 2021/4/25 14:17
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
from env.Line_Env import Line_Env
#from env.Maze_Env import Maze_Env
from env.maze_env import Maze_Env
from brain.brain import Brain
from trainer import Trainer



def run_line():
    env = Line_Env()  # 创建环境
    agent = Brain(env.n_features,env.n_actions)  # 创建agent
    trainer = Trainer(env, agent)  # 创建训练器
    # 训练agent模型
    trainer.train_q_learning(max_episodes=10)
    # 绘制检测数据
    trainer.draw_plot()


def run_maze():
    env = Maze_Env()  # 创建环境
    agent = Brain(env.n_features,env.n_actions)
    trainer = Trainer(env, agent)  # 创建训练器
    #trainer = Maze_Trainer_Sarsa(env, agent)
    # 训练agent模型
    # env.after(100, trainer.train(max_episodes=10))  # 在窗口主循环中添加方法
    #trainer.train_q_learning(max_episodes=15)
    trainer.train_dqn(max_episodes=15)
    # 绘制检测数据
    trainer.draw_plot()
    env.mainloop()  # 调用主循环显示窗口


if __name__ == '__main__':
    run_maze()
    #run_line()