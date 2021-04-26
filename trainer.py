'''''''''
@file: trainer.py
@author: MRL Liu
@time: 2021/2/15 15:43
@env: Python,Numpy
@desc: Maze项目的训练器，负责切换不同的训练流程
@ref: 
@blog: https://blog.csdn.net/qq_41959920
'''''''''
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #使用中文字符
plt.rcParams['axes.unicode_minus'] = False #显示负数的负号



class Trainer(object):
    def __init__(self,env,agent):
        self.env = env
        self.agent = agent
        # 每回合最大步数记录器
        self.list_step_train = []
        self.list_reward_train = []

    def train_q_learning(self,max_episodes):
        print('仿真训练任务启动...')
        # 训练主循环
        for episode in range(1,max_episodes+1):
            step_counter = 0
            reward_episode = 0
            # 获取初始环境状态
            observation = self.env.reset()
            # 开始本回合的仿真
            while True:
                self.env.render()
                # 获取动作和环境反馈
                action = self.agent.choose_action(observation)# agent根据当前状态采取动作
                observation_, reward, done = self.env.step(observation,action)# env根据动作做出反馈
                # 学习本回合的经验(s, a, r, s)
                #reward-=step_counter*0.01
                self.agent.update(observation, action, reward, observation_,done)
                # 更新
                observation = observation_
                step_counter += 1
                reward_episode += reward
                # 检测本回合是否需要停止
                if done:
                    self.list_reward_train.append(reward_episode)  # 记录最大回合奖励
                    self.list_step_train.append(step_counter)  # 记录最大回合数
                    print('episode：{} ,step：{},reward_episode：{}'.format(episode, step_counter,reward_episode))
                    break
        print('仿真训练任务结束')

    def train_sarsa(self,max_episodes):
        print('仿真训练任务启动...')
        # 训练主循环
        for episode in range(1,max_episodes+1):
            step_counter = 0
            reward_episode = 0
            # 获取初始环境状态
            observation = self.env.reset()
            action = self.agent.choose_action(observation)  # agent根据当前状态采取动作
            # 开始本回合的仿真
            while True:
                self.env.render()
                # 获取动作和环境反馈
                observation_, reward, done = self.env.step(observation,action)# env根据动作做出反馈
                action_ = self.agent.choose_action(observation_)  # agent根据当前状态采取动作
                # 学习本回合的经验(s, a, r, s)
                #reward-=step_counter*0.01
                self.agent.update(observation, action, reward, observation_,action_,done)
                # 更新
                observation = observation_
                action = action_
                step_counter += 1
                reward_episode += reward
                # 检测本回合是否需要停止
                if done:
                    self.list_reward_train.append(reward_episode)  # 记录最大回合奖励
                    self.list_step_train.append(step_counter)  # 记录最大回合数
                    print('episode：{} ,step：{},reward_episode：{}'.format(episode, step_counter,reward_episode))
                    break
        print('仿真训练任务结束')

    def draw_plot(self):
        # 创建画布
        fig = plt.figure(figsize=(6, 3))  # 创建一个指定大小的画布
        # 创建画布
        print('绘制数据')
        # 添加第1个窗口
        ax1 = fig.add_subplot(121)  # 添加一个1行1列的序号为1的窗口
        # 添加标注
        ax1.set_title('训练中的累计步数变化状况', fontsize=14)  # 设置标题
        ax1.set_xlabel('x-回合数', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
        ax1.set_ylabel('y', fontsize=14, fontstyle='oblique')
        # 绘制函数
        x_data_train = range(1,len(self.list_step_train)+1)
        y_data_train = self.list_step_train
        line1, = ax1.plot(x_data_train, y_data_train, color='blue', label="训练值")
        ax1.legend(handles=[line1], loc=1)  # 绘制图例说明
        plt.grid(True)  # 启用表格
        # 添加第1个窗口
        ax1 = fig.add_subplot(122)  # 添加一个1行1列的序号为1的窗口
        # 添加标注
        ax1.set_title('训练中的累计奖励变化状况', fontsize=14)  # 设置标题
        ax1.set_xlabel('x-回合数', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
        ax1.set_ylabel('y', fontsize=14, fontstyle='oblique')
        # 绘制函数
        x_data_train = range(1, len(self.list_reward_train) + 1)
        y_data_train = self.list_reward_train
        line1, = ax1.plot(x_data_train, y_data_train, color='blue', label="训练值")
        ax1.legend(handles=[line1], loc=1)  # 绘制图例说明
        plt.grid(True)  # 启用表格














