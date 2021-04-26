'''''''''
@file: double_dqn.py
@author: MRL Liu
@time: 2021/4/26 16:43
@env: Python,Numpy
@desc: 名为double DQN的AI算法
@ref: https://morvanzhou.github.io/tutorials/
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import os
import  numpy as np
import tensorflow as tf
from .replaybuffer import ReplayBuffer

np.random.seed(1)
tf.set_random_seed(1)

MODEL_SAVE_PATH='./models/'
LOGS_SAVE_PATH='./logs/'
MODEL_NAME='model.ckpt'

class Double_DQN(object):
    # 初始化参数
    def __init__(
            self,
            n_features,# 观测值个数
            n_actions, #动作个数
            learning_rate=0.01,# 学习率
            e_greedy=0.9,# e-greedy
            e_greedy_increment=None, #是否让greedy变化
            memory_size = 100, # 记忆池的行数据大小
            batch_size=16,  # 每次采样数据的大小
            replace_target_iter=300,
            gamma=0.9, # 回报折扣因子
            output_graph = False, # 是否输出TensorBoard
            save_model=False,
            read_saved_model=False,
            sess=None,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        # 贪婪值
        self.epsilon_increment = e_greedy_increment
        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # 更新相关参数
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_rate =learning_rate
        self.gamma = gamma
        self.replace_target_iter=replace_target_iter
        # 相关设置
        self.output_graph = output_graph
        self.save_model = save_model
        self.read_saved_model = read_saved_model
        # 保存路径
        self.model_save_path = MODEL_SAVE_PATH
        self.logs_save_path = LOGS_SAVE_PATH
        self.model_name = MODEL_NAME
        # 初始化经验池
        self.memory = ReplayBuffer(memory_size, self.n_features)
        # 定义计算图
        self.define_graph(sess)
        # 存储损失
        self.cost_his = []

    def choose_action(self,observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # 贪婪地选择动作
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:  # 随机地选择动作
            action = np.random.randint(0, self.n_actions)
        return action
    def update(self, s, a, r, s_,done):
        self.store_in_memory(s, a, r, s_)
        # 以固定频率进行学习本回合的经验(s, a, r, s)
        if (self.step_counter > self.memory_size) and (self.step_counter % 5 == 0):
            self.learn(done)
        elif self.step_counter% 20 == 0:  # 指定目录下打印消息
            print("已经收集{}条数据".format(self.memory.pointer))
            self.memory.save_memory_json()  # 保存数据
        self.step_counter+=1
    def learn(self,done):
        # 检查是否复制参数给target_net
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_net的参数被更新\n')
        # 获取批次数据
        batch_memory = self.memory.sample(batch_size=self.batch_size)
        # 获取目标Q值
        q_next, q_eval4next = self.sess.run(
            fetches=[self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # next observation
                self.s: batch_memory[:, -self.n_features:]  # next observation
            })
        # 获取当前Q值
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        # 获取采样数据的索引，要修改的矩阵的行
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        # 获取评估的动作的索引，要修改的矩阵的列
        action = batch_memory[:,self.n_features].astype(int)
        # 获取要修改Q值的立即回报
        reward = batch_memory[:, self.n_features + 1]

        # 计算Q现实值，只修改矩阵中对应状态动作的Q值
        if done is not True:
            # DDQN的计算方法
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
            # Nature DQN的计算方法
            # selected_q_next = np.max(q_next,axis=1)
            q_target[batch_index,action] = reward+self.gamma*selected_q_next
        else:
            q_target[batch_index, action] = reward
        # 更新评估网络并获取其训练操作
        if self.save_model and self.learn_step_counter % 100 == 0:  # 定期保存cnpk模型
            self.saver.save(self.sess, os.path.join(self.model_save_path, self.model_name),
                            global_step=self.learn_step_counter)
            # 执行优化器、损失值和step
            _, cost = self.sess.run([self._train_op, self.loss],
                                          feed_dict={self.s: batch_memory[:, :self.n_features],
                                                     self.s_: batch_memory[:, -self.n_features:],
                                                     self.q_target: q_target})
            print('learn_step: %d , loss: %g. and save model successfully' % (self.learn_step_counter, cost))
            self.cost_his.append(cost)
        elif self.output_graph and self.learn_step_counter % 10 == 0:
            _, cost, summary = self.sess.run([self._train_op, self.loss, self.merged_summary_op],
                                             feed_dict={self.s: batch_memory[:, :self.n_features],
                                                        self.s_: batch_memory[:, -self.n_features:],
                                                        self.q_target: q_target})
            self.train_writer.add_summary(summary, self.learn_step_counter)  # 添加日志
            print('learn_step: %d , loss: %g. ' % (self.learn_step_counter, cost))
            self.cost_his.append(cost)
        elif self.learn_step_counter % 10 == 0:
            _, cost = self.sess.run([self._train_op, self.loss],
                                             feed_dict={self.s: batch_memory[:, :self.n_features],
                                                        self.s_: batch_memory[:, -self.n_features:],
                                                        self.q_target: q_target})
            print('learn_step: %d , loss: %g. ' % (self.learn_step_counter, cost))
            self.cost_his.append(cost)
        # 逐步提高的利用概率,让算法尽快收敛的编程技巧
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return



    # 向记忆池存入数据
    def store_in_memory(self,s,a,r,s_):
        self.memory.store_transition(s, a, r, s_)


    def define_graph(self,sess):
        # 定义目标网络的输入输出
        self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            with tf.variable_scope('output'):
                self.q_next = self._define_fc_net(self.s_, n_Layer=20, c_names=['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES])
                #tf.summary.scalar("q_next", self.q_next)  # 使用TensorBoard监测该变量
        # 定义评估网络的输入输出
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s')
        self.q_target = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            with tf.variable_scope('output'):
                self.q_eval = self._define_fc_net(self.s, n_Layer=20,c_names=['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES])
                #tf.summary.scalar("q_eval", self.q_eval)  # 使用TensorBoard监测该变量
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
                tf.summary.scalar("loss", self.loss)  # 使用TensorBoard监测该变量
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.learn_rate).minimize(self.loss)
        # 定时复制参数给target_net
        self.learn_step_counter = 0
        self.step_counter = 0
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.merged_summary_op = tf.summary.merge_all()  # 合并所有的summary为一个操作节点，方便运行
        # 初始化会话
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        # 是否保存模型：
        if self.save_model:
            self.saver = tf.train.Saver()  # 网络模型保存器
        else:
            self.saver = None
        if self.read_saved_model:
            ckpt = tf.train.get_checkpoint_state(os.path.join(self.model_save_path,""))  # 获取ckpt的模型文件的路径
            print(os.path.join(self.model_save_path,""))
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)  # 恢复模型参数
                strNum = ckpt.model_checkpoint_path.split(',')[-1].split('-')[-1]
                self.learn_step_counter = int(strNum)
                print('成功读取指定模型：'+ckpt.model_checkpoint_path)
            else:
                print('无法找到指定的checkpoint文件')
        # 是否输出图
        if self.output_graph:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.logs_save_path, ""), self.sess.graph)
        else:
            self.train_writer = None
        # 初始化所有变量
        self.sess.run(tf.global_variables_initializer())

    def _define_fc_net(self,input,n_Layer,c_names):
        layer1 = self._define_fc_layer(input,self.n_features,n_Layer,layer_name='layer1',activation_function=tf.nn.relu,c_names = c_names)
        out = self._define_fc_layer(layer1, n_Layer, self.n_actions, layer_name='out', activation_function=None,c_names=c_names)

        return out
    def _define_fc_layer(self,inputs,in_size, out_size, layer_name, activation_function=None,c_names=None,regularizer__function=None,is_historgram=True):
        """ 定义一个全连接神经层"""
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('weights'):
                weights = tf.get_variable('w', [in_size, out_size],initializer=tf.random_normal_initializer(0., 0.3),collections=c_names)
                if regularizer__function != None:  # 是否使用正则化项
                    tf.add_to_collection('losses', regularizer__function(weights))  # 将正则项添加到一个名为'losses'的列表中
                if is_historgram:  # 是否记录该变量用于TensorBoard中显示
                    tf.summary.histogram(layer_name + '/weights', weights)  # 第一个参数是图表的名称，第二个参数是图表要记录的变量
            with tf.variable_scope('biases'):
                biases = tf.get_variable('b', [1, out_size], initializer=tf.constant_initializer(0.1),collections=c_names)
                if is_historgram:  # 是否记录该变量用于TensorBoard中显示
                    tf.summary.histogram(layer_name + '/biases', biases)
            with tf.variable_scope('wx_plus_b'):
                # 神经元未激活的值，矩阵乘法
                wx_plus_b = tf.matmul(inputs, weights) + biases
            # 使用激活函数进行激活
            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)
            if is_historgram:  # 是否记录该变量用于TensorBoard中显示
                tf.summary.histogram(layer_name + '/outputs', outputs)
            # 返回神经层的输出
        return outputs

    # 显示图
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
