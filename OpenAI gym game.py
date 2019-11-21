#介绍使用2.0搭建NN，使用纯监督学习（supervised learning）的方法，玩转OpenAI gym game
# OpenAI gym是一个开源的游戏的模拟环境，主要用来开发和比较强化学习（RL）的算法。

#OpenAI gym 初尝试
#使用OpenAI gym 库提供的CartPole Game环境  在本示例中，Action是随机选择的
# import gym
# import random
# import time
# import numpy as np
# env = gym.make("CartPole-v0") #加载游戏环境
# state = env.reset()  #list   状态
# score = 0
# while True:
#     time.sleep(0.1)
#     env.render()#显示画面
#     action = random.randint(0,1)#随机选择一个动作  int  0向左/1向右
#     state,reward,done,_ = env.step(action)#执行这个动作  reward奖励（每走一步得一分）
#     score += reward  #每回合的得分+
#     if done:  #游戏结束  Done 游戏上限是200回合
#         print('score:',score) #打印分数
#         break
# env.close()

#搭建神经网络 目的就是将随机选择的Action部分，变为由神经网络模型来选择。
# 神经网络的输入是State，输出是Action。在这里，Action 用独热编码来表示，
# 即 [1, 0] 表示向左，[0, 1] 表示向右。这样我们可以方便地使用np.argmax()获取预测的 Action 的值。
# np.argmax([0.3, 0.7]) # 1，假如神经网络的输出是 [0.3, 0.7]，那Action值为1，表示向右。
# np.argmax([0.8, 0.2]) # 0，表示向左

#接下来搭建一个4 x 64 x 20 x 2 的网络，输入层为4，输出层为2    （train.py）
import random
import gym
import numpy as np
from tensorflow.keras import layers,models

env = gym.make("CartPole-v0") #加载游戏环境
STATE_DIM,ACTION_DIM = 4,2  #state 维度 4，action 维度 2
model = models.Sequential([
    layers.Dense(64,input_dim=STATE_DIM,activation='relu'),
    layers.Dense(20,activation='relu'),
    layers.Dense(ACTION_DIM,activation='linear')
])
print(model.summary()) #打印神经网络信息

#训练数据从哪里来？  随机产生的数据，得分很低，如果不过滤，数据集质量很低
#最终的办法：试
#简而言之，我们在过程中计算score，如果最终得分达到设定的标准，这个分数所对应的所有State和Action就可以作为我们的训练数据了
def generate_data_one_episode():
    '''生成单次游戏的训练数据'''
    x,y,score = [],[],0
    state = env.reset()
    while True:
        action = random.randrange(0,2)
        x.append(state)
        y.append([1,0] if action==0 else[0,1]) #记录数据
        state,reward,done,_ = env.step(action)#执行动作
        score += reward
        if done:
            break
    return x,y,score
def generate_training_data(expected_score=100):
    '''生成N次游戏的训练数据，并进行筛选，选择>100 的数据作为训练集'''
    data_X,data_Y,scores=[],[],[]
    for i in range(10000):
        x,y,score = generate_data_one_episode()
        if score > expected_score:
            data_X += x
            data_Y += y
            scores.append(score)
    print('dataset size:{}, max score:{}'.format(len(data_X),max(scores)))
    return np.array(data_X),np.array(data_Y)

#训练并保存模型
data_X,data_Y=generate_training_data()
model.compile(loss='mse',optimizer='adam',epochs=5)
model.fit(data_X,data_Y)
model.save('CartPole-v0-nn.h5')#保存模型

#模型测试/预测
import time
import numpy as np 
import gym
from tensorflow.keras import models

saved_model = models.load_model('CartPole-v0-nn.h5')#加载模型
env = gym.make('CartPole-v0')#加载游戏环境
for i in range(s):/
    state = env.reset()
    score = 0
    while True:
        time.sleep(0.01)
        env.render() #显示画面
        action = np.argmax(saved_model.predict(np.array([state]))[0]) #预测动作
        state,reward,done,_=env.step(action)  #执行这个动作
        score+=reward #每回合得分
        if done: #游戏结束
            print('using nn,score:',score)  #打印分数
            break
env.close()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      