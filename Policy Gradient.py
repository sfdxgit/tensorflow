#强化学习   实战策略梯度算法（Policy Gradient）
# DQN的动作更确定，因为DQN每次总是选择Q值最大的动作，而Policy Gradient按照概率选择，会产生更多的不确定性
# policy_gradient.py
import matplotlib.pyplot as plt 
import gym
import numpy as np 
from tensorflow.keras import models,layers,optimizers

env = gym.make('CartPole-v0')
STATE_DIM,ACTION_DIM = 4,2  #输入层=4，输出层=2
model = models.Sequential([
    layers.Dense(100,input_dim=STATE_DIM,activation='relu'),
    layers.Dropout(0.1),#随机失活 10% 一开始容易陷入局部最优和过拟合，Dropout可以有效避免
    layers.Dense(ACTION_DIM,activation='softmax')
])
model.compile(loss='mean_squared_error',
    optimizer=optimizers.Adam(0.001))
def choose_action(s):
    '''预测动作'''
    prob=model.predict(np.array([s]))[0]
    return np.random.choice(len(prob),p=prob)
#优化策略
#带衰减reward的累加期望  discount_reward[i] = reward[i] + gamma * discount_reward[i+1]
#越是前面的步骤，累加期望越高
def discount_rewards(rewards,gamma=0.95):
    '''计算衰减reward的累加期望，并中心化和标准化处理'''
    prior=0
    out=np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        prior = prior*gamma+rewards[i]
        out[i] = prior
    return out/np.std(out-np.mean(out))
#给loss加权重   一个动作的累加期望很高，自然希望该动作出现的概率变大，这就是学习的目的
#一般，通过构造标签（y_true/label），来训练神经网络；还可以通过改变损失函数达到目的，对于累加期望大的动作，可以放大loss的值
#可以给损失函数加一个权重 loss = discount_reward * loss   discount_reward可以理解为策略梯度算法中的梯度
def train(records):
    s_batch = np.array([record[0] for record in records])
    #action 独热编码处理，方便求动作概率，即prob_batch
    a_batch = np.array([[1 if record[1] == i else 0 for i in range(ACTION_DIM)] for record in records])
    #假设predict的概率是[0.3,0.7],选择的动作是[0,1]
    #则动作[0,1]的概率等于[0.3,0.7] = [0.3,0.7]*[0,1]
    prob_batch = model.predict(s_batch)*a_batch
    r_batch = discount_rewards([record[2] for record in records])
    model.fit(s_batch,prob_batch,sample_weight = r_batch,verbose = 0)
#训练过程与结果
episodes = 2000 #至多2000次
score_list= [] #记录所有分数
for i in range(episodes):
    s = env.reset()
    score = 0
    replay_records = []
    while True:
        a = choose_action(s)
        next_s,r,done,_ = env.step(a)
        replay_records.append((s,a,r))
        score += r
        s = next_s
        if done:
            train(replay_records)
            score_list.append(score)
            print('episode:',i,'score:',score,'max:',max(score_list))
            break
    #最后10次的平均分大于195时，停止并保存模型
    if np.mean(score_list[-10:])>195:
        model.save('CartPole-v0-pg.h5')
        break
env.close()

#画一张图，多了三行多项式拟合代码，能够更好地展现整个分数的变化趋势
plt.plot(score_list)
x=np.array(range(len(score_list)))
smooth_func=np.poly1d(np.polyfit(x,score_list,3))
plt.plot(x,smooth_func(x),label = 'Mean',linestyle='--')
plt.show()
# 测试  test_policy_gradient.py
import time
import numpy as np
import gym
from tensorflow.keras import models
saved_model = models.load_model('CartPole-v0-pg.h5')
env = gym.make('CartPole-v0')
for i in range(5):
    s = env.reset()
    score = 0
    while True:
        time.sleep(0.01)
        env.render()
        prob = saved_model.predict(np.array([s]))[0]
        a = np.random.choice(len(prob),p = prob)
        s,r,done,_ = env.step(a)
        score += r
        if done:
            print('using policy gradient,score:',score) #打印分数
            break
env.close()   