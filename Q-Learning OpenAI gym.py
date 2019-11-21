#强化学习RL 的经典算法Q-Learning，玩转OpenAI gym game  Q-Learning的目的是创建一张Q-Table
#初始化一张Q表    q_learning.py
import pickle #保存模型用
from collections import defaultdict
import gym
import numpy as np 

#默认将Action 0，1，2的价值初始化为0
Q=defaultdict(lambda: [0,0,0])

#连续状态映射：Q-Table用字典表是，State中的值是浮点数，是连续的，意味着有无数种状态，这样更新Q-Table的值是不可能实现的。
#所以需要对State进行线性转换，归一化处理。将State中的值映射到[0，40]的空间中，这样，就将无数种状态映射到40*40种状态了。
env=gym.make("MountainCar-v0")
def transform_state(state):
    '''将position、velocity 通过线性转换映射到[0,40]范围内'''
    pos,v=state
    pos_low,v_low = env.observation_space.low 
    pos_high,v_high = env.observation_space.high 
    a = 40*(pos-pos_low)/(pos_high-pos_low)
    b = 40*(v-v_low)/(v_high-v_low)
    return int(a),int(b)
# print(transform_state([-1.0,0.01]))

#更新Q-Table  公式Q[s][a] = (1 - lr) * Q[s][a] + lr * (reward + factor * max(Q[next_s]))
#开始训练    factor折扣因子：factor越大，表示越重视历史经验，factor为0时，只关心当前利益（reward）
lr,factor = 0.7,0.95
episodes = 10000 #训练10000次
score_list = [] #记录所有分数
for i in range(episodes):
    s=transform_state(env.reset())
    score = 0
    while True:
        a = np.argmax(Q[s])
        #训练刚开始，多一点随机性，以便于有更多的状态
        if np.random.random()>i*3/episodes:
            a=np.random.choice([0,1,2])
        # 执行动作
        next_s, reward, done, _ = env.step(a)    
        next_s = transform_state(next_s)
        #根据上面公式更新Q-Table
        Q[s][a] = (1-lr)*Q[s][a]+lr*(reward+factor*max(Q[next_s]))
        score+=reward
        s=next_s
        if done:
            score_list.append(score)
            print('episose:',i,'score',score,'max:',max(score_list))
            break
env.close()
#保存模型
with open('MountainCar-v0-q-learning.pickle','wb') as f:
    pickle.dump(dict(Q),f)
    print('model saved')


#测试模型     test_q_learning.py
import time
import pickle
import gym
import numpy as np 
#加载模型
with open('MountainCar-v0-q-learning.pickle','rb') as f:
    Q=pickle.load(f)
    print('model loaded')
env = gym.make('MountainCar-v0')
s=env.reset()
score=0
while True:
    env.render()
    time.sleep(0.01)
    #transform_state函数与训练时一致
    s=transform_state(s)
    a=np.argmax(Q[s]) if s in Q else 0
    s,reward,done,_=env.step(a)
    score += reward
    if done:
        print('score:',score)
        break
env.close()