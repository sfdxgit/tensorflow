#过拟合和欠拟合  防止过拟合：权重正则化Dropout
# 探索正则化(weight regularization)和 dropout 两种避免过拟合的方式改善 IMDB 影评分类效果。

# 采用 multi-hot encoding的方式处理 IMDB 数据集,使其快速达到过拟合
# 给每一个单词一个编号，每句话用长度为10,000的一维向量表示，将出现单词的位置置为1即可
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
import numpy as np 
import matplotlib.pyplot as plt 

#解决中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.size']=20

N=10000#每句话用长度为N的一维向量表示
def multi_hot_encoding(sentences,dim=10000):
    results = np.zeros((len(sentences),dim))
    for i,word_indices in enumerate(sentences):
        results[i,word_indices] = 1.0
    return results
(train_x,train_y),(test_x,test_y) = imdb.load_data(num_words=N)
train_x = multi_hot_encoding(train_x)
test_x = multi_hot_encoding(test_x)
plt.plot(train_x[0])
plt.show()

#先搭一个基线模型，再创建大、小两个模型比较
def build_and_train(hidden_dim,regularizer=None,dropout=0):
    model = keras.Sequential([
        keras.layers.Dense(hidden_dim,activation='relu',
            input_shape=(N,),
            kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(hidden_dim,activation='relu',
            kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',
        metrics=['accuracy','binary_crossentropy'])
    history = model.fit(train_x,train_y,epochs=10,batch_size=512,
        validation_data=(test_x,test_y),verbose=0)
    return history
baseline_history = build_and_train(16)
smaller_history = build_and_train(4)
larger_history = build_and_train(512)
#画图看这三个模型在训练集上的表现
def plot_history(histories,key='binary_crossentropy'):#binary_crossentropy交叉熵损失函数，一般用于二分类
    plt.figure(figsize=(10,6))

    for name,history in histories:
        val = plt.plot(history.epoch,history.history['val_'+key],
            '--',label=name+'验证集')
        plt.plot(history.epoch,history.history[key],
            color=val[0].get_color(),label=name+'训练集')
    plt.xlabel('Epochs')
    plt.ylabel('Loss - '+key)
    plt.legend()
    plt.xlim([0,max(history.epoch)])
plot_history([('基线',baseline_history),
    ('较小',smaller_history),
    ('较大',larger_history)])
#如何防止过拟合    权重正则化
l2_model_history = build_and_train(16,keras.regularizers.l2(0.001))
dpt_model_history = build_and_train(16,dropout=0.2)
plot_history([('基线',baseline_history),
    ('L2正则',l2_model_history),
    ('Dropout',dpt_model_history)])