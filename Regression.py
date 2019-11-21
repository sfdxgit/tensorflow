#回归预测烟油效率   Auto MPG 数据集
#回归通常用来预测连续值

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#获取数据
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
dataset_path = keras.utils.get_file("auto-mpg.data",url)
column_names = ['MPG','气缸','排量','马力','重量','加速度', '年份', '产地']
raw_dataset = pd.read_csv(dataset_path,names=column_names,
    na_values="?",comment='\t',
    sep=" ",skipinitialspace=True)#skipinitialspace忽略分隔符后的空白 false默认不忽略
dataset = raw_dataset.copy()
# dataset.head(3)#查看前三条数据

#清洗数据 检查是否有NA值
dataset.isna().sum()
# print(dataset.isna().sum()) #马力
dataset = dataset.dropna()#去除马力为NA值的行
#在获取的数据集中，origin不是数值类型，需转为独热编码
origin = dataset.pop('产地') #原数据中将不存在该列
dataset['美国']=(origin==1)*1.0
dataset['欧洲']=(origin==2)*1.0
dataset['日本']=(origin==3)*1.0
# print(dataset.head(3))

#划分训练集和测试集
    #随机分配80%的数据作为训练集
    #frac是保留80%的数据
    #random_state相当于随机数的种子，在这里固定一个值是为了每次运行，随机分配得到的样本集是相同的
train_dataset = dataset.sample(frac=0.8,random_state=0)
    #除掉训练集数据剩余的数据
test_dataset = dataset.drop(train_dataset.index)
#检查数据  解决中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei']#用来显示中文标签
plt.rcParams['axes.unicode_minus']=False #解决负号'-'显示为方块的问题
sns.pairplot(train_dataset[["MPG","气缸","排量","重量"]],diag_kind="kde")
# plt.show()

train_stats = train_dataset.describe()#快速浏览每一属性的平均值、标准差、最大值、最小值等
train_stats.pop("MPG")
train_stats = train_stats.transpose()#transpose用来转换矩阵维度
# print(train_stats)

#分离label
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop('MPG')

#归一化数据
def norm(x):
    return (x-train_stats['mean'])/train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#搭建模型  模型包含2个全连接的隐藏层构成，输出层返回一个连续值
def build_model():
    input_dim = len(train_dataset.keys())
    model = keras.Sequential([
        layers.Dense(64,activation='relu',input_shape=[input_dim,]),
        layers.Dense(64,activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mse',metrics=['mae','mse'],
        optimizer = tf.keras.optimizers.RMSprop(0.001))
    return model
model = build_model()
#打印模型的描述信息，每一层大小、参数个数等
# print(model.summary())

#训练模型  自定义
import sys
EPOCHS = 1000
class ProgressBar(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        #显示进度条
        self.draw_progress_bar(epoch+1,EPOCHS)
    
    def draw_progress_bar(self,cur,total,bar_len=50):
        cur_len = int(cur / total*bar_len)
        sys.stdout.write("\r")
        sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len,bar_len,cur,total))
        sys.stdout.flush()
# history = model.fit(
#     normed_train_data,train_labels,
#     epochs = EPOCHS,validation_split = 0.2,verbose = 0,
#     callbacks=[ProgressBar()]
# )#history中存储训练过程
 #借助matplotlib将驯良过程可视化
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail(3))
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('metric-MSE')
    plt.plot(hist['epoch'],hist['mse'],label='训练集')
    plt.plot(hist['epoch'],hist['val_mse'],label='验证集')
    plt.ylim([0,20])#获取或者是设定y坐标轴的范围，当前axes上的座标轴
    plt.legend()#给图加上图例
    plt.show()

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('metric-MAE')
    plt.plot(hist['epoch'],hist['mae'],label='训练集')
    plt.plot(hist['epoch'],hist['val_mae'],label='验证集')
    plt.ylim([0,5])
    plt.legend()
    plt.show()
# plot_history(history)

#当训练集的loss降低，而验证集的loss升高，说明出现过拟合（高方差），训练应该早一点结束
#使用 keras.callbacks.EarlyStopping，每一波(epoch)训练结束时，测试训练情况，如果训练不再有效果（验证集的loss，即val_loss 不再下降），则自动地停止训练
model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
history = model.fit(
    normed_train_data,train_labels,
    epochs = EPOCHS,validation_split = 0.2,verbose = 0,
    callbacks=[early_stop,ProgressBar()]
)
# plot_history(history)

#测试集评估效果
loss,mae,mse = model.evaluate(normed_test_data,test_labels,verbose=0)
print("测试集平均绝对误差MAE:{:5.2f}MPG".format(mae))
#预测  使用测试集中的数据来预测MPG值
test_pred = model.predict(normed_test_data).flatten()
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100,100],[-100,100])
plt.show()

#结论：1均方误差(Mean Squared Error, MSE) 常作为回归问题的损失函数(loss function)，与分类问题不太一样。
# 2同样，评价指标(evaluation metrics)也不一样，分类问题常用准确率(accuracy)，回归问题常用 平均绝对误差 (Mean Absolute Error, MAE)
# 3每一列数据都有不同的范围，每一列，即每一个feature的数据需要分别缩放到相同的范围。常用归一化的方式，缩放到[0, 1]。
# 4如果训练数据过少，最好搭建一个隐藏层少的小的神经网络，避免过拟合。
# 5早停法(Early Stoping)也是防止过拟合的一种方式。
