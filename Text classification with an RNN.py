#RNN LSTM 文本分类  LSTJM长短期记忆模型
#RNN 循环神经网络  使用RNN分类影评数据 IMDB
# RNN广泛适用于NLP（自然语言处理）  NLP往往需要能够更好地处理序列信息的神经网络
# RNN中，隐藏层状态，不仅取决于当前输入层的输出，还和上一步隐藏层的状态有关
# 长短期记忆模型（LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。LSTM能在更长的序列中有更好的表现                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

#下载IMDB
import matplotlib.pyplot as plt 
import tensorflow_datasets as tfds 
import tensorflow as tf
from tensorflow.keras import Sequential,layers
ds,info = tfds.load('imdb_reviews/subwords8k',with_info=True,
    as_supervised=True)
train_ds,test_ds = ds['train'],ds['test']

BUFFER_SIZE,BATCH_SIZE = 10000,64
train_ds = train_ds.shuffle(BUFFER_SIZE) #随机排序
train_ds = train_ds.padded_batch(BATCH_SIZE,train_ds.output_shapes)
#padded_batch( batch_size, padded_shapes, padding_values=None #默认使用各类型数据的默认值，一般使用时可忽略该项) 
# 参数padded_shapes ,指明每条记录中各成员要pad成的形状，成员若是scalar，则用[]，若是list，则用[mx_length]，若是array，则用[d1,...,dn]
test_ds = test_ds.padded_batch(BATCH_SIZE,test_ds.output_shapes)

#文本预处理
#通过tfds获取的数据已经经过文本预处理，即Tokenizer，向量化文本（将文本转为数字序列）
tokenizer = info.features['text'].encoder
print('词汇个数：',tokenizer.vocab_size)
sample_str = 'welcome to geektutu.com'
tokenized_str = tokenizer.encode(sample_str)
print('向量化文本：',tokenized_str)
for ts in tokenized_str:
    print(ts,'-->',tokenizer.decode([ts]))
# 词汇个数: 8185
# 向量化文本: [6351, 7961, 7, 703, 3108, 999, 999, 7975, 2449]
# 6351 --> welcome
# 7961 -->  
# 7 --> to 
# 703 --> ge
# 3108 --> ek
# 999 --> tu
# 999 --> tu
# 7975 --> .
# 2449 --> com

# 搭建 RNN 模型
model = Sequential([
    layers.Embedding(tokenizer.vocab_size,64), #经过Embedding层的转换，将产生大小固定的64的向量，而这个转换时可训练的，经过足够的训练之后，相似语义的句子将产生相似的向量
    layers.Bidirectional(layers.LSTM(64)), #在LSTM层外面套一个壳(层封装器，layer wrappers)，这是RNN的双向封装器，用于对序列进行前向和后向计算
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',
    metrics=['accuracy'])
history1 = model.fit(train_ds,epochs=3,validation_data=test_ds)
loss,acc = model.evaluate(test_ds)
print('准确率：',acc)#0.81039  使用一层LSTM
#将训练结果可视化
    #解决中文乱码问题
plt.rcParams['font.san-serif'] = ['SimHei']#设置字体
plt.rcParams['axes.unicode_minus'] = False #字符显示
plt.rcParams['font.size'] = 20
def plot_graphs(history,name):
    plt.plot(history.history[name])
    plt.plot(history.history['验证集-'+name])
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.legend([name,'验证集-'+name])
    plt.show()
plot_graphs(history1,'accuracy')

#添加更多LSTM层
model = Sequential([
    layers.Embedding(tokenizer.vocab_size,64)
    layers.Bidirectional(layers.LSTM(64,return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',  #交叉熵损失函数，一般用于二分类
    metrics=['accuracy'])
history = model.fit(train_ds,epochs=3,validation_data=test_ds)
loss,acc = model.evaluate(test_ds)
print('准确率：',acc)#0.83096   使用两层LSTM
plot_graphs(history,'accuracy')