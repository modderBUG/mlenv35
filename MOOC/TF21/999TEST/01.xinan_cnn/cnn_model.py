import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D
from tensorflow.keras import Model
from tensorflow import keras as kr
import numpy as np
import sys
import os
import matplotlib.pyplot  as plt

"""
重构文本预测模型。使用tensorflow2.0。
"""

if sys.version_info[0] > 2:
    is_py3 = True
else:
    # reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


# 预处理 - 打开文件
def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


# 预处理 - 拼接训练集字符串
def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


# 预处理 - 读取文本数据和标签
def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('<->')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels


# 原生 #弃用# 因为新模型不用转换y标签
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad



def read_category():
    """读取分类目录，固定"""
    categories = ['0', '1', '2', '3', '4']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

# 数据预处理 - 词汇表映射
def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


# 模型超参数
embedding_dim = 64  # 词向量维度
seq_length = 600  # 序列长度

num_classes = 5  # 类别数
num_filters = 256  # 卷积核数目
kernel_size = 5  # 卷积核尺寸
vocab_size = 5000  # 词汇表达小

hidden_dim = 128  # 全连接层神经元

dropout_keep_prob = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率

batch_size = 64  # 每批训练大小
num_epochs = 10  # 总迭代轮次

print_per_batch = 100  # 每多少轮输出一次结果
save_per_batch = 10  # 每多少轮存入tensorboard

# 配置词汇表路径、训练集、测试集路径
train_dir = r'./datasets/train.txt'
val_dir = r'./datasets/val.txt'
vocab_dir = r'./datasets/c_vocab.txt'
words, word_to_id = read_vocab(vocab_dir)
categories, cat_to_id = read_category()

# 输入数据
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
x_test, y_test = process_file(val_dir, word_to_id, cat_to_id, seq_length)


# 模型类，卷积、全局最大池化、全连接1、全连接2
class CnnModel(Model):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.embeddings = Embedding(vocab_size, embedding_dim)

        self.c1 = Conv1D(num_filters, kernel_size)

        self.p1 = GlobalMaxPooling1D()  # GlobalAveragePooling1D()

        self.d1 = Dense(hidden_dim, activation='relu')

        self.d2 = Dense(num_classes, activation='softmax')

    @tf.function
    def call(self, x):
        x = self.embeddings(x)
        x = self.c1(x)
        x = self.p1(x)
        x = self.d1(x)
        y = self.d2(x)
        return y


# 模型创建
model = CnnModel()
# 模型编译
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              # loss=tf.keras.losses.BinaryCrossentropy(),
              loss=tf.keras.losses.CategoricalCrossentropy(),


              # loss = tf.keras.losses.mean_squared_error(),
              metrics=['categorical_accuracy'])
# 模型保存
checkpoint_path = "./ckpt/cnn_model/cnn_model.ckpt"  # 路径需要调整
if os.path.exists(checkpoint_path + '.index'):
    print("load the model")
    model.load_weights(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
# 喂入数据
history = model.fit(x_train, y_train, batch_size=64, epochs=1,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])

# 执行模型
model.summary()


tf.saved_model.save()
tf.saved_model.load

# -------------绘制误差图像-----------------------------
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training And Validation Accuracy')
plt.legend()
plt.show()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Training And Validation loss')
plt.legend()
plt.show()
# ---------------------end---------------------------------
