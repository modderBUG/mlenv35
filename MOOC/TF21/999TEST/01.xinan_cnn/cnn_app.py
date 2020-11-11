import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D
from tensorflow.keras import Model
from tensorflow import keras as kr
import sys
import numpy as np

"""
用于文本分类的app，加载模型输入预测值，输出预测结果。右键直接运行查看
"""

# 要使用模型 先还原模型
checkpoint_path = r"./ckpt/cnn_model/cnn_model.ckpt"
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

train_dir = r'./datasets/train.txt'
val_dir = r'./datasets/val.txt'
vocab_dir = r'./datasets/c_vocab.txt'


# 模型类，卷积、全局最大池化、全连接1、全连接2
class CnnModel(Model):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.embeddings = Embedding(vocab_size, embedding_dim)

        self.c1 = Conv1D(num_filters, kernel_size)

        self.p1 = GlobalMaxPooling1D()  # GlobalAveragePooling1D()

        self.d1 = Dense(hidden_dim, activation='relu')

        self.d2 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.embeddings(x)
        x = self.c1(x)
        x = self.p1(x)
        x = self.d1(x)
        y = self.d2(x)
        return y


# 模型创建
model = CnnModel()

# 模型加载
model.load_weights(checkpoint_path)

# 和训练脚本一样的数据预处理
if sys.version_info[0] > 2:
    is_py3 = True
else:
    # reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


# 数据预处理 - 词汇表映射
def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def process_file_my(contents, word_to_id, max_length=600):
    """将文件转换为id表示"""
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)

    return x_pad


words, word_to_id = read_vocab(vocab_dir)

# 待预测的文本，传入列表
# 1 2
content_text = [";【 新葡京】女神节9000万红包回馈 ，8-12号晚6-10点免费抢红包V信:zgh91575 官网:5197ff.com",
                "看免费真人AV、www.amh1588.com复制到浏览器打开哟"]  # np.array([[4,0.01,1,1],[2,0.19,0,5],[3,0.17,1,0]])  #1,2
img_arr = process_file_my(content_text, word_to_id)
x_predict = img_arr
result = model.predict(x_predict)

result = tf.argmax(result,1)


print('\n')
tf.print(result)
