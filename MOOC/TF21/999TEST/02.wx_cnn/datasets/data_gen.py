from PIL import Image
import numpy as np
import os

# 生成 路径对应标签文件夹 路劲 值 形式
def gen_txt(path,name):
    train_dir= path
    train_dir_list = os.listdir(train_dir)
    for index,item in enumerate(train_dir_list):
        class_dir_list = os.listdir(os.path.join(train_dir,item))
        for index2,file in enumerate(class_dir_list):
            # if index2 == 8:break
            with open(name,"a+") as f:
                f.write(os.path.join(train_dir,item,file)+'\t'+str(index)+"\n")


def generated_data(txt):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    x, y_ = [], []
    for content in contents:
        value = content.split()
        img_path =value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img = img / 255.0
        x.append(img)
        y_.append(value[1])

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_


# train_path = ''
train_txt = 'datasets/train_txt.txt'
x_train_savepath = 'datasets/img_x_train.npy'
y_train_savepath = 'datasets/img_y_train.npy'

# test_path = ''
test_txt = 'datasets/test_txt.txt'
x_test_savepath = 'datasets/img_x_test.npy'
y_test_savepath = 'datasets/img_y_test.npy'


def save_data():
    if os.path.exists(x_train_savepath):
        # load
        print("--------Load Datasets--------")
        x_train_save = np.load(x_train_savepath,allow_pickle=True)
        y_train = np.load(y_train_savepath,allow_pickle=True)

        x_test_save = np.load(x_test_savepath,allow_pickle=True)
        y_test = np.load(y_test_savepath,allow_pickle=True)

        x_train = np.reshape(x_train_save,(len(x_train_save),28,28))
        x_test = np.reshape(x_test_save,(len(x_test_save),28,28))
    else:
        print("---------Generate Datasets-----------")
        x_train,y_train = generated_data(train_txt)
        x_test,y_test = generated_data(test_txt)

        print("-----------Save    Datasets------------")
        x_train_save = np.reshape(x_train,(len(x_train),-1))
        x_test_save = np.reshape(x_test,(len(x_test),-1))

        np.save(x_train_savepath,x_train_save)
        np.save(x_test_savepath,x_test_save)
        np.save(y_train_savepath,y_train)
        np.save(y_test_savepath,y_test)
    return x_train,y_train,x_test,y_test

gen_txt(r'D:\Data\classify\train',"datasets/train_txt.txt")
gen_txt(r'D:\Data\classify\test',"datasets/test_txt.txt")

print("----ok------")

save_data()
print("----ok------")
# x_train_save = np.load(x_train_savepath,allow_pickle=True)
#
# y_train = np.load(y_train_savepath,allow_pickle=True)
# print(x_train_save)

