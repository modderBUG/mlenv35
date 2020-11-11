import os


base_path=r"C:\Users\wuxiaowei_a\PycharmProjects\Machine-Learning\Naive Bayes\SogouC\Sample"

all =  os.listdir(r"C:\Users\wuxiaowei_a\PycharmProjects\Machine-Learning\Naive Bayes\SogouC\Sample")
all_data=[]
def gen_data():
    for dir in all:
        path_list = os.listdir(os.path.join(base_path,dir))
        for path in path_list:
            file_path = os.path.join(base_path,dir,path)
            title=""
            if dir =="C000008":
                title = "财经"
            elif dir =="C000010":
                title="IT"
            elif dir =="C000013":
                title="健康"
            elif dir =="C000014":
                title="体育"
            elif dir =="C000016":
                title="旅游"
            elif dir =="C000020":
                title="教育"
            elif dir =="C000022":
                title="招聘"
            elif dir =="C000023":
                title="文化"
            elif dir =="C000024":
                title="军事"

            with open(file_path,'r',encoding="utf-8") as f:
                content = f.read()
                item ={
                    "cls":title,
                    "content":content.replace("\r\n","").replace("\n","").replace("\t","").replace(" ","")
                }
                all_data.append(item)

    with open("aaa.dat","w+",encoding="utf-8") as f:
        for item in all_data:
            f.write(item["cls"]+"\t"+item["content"]+"\n")
import jieba
import pandas as pd
def drop_stopwords(contents,stopwords):
    contents_clean=[]
    all_words=[]
    for line in contents:
        line_clean=[]
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(str(word))
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words
def create_words(data):
    words = []
    for index in range(len(data)):
        try:
            words.append( ' '.join(data[index]))
        except Exception:
            print(index)
    return words
def deal_data(content):
    # content = df_news.content.values.tolist()
    content_S = []
    for line in content:
        current_segment = jieba.lcut(line)
        if len(current_segment) > 1 and current_segment != "\r\n":
            content_S.append(current_segment)

    df_content = pd.DataFrame({"content_S": content_S})
    # print(df_content.head())
    stopwords = pd.read_csv("chineseStopWords.txt", index_col=False, sep='\t', quoting=3, names=['stopword'],
                            encoding='gbk')
    contents = df_content.content_S.values.tolist()
    stopwords = stopwords.stopword.values.tolist()
    contents_clean, all_words = drop_stopwords(contents, stopwords)

    df_train = pd.DataFrame({'content': contents_clean})

    train_words = create_words(df_train['content'].values)

    return train_words



import random

if __name__=='__main__':
    # gen_data()
    # deal_data()
    # maxsimi = round(float(random.random()),2)
    # print(maxsimi)
    pass
