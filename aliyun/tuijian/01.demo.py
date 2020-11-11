import pandas as pd
import jieba
#coding:utf-8

"""
基于贝叶斯网络的新闻分类。
1.读取文本，用结巴将词语分开。
2.用stopwords匹配，去除所有步骤1所有的stopwords。
3.转为词向量。
4.用CountVectorize提取文本特征
5.用MultinomialNB贝叶斯分类算法训练数据。预测数据
"""

df_news = pd.read_table(r'aaa.dat',names=['category','content'],encoding='utf-8',sep="\t")
df_news = df_news.dropna()
# print(df_news.head())

content = df_news.content.values.tolist()

content_S = []
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment)>1 and current_segment!="\r\n":
        content_S.append(current_segment)

# print(content_S)

df_content = pd.DataFrame({"content_S":content_S})
# print(df_content.head())

stopwords = pd.read_csv("chineseStopWords.txt",index_col=False,sep='\t',quoting=3,names=['stopword'],encoding='gbk')

# print(stopwords.head())


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


contents =df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean,all_words =drop_stopwords(contents,stopwords)

df_content = pd.DataFrame({'contents_clean':contents_clean})
# print(df_content)

df_all_words = pd.DataFrame({'all_words':all_words})
# print(df_all_words.head())
import numpy
words_count = df_all_words.groupby(by=['all_words']).agg(计数=pd.NamedAgg(column='all_words', aggfunc='size')).reset_index().sort_values(
    by='计数', ascending=False)
# words_count =words_count.agg({"count":numpy.size}).reset_index().sort_values(by=["count"],ascending=False)
# print(words_count[ :20])

"""
词汇云....省略
"""
import jieba.analyse
index=24
# print(df_news["content"][index])
content_S_str = "".join(content_S[index])

"""
开始
"""


from gensim import corpora,models,similarities
import gensim

dictionary = corpora.Dictionary(contents_clean)
corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]

lda =gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=20)

# print(lda.print_topic(1,topn=5))

# for topic in lda.print_topics(num_topics=20,num_words=5):
    # print(topic[1])


df_train = pd.DataFrame({'content':contents_clean,'label':df_news['category']})
# print(df_train.tail())


print(df_train.label.unique())

label_mapping = {'财经':1,'IT':2,'健康' :3,'体育':4, '旅游':5, '教育':6, '招聘':7, '文化':8, '军事':9}
df_train['label'] = df_train['label'].map(label_mapping)

# print(df_train.head())


"""
来了，要来了！
"""

from sklearn.model_selection import train_test_split
def create_words(data):
    words = []
    for index in range(len(data)):
        try:
            words.append( ' '.join(data[index]))
        except Exception:
            print(index)
    return words

x_train,x_test,y_train,y_test = train_test_split(df_train['content'].values,df_train['label'].values,random_state=0)

train_words = create_words(x_train)
test_words = create_words(x_test)

# 这里需要向将内容转换为词向量再进行分类：

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(analyzer = 'word',max_features=4000,lowercase=False)
vec.fit(train_words)
classifier = MultinomialNB()
classifier.fit(vec.transform(train_words),y_train)
score = classifier.score(vec.transform(test_words), y_test)


# -*- coding: utf-8 -*-
a=["在弗尔切克的个人网站上，他将自己描述为小说家、哲学家、电影制作人和调查记者，同时也是一个“反对西方帝国主义和将西方政权模式强加给世界的革命者、国际主义者和环球旅行者”，他长期关注包括伊拉克、斯里兰卡、波斯尼亚、卢旺达和叙利亚在内的数十个战乱和冲突地区。"]
b=["CrawlSpider类可以帮我们在页面上通过正则提取满足条件的url自动进一步爬取，而普通爬虫则花费了大量时间去构造url。"]
from  aliyun.tuijian.utils import deal_data

train_words =deal_data(a)
print("train_words：",train_words)
print("vec.transform(train_words)：",vec.transform(train_words))
result =  classifier.predict(vec.transform(train_words))
result2 =  classifier.predict(vec.transform(deal_data(b)))
print(result)
print(result2)

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(analyzer='word',max_features = 40000,lowercase=False)
# vectorizer.fit(train_words)
#
# classifier.fit(vectorizer.transform(train_words),y_train)
# score = classifier.score(vectorizer.transform(test_words), y_test)
#
# print(score)