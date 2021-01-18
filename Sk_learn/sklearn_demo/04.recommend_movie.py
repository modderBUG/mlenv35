import json
import numpy as np

'''
    推荐引擎 (用户画像):把最需要的推荐给用户。
        在不同的机器学习场景中通常需要分析相似样本。而统计相似样本的方式可以基于欧氏距离分数，也可基于皮氏距离分数。
            欧氏距离分数 :
                欧氏距离分数 = 1/(1+欧式距离)
                            ----计算所得欧氏距离分数区间处于：(0, 1]，越趋于0样本间的欧氏距离越远，样本越不相似；
                                越趋于1，样本间的欧氏距离越近，越相似。

            皮尔逊相关系数:
                    A = [1,2,3,1,2]
                    B = [3,4,5,3,4]
                    m = np.corrcoef(A, B)
                    皮尔逊相关系数 = 协方差 / 标准差之积
                    相关系数处于[-1, 1]区间。越靠近-1代表两组样本反相关，越靠近1代表两组样本正相关

            生成推荐清单:
                    1.找到所有皮尔逊系数正相关的用户
                    2.遍历当前用户的每个相似用户，拿到相似用户看过但是当前用户没有看过的电影作为推荐电影
                    3.多个相似用户有可能推荐同一部电影，则取每个相似用户对该电影的评分得均值作为推荐度。


    案例：解析ratings.json，根据每个用户对已观看电影的打分计算样本间的欧氏距离，输出欧氏距离得分矩阵。
'''

with open('./ml_data/ratings.json', 'r') as f:
    ratings = json.loads(f.read())

print(ratings)

# 整理用户之的相似度得分矩阵
users, scmat = list(ratings.keys()), []
for user1 in users:
    scrow = []
    for user2 in users:
        # 计算user1与user2的相似度 添加到scrow
        movies = set()
        for movie in ratings[user1]:
            if movie in ratings[user2]:
                movies.add(movie)
        if len(movies) == 0:
            score = 0
        else:
            A, B = [], []
            for movie in movies:
                A.append(ratings[user1][movie])
                B.append(ratings[user2][movie])
            A = np.array(A)
            B = np.array(B)
            # 计算A与B的相似度---欧式距离
            # score = 1 / (1 + np.sqrt(((A - B) ** 2)).sum())
            # 计算A与B的相似度---皮尔逊相关系数
            score = np.corrcoef(A, B)[0, 1]
        scrow.append(score)
    scmat.append(scrow)

users = np.array(users)
scmat = np.array(scmat)

for scrow in scmat:
    print(' '.join(['{:.2f}'.format(score) for score in scrow]))

# 按照相似度从高到低排列每个用户的相似用户
for i, user in enumerate(users):
    # 获取所有相似用户得分，去掉自己，排序
    sorted_indices = scmat[i].argsort()[::-1]
    sorted_indices = sorted_indices[sorted_indices != i]  # 剔除自己
    # user的所有相似用户
    sim_users = users[sorted_indices]
    # user所有相似用户的相似度得分
    sim_scores = scmat[i, sorted_indices]
    # print(user, sim_users, sim_scores, sep='\n')

    # 生成推荐清单
    # 正相关得分的掩码
    positive_mask = sim_scores > 0
    # 获取所有正相关用户的用户名
    sim_users = sim_users[positive_mask]
    # 为user构建推荐清单，找到每个sim_user看过，但是当前用户没有看过的电影，存入字典结构
    # 存储推荐清单:{'电影1':[4.0,5.0],'电影2':[3.0,4.0]}
    reco_movies = {}
    for j, sim_user in enumerate(sim_users):
        for movie, score in ratings[sim_user].items():
            # 相似用户看过但是当前用户没有看过
            if movie not in ratings[user].keys():
                if movie not in reco_movies:
                    reco_movies[movie] = [score]
                else:
                    reco_movies[movie].append(score)
    # print(user, reco_movies, sep='  ')

    # 对推荐清单进行排序
    movie_list = sorted(reco_movies.items(), key=lambda x: np.average(x[1]), reverse=True)
    print(user, movie_list, sep=' ')
    # 输出结果：{'John Carson': {'Inception': 2.5, 'Pulp Fiction': 3.5, 'Anger Management': 3.0, 'Fracture': 3.5, 'Serendipity': 2.5, 'Jerry Maguire': 3.0}, 'Michelle Peterson': {'Inception': 3.0, 'Pulp Fiction': 3.5, 'Anger Management': 1.5, 'Fracture': 5.0, 'Jerry Maguire': 3.0, 'Serendipity': 3.5}, 'William Reynolds': {'Inception': 2.5, 'Pulp Fiction': 3.0, 'Fracture': 3.5, 'Jerry Maguire': 4.0}, 'Jillian Hobart': {'Pulp Fiction': 3.5, 'Anger Management': 3.0, 'Jerry Maguire': 4.5, 'Fracture': 4.0, 'Serendipity': 2.5}, 'Melissa Jones': {'Inception': 3.0, 'Pulp Fiction': 4.0, 'Anger Management': 2.0, 'Fracture': 3.0, 'Jerry Maguire': 3.0, 'Serendipity': 2.0}, 'Alex Roberts': {'Inception': 3.0, 'Pulp Fiction': 4.0, 'Jerry Maguire': 3.0, 'Fracture': 5.0, 'Serendipity': 3.5}, 'Michael Henry': {'Pulp Fiction': 4.5, 'Serendipity': 1.0, 'Fracture': 4.0}}1.00 0.40 0.40 0.57 0.59 0.75 0.990.40 1.00 0.20 0.31 0.41 0.96 0.380.40 0.20 1.00 1.00 -0.26 0.13 -1.000.57 0.31 1.00 1.00 0.57 0.03 0.890.59 0.41 -0.26 0.57 1.00 0.21 0.920.75 0.96 0.13 0.03 0.21 1.00 0.660.99 0.38 -1.00 0.89 0.92 0.66 1.00John Carson []Michelle Peterson []William Reynolds [('Serendipity', [2.5, 2.5, 3.5, 3.5]), ('Anger Management', [3.0, 3.0, 1.5])]Jillian Hobart [('Inception', [2.5, 3.0, 2.5, 3.0, 3.0])]Melissa Jones []Alex Roberts [('Anger Management', [1.5, 3.0, 2.0, 3.0])]Michael Henry [('Jerry Maguire', [3.0, 3.0, 4.5, 3.0, 3.0]), ('Inception', [2.5, 3.0, 3.0, 3.0]), ('Anger Management', [3.0, 2.0, 3.0, 1.5])]