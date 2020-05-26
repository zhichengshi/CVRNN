import sys
sys.path.append('./')

import os
import faiss
import numpy as np
from cvRnn.config import *
import cvRnn.model as cmodel
from cvRnn.sampling import *
import tensorflow as tf
import sklearn.cluster as sc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
import math
import prettytable as pt



def draw2DPicture(code_vectors, labels, draw_len):  # 使用kmeans聚类，并绘制二维图像
    model = sc.KMeans(n_clusters=50)
    model.fit(code_vectors)
    prediction = model.predict(code_vectors)  # 预测

    # 降维
    tsne = TSNE(n_components=2, init='pca', verbose=1)
    embedd = tsne.fit_transform(code_vectors)

    # 可视化
    plt.figure(figsize=(10, 10))
    plt.scatter(embedd[:draw_len, 0], embedd[:draw_len, 1])

    for i in range(draw_len):
        x = embedd[i][0]
        y = embedd[i][1]
        plt.text(x, y, labels[i],fontsize=15)
    plt.show()


def generateCodeMatrix(read_path, logdir, vector, vector_lookup, dump_path, sampleFunc):  # 构建代码向量矩阵
    # innitial the network
    nodes_node, children_node, statement_len_list, code_vector, logits = cmodel.init_net(embedding_size, label_size)
    # sess = tf.Session()
    # tf.reset_default_graph()
    with tf.Session() as sess:
        # load
        with tf.name_scope('saver'):
            saver = tf.train.Saver()
            saver.restore(sess, 'log/cvrnn/cvrnn.ckpt-30')

        # generate
        code_vectors = []
        labels = []
        for nodes, children, statement_len, label_vector in sampleFunc(read_path, vector, vector_lookup):
            try:
                code_vector_element = sess.run([code_vector], feed_dict={
                    nodes_node: nodes,
                    children_node: children,
                    statement_len_list: statement_len
                })
            except Exception:
                continue
            code_vectors.append(code_vector_element[0][0])
            labels.append(np.argmax(label_vector) + 1)

        # dump
        with open(dump_path, "wb") as f:
            assert len(code_vectors) == len(labels)
            pickle.dump((code_vectors, labels), f)
        
    tf.reset_default_graph() # 清空计算图，函数执行结束之后不会自动释放显存
        



def mixNegativePositive(positive_vectors, positive_labels, negative_db_vectors, negative_db_labels):  # 从代码库中选取作为query的代码向量
    positive_vectors = np.asarray(positive_vectors)
    indices = []

    query_vectors = []
    query_labels = []
    db_vectors = []
    db_labels = []

    visit = set()

    for i in range(len(positive_labels)):  # 将leetcode上的数据分为查询和被查询部分
        if positive_labels[i] in visit:
            query_labels.append(positive_labels[i])
            query_vectors.append(positive_vectors[i])
        else:
            visit.add(positive_labels[i])
            db_vectors.append(positive_vectors[i])
            db_labels.append(positive_labels[i])
            
    negative_db_vectors=list(negative_db_vectors)
    negative_db_vectors.extend(db_vectors)  # 将被查询的数据集混入负样本
    negative_db_labels.extend(db_labels)

    return query_vectors, query_labels, negative_db_vectors, negative_db_labels


def searchCode(query, query_label, code_db, label_db):  # 从代码库中搜索代码向量
    code_db = np.vstack(code_db)
    query = np.vstack(query)
    dimension = len(code_db[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(code_db)
    D, I = index.search(query, 10)

    predict_label = []
    label_db = np.asarray(label_db)
    for row in I:
        predict_label.append(label_db[row])  # 除去第一个向量，第一个向量是查询向量

    predict_label = np.vstack(predict_label)

    return predict_label


def top(real, predict):
    top1 = []
    top2 = []
    top3 = []
    top5 = []
    top10 = []

    for i in range(len(real)):
        try:
            index = np.argwhere(predict[i] == real[i])[0][0]
            index += 1
        except Exception:
            top1.append(0)
            top2.append(0)
            top3.append(0)
            top5.append(0)
            top10.append(0)
            continue

        top10.append(1)

        if index <= 5:
            top5.append(1)
        else:
            top5.append(0)

        if index <= 3:
            top3.append(1)
        else:
            top3.append(0)

        if index <= 2:
            top2.append(1)
        else:
            top2.append(0)

        if index <= 1:
            top1.append(1)
        else:
            top1.append(0)

    return np.mean(top1), np.mean(top2), np.mean(top3), np.mean(top5), np.mean(top10)


def MRR(real, predict):
    predict = list(predict)
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum+1.0/float(index+1)
    return sum/float(len(real))


def NDCG(real, predict):
    predict = list(predict)
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i+1
            dcg += (math.pow(2, itemRelevance)-1.0)*(math.log(2)/math.log(rank+1))
    return dcg/float(idcg)


def IDCG(n):
    idcg = 0
    itemRelevance = 1
    for i in range(n):
        idcg += (math.pow(2, itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
    return idcg


def printMetric(real, predict):
    top1, top2, top3, top5, top10 = top(real, predict)
    ndcgs = []
    mrrs = []
    maps = []
    for i in range(len(predict)):
        ndcgs.append(NDCG([real[i]], predict[i]))
        mrrs.append(MRR([real[i]], predict[i]))
    ndcg_value = sum(ndcgs)/len(ndcgs)
    mrr_value = sum(mrrs)/len(mrrs)

    tb = pt.PrettyTable()
    tb.field_names = ["top1", 'top2', "top3", "top5", "top10", "ndcg", "mrr"]
    tb.add_row([top1, top2, top3, top5, top10, ndcg_value, mrr_value])
    print(tb)
