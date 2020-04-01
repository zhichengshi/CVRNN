import sys 
sys.path.append('./') # 

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import sklearn.cluster as sc
import tensorflow as tf
from sampling import *
import model as cmodel
from config import *
import numpy as np 
import faiss 
from metric import top

def draw2DPicture(code_vectors,labels,draw_len):# 使用kmeans聚类，并绘制二维图像
    model = sc.KMeans(n_clusters=50)
    model.fit(code_vectors)
    prediction = model.predict(code_vectors)  # 预测


    # 降维
    tsne = TSNE(n_components=2, init='pca', verbose=1)
    embedd = tsne.fit_transform(code_vectors)

    # 可视化
    plt.figure(figsize=(40, 20))
    plt.scatter(embedd[:draw_len, 0], embedd[:draw_len, 1])

    for i in range(draw_len):
        x = embedd[i][0]
        y = embedd[i][1]
        plt.text(x, y, labels[i])
    plt.show()

def generateCodeMatrix(read_path,logdir,vector, vector_lookup): # 构建代码向量
    # innitial the network
    nodes_node, children_node, statement_len_list, code_vector, logits = cmodel.init_net(embedding_size, label_size)
    sess = tf.Session()

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ('Checkpoint not found.')

    correct_labels = []
    # make predictions from the input
    predictions = []
    step = 1

    # 存储代码向量的列表
    code_vectors = []
    # 标记
    labels = []
    for nodes, children, statement_len, label_vector in generateSample(read_path, vector,vector_lookup):
        code_vector_element = sess.run([code_vector], feed_dict={
            nodes_node: nodes,
            children_node: children,
            statement_len_list: statement_len
        })

        code_vectors.append(code_vector_element[0][0])
        labels.append(np.argmax(label_vector) + 1)

    print('the numebr of code vector is:',len(code_vectors))
    code_vector_path = "dataset/code_vector_c.pkl"
    with open(code_vector_path,"wb") as f:
        assert len(code_vectors)==len(labels)
        pickle.dump((code_vectors, labels), f)

def sampleSearchVector(code_vectors,labels):# 从代码库中选取作为query的代码向量
    code_vectors=np.asarray(code_vectors)
    indices=[]
    for i in range(1,51):
        indices.append(labels.index(i))
    return code_vectors[indices],list(range(1,51))

def searchCode(query,query_label,code_db,label_db): # 从代码库中搜索代码向量
    dimension=len(code_db[0])
    index=faiss.IndexFlatL2(dimension)
    index.add(code_db)
    D,I=index.search(query,11) #返回前11个 D:distance I:index

    predict_label=[]
    label_db=np.asarray(label_db)
    for row in I:
        predict_label.append(label_db[row[1:]])  # 除去第一个向量，第一个向量是查询向量

    predict_label=np.vstack(predict_label)

    return top(query_label,predict_label)



    

    




    



