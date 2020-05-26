from sampling import generateNegativeSample, generatePositiveSample
import model
import numpy as np
import _pickle as pkl
import math
import tensorflow as tf
from cvRnn.config import *
import os
import sys
sys.path.append('./')


def generateCodeMatrix(read_path, logdir, vector, vector_lookup, dump_path, sampleFunc):  # 构建代码向量
    # innitial the network
    nodes, children, code_vector, prediction = model.init_net(embedding_size, label_size)
    with tf.Session() as sess:
        with tf.name_scope('saver'):
            saver = tf.train.Saver()
            saver.restore(sess, 'log/tbcnn/cnn_tree.ckpt-30')
            ckpt = tf.train.get_checkpoint_state(logdir)
        correct_labels = []
        # make predictions from the input
        predictions = []
        step = 1

        # 存储代码向量的列表
        code_vectors = []
        # 标记
        labels = []
        for nodes_node, children_node, label_vector in sampleFunc(read_path, vector, vector_lookup):
            try:
                code_vector_element = sess.run([code_vector], feed_dict={
                    nodes: nodes_node,
                    children: children_node,
                })
            except Exception:
                continue

            code_vectors.append(code_vector_element[0][0])
            labels.append(np.argmax(label_vector) + 1)

    with open(dump_path, "wb") as f:
        assert len(code_vectors) == len(labels)
        pkl.dump((code_vectors, labels), f)

    tf.reset_default_graph() # 清空计算图，函数执行结束之后不会自动释放显存


if __name__ == "__main__":
    positive_path = 'dataset/leetcodeCorpus.pkl'
    positive_dump_path = 'dataset/matrix/tbcnn/positive.pkl'

    negative_path1 = 'dataset/104/train/train1.pkl'
    negative_path2 = 'dataset/104/train/train2.pkl'
    negative_dump1 = 'dataset/matrix/tbcnn/negative1.pkl'
    negative_dump2 = 'dataset/matrix/tbcnn/negative2.pkl'
    negative_dump = 'dataset/matrix/tbcnn/negative.pkl'


    log_dir = 'log/tbcnn'
    embedding_path = 'dataset/embeddings/numpy_random_embeddings.pkl'

    with open(embedding_path, 'rb') as f:
        dataset = pkl.load(f)
        vector = dataset[0]
        vector_lookup = dataset[1]

    generateCodeMatrix(positive_path, log_dir, vector, vector_lookup, positive_dump_path, generatePositiveSample)
    generateCodeMatrix(negative_path1, log_dir, vector, vector_lookup, negative_dump1, generateNegativeSample)
    generateCodeMatrix(negative_path2, log_dir, vector, vector_lookup, negative_dump2, generateNegativeSample)

    with open(negative_dump1,'rb') as f:
        negative1=pkl.load(f)

    with open(negative_dump2,'rb') as f:
        negative2=pkl.load(f)

    vectors=list(negative1[0])+list(negative2[0])
    labels=negative1[1]+negative2[1]

    os.remove(negative_dump1)
    os.remove(negative_dump2)

    with open(negative_dump,'wb') as f:
        pkl.dump((vectors,labels),f)
