from config import *
import sys
import model
import tensorflow as tf
import os
from sampling import *
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import logging
import _pickle as pkl

def buildLogger(log_file, part):
    logger = logging.getLogger(part)
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    # FileHandler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def train_model(logdir, pickle_data_path_trains, pickle_data_path_validate, vector, vector_lookup):
    logger = buildLogger("log/cvRnnTrain.log", "train")

    # init the network
    nodes_node, children_node, statement_len_list, code_vector, logits = model.init_net(
        embedding_size, label_size
    )

    # for calculate the training accuracy
    out_node = model.out_layer(logits)

    label_node, loss = model.loss_layer(logits, label_size)

    global_ = tf.Variable(tf.constant(0))
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_, decay_steps, decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    tf.summary.scalar('loss', loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir, sess.graph)

    checkfile = os.path.join(logdir, 'cvrnn.ckpt')

    step = 0
    error_sum = 0
    for epoch in range(1, epochs+1):
        for train_path in pickle_data_path_trains:
            for nodes, children, statement_len, label_vector in generatePositiveSample(train_path, vector, vector_lookup):
                try:
                    _, _, summary, error, out = sess.run([learning_rate, train_step, summaries, loss, out_node], feed_dict={
                        nodes_node: nodes,
                        children_node: children,
                        statement_len_list: statement_len,
                        label_node: label_vector,
                        global_: step
                    })
                except Exception:
                    continue
                error_sum += error
                if step % 10000 == 0 and step !=0:
                    print(f'Epoch: {epoch},Step:{step},Loss:{error_sum/10000}')
                    error_sum = 0
                    writer.add_summary(summary, step)
                step += 1

        # 计算验证精度
        correct_labels = []
        predictions = []
        print("Epoch", epoch, 'Computing validate accuracy...')
        for nodes, children, statement_len, label_vector in generatePositiveSample(pickle_data_path_validate, vector, vector_lookup):
            try:
                output = sess.run([out_node], feed_dict={
                    nodes_node: nodes,
                    children_node: children,
                    statement_len_list: statement_len
                })
            except Exception:
                continue

            correct_labels.append(np.argmax(label_vector))
            predictions.append(np.argmax(output))
        labels = []
        for i in range(1, label_size + 1):
            labels.append(str(i))
        validata_acc = accuracy_score(correct_labels, predictions)
        logger.info(f'Epoch:{epoch},ValidAccuracy:{validata_acc}')
    
        # save model
        saver.save(sess, checkfile, epoch)


if __name__ == "__main__":

    node_path = "dataset/embeddings/word2vec_embeddings.pkl"
    with open(node_path, "rb") as f:
        data = pickle.load(f)
        vector = data[0]
        vector_lookup = data[1]
    

    train_paths = []  #!!! 因为训练集过大所以分为两个train文件分批加载
    train_dir = "dataset/104/train"
    for root, _, files in os.walk(train_dir):
        for file in files:
            train_paths.append(os.path.join(root, file))

    validate_path = "dataset/104/valid.pkl"
    sess_graph_path = "log/cvrnn"
 
    train_model(sess_graph_path, train_paths, validate_path, vector, vector_lookup)
