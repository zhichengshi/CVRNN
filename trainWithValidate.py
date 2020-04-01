from config import *
import sys

import model
import tensorflow as tf
import os
from sampling import *
from sklearn.metrics import classification_report, accuracy_score

def train_model(logdir, pickle_data_path_train, pickle_data_path_validate, vector, vector_lookup):
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

    #  continue to train
    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, logdir+"/cnn_tree.ckpt-100000")
        else:
            raise ('Checkpoint not found.')
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir, sess.graph)



    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir, sess.graph)

    checkfile = os.path.join(logdir, 'cnn_tree.ckpt')

    step = 1
    error_sum = 0
    for epoch in range(1, epochs):
        # for nodes, children, statement_len, label_vector in generateSample(pickle_data_path_train, vector,vector_lookup):
        #     _, _, summary, error, out = sess.run([learning_rate, train_step, summaries, loss, out_node], feed_dict={
        #         nodes_node: nodes,
        #         children_node: children,
        #         statement_len_list: statement_len,
        #         label_node: label_vector,
        #         global_: step
        #     })
        #     error_sum += error
        #     if step % 1000 == 0:
        #         print('Epoch:', epoch, 'Step:', step, 'Loss:', error_sum / 1000)
        #         error_sum = 0
        #         writer.add_summary(summary, step)

        #     if step % check_every == 0:
        #         # save state so we can resume later
        #         saver.save(sess, checkfile, step)
        #     step += 1

        # 计算验证精度
        if epoch % 5 == 0:
            correct_labels = []
            predictions = []
            print("Epoch", epoch, 'Computing validate accuracy...')
            for nodes, children, statement_len, label_vector in generateSample(pickle_data_path_validate,
                                                                                           vector,
                                                                                           vector_lookup):
                output = sess.run([out_node], feed_dict={
                    nodes_node: nodes,
                    children_node: children,
                    statement_len_list: statement_len
                })

                correct_labels.append(np.argmax(label_vector))
                predictions.append(np.argmax(output))
            labels = []
            for i in range(1, label_size + 1):
                labels.append(str(i))
            validata_acc = accuracy_score(correct_labels, predictions)
            print('Accuracy:', validata_acc)
            print(classification_report(correct_labels, predictions, target_names=labels))


    saver.save(sess, checkfile, step)


if __name__ == "__main__":
    sys.setrecursionlimit(1000000)
    node_path = "dataset/embeddings.pkl"
    with open(node_path, "rb") as f:
        data = pickle.load(f)
        vector = data[0]
        vector_lookup = data[1]

    training_path = "dataset/104/test.pkl"
    validate_path = "dataset/104/validate.pkl"
    sess_graph_path = "log/train"
    train_model(sess_graph_path, training_path, validate_path, vector, vector_lookup)


