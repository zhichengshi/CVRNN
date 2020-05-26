"""Build a CNN network that learns a convolution over a tree structure as
described in Lili Mou et al. (2015) https://arxiv.org/pdf/1409.5718.pdf"""
import sys
sys.path.append('./')
from cvRnn.config import *
import tensorflow as tf
import math



def init_net(feature_size, label_size):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes = tf.placeholder(tf.float32, shape=(batch_size, None, None, feature_size), name='tree')
        children = tf.placeholder(tf.int32, shape=(batch_size, None, None, None), name='children')
        subtree_len_list = tf.placeholder(tf.int32, shape=(batch_size,))

    with tf.name_scope('network'):
        hidden_list = []
        for i in range(0, nodes.shape[0]):
            conv1 = conv_layer(num_conv, tree_embedding_size, nodes[i], children[i], feature_size)
            pooling = pooling_layer(conv1)
            # hidden = hidden_layer(pooling, 100, label_size)
            hidden_list.append(pooling)
        hidden_stack = tf.stack(hidden_list)

        code_vector = bi_directional_rnn_layer(hidden_stack, subtree_len_list)  # 双向循环神经网络
        # code_vector = multi_rnn_layer(hidden_stack, subtree_len_list)  # 双向循环神经网络
        # code_vector=max_pooling(hidden_stack)
        prediction = softmax_layer(code_vector, rnn_size, label_size)

    with tf.name_scope('summaries'):
        tf.summary.scalar('tree_size', tf.shape(nodes)[2])
        tf.summary.scalar('child_size', tf.shape(children)[3])
        # tf.summary.image('inputs', tf.expand_dims(nodes, axis=3))
        # tf.summary.image('conv1', tf.expand_dims(conv1, axis=3))

    return nodes, children, subtree_len_list, code_vector, prediction


def conv_layer(num_conv, output_size, nodes, children, feature_size):
    """Creates a convolution layer with num_conv convolutions merged together at
    the output. Final output will be a tensor with shape
    [batch_size, num_nodes, output_size * num_conv]"""

    with tf.name_scope('conv_layer'):
        nodes = [
            conv_node(nodes, children, feature_size, output_size)
            for _ in range(num_conv)
        ]
        return tf.concat(nodes, axis=2)


def conv_node(nodes, children, feature_size, output_size):
    """Perform convolutions over every batch sample."""
    with tf.name_scope('conv_node'):
        std = 1.0 / math.sqrt(feature_size)
        w_t, w_l, w_r = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wt'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wr'),
        )
        init = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / feature_size))
        # init = tf.zeros([output_size,])
        b_conv = tf.Variable(init, name='b_conv')

        with tf.name_scope('summaries'):
            tf.summary.histogram('w_t', [w_t])
            tf.summary.histogram('w_l', [w_l])
            tf.summary.histogram('w_r', [w_r])
            tf.summary.histogram('b_conv', [b_conv])

        return conv_step(nodes, children, feature_size, w_t, w_r, w_l, b_conv)


def children_tensor(nodes, children, feature_size):
    """Build the children tensor from the input nodes and child lookup."""
    with tf.name_scope('children_tensor'):
        max_children = tf.shape(children)[2]
        batch_size = tf.shape(nodes)[0]
        num_nodes = tf.shape(nodes)[1]

        # replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        # zero_vecs is (batch_size, num_nodes, 1)
        zero_vecs = tf.zeros((batch_size, 1, feature_size))
        # vector_lookup is (batch_size x num_nodes x feature_size)
        vector_lookup = tf.concat([zero_vecs, nodes[:, 1:, :]], axis=1)
        # children is (batch_size x num_nodes x num_children x 1)
        children = tf.expand_dims(children, axis=3)
        # prepend the batch indices to the 4th dimension of children
        # batch_indices is (batch_size x 1 x 1 x 1)
        batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
        # batch_indices is (batch_size x num_nodes x num_children x 1)
        batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
        # children is (batch_size x num_nodes x num_children x 2)
        children = tf.concat([batch_indices, children], axis=3)
        # output will have shape (batch_size x num_nodes x num_children x feature_size)
        # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
        return tf.gather_nd(vector_lookup, children, name='children')


def eta_t(children):
    """Compute weight matrix for how much each vector belongs to the 'top'"""
    with tf.name_scope('coef_t'):
        # children is shape (batch_size x max_tree_size x max_children)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]
        # eta_t is shape (batch_size x max_tree_size x max_children + 1)
        return tf.tile(tf.expand_dims(tf.concat(
            [tf.ones((max_tree_size, 1)), tf.zeros((max_tree_size, max_children))],
            axis=1), axis=0,
        ), [batch_size, 1, 1], name='coef_t')


def eta_r(children, t_coef):
    """Compute weight matrix for how much each vector belogs to the 'right'"""
    with tf.name_scope('coef_r'):
        # children is shape (batch_size x max_tree_size x max_children)
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]

        # num_siblings is shape (batch_size x max_tree_size x 1)
        num_siblings = tf.cast(
            tf.count_nonzero(children, axis=2, keep_dims=True),
            dtype=tf.float32
        )
        # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
        num_siblings = tf.tile(
            num_siblings, [1, 1, max_children + 1], name='num_siblings'
        )
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2, name='mask'
        )

        # child indices for every tree (batch_size x max_tree_size x max_children + 1)
        child_indices = tf.multiply(tf.tile(
            tf.expand_dims(
                tf.expand_dims(
                    tf.range(-1.0, tf.cast(max_children, tf.float32), 1.0, dtype=tf.float32),
                    axis=0
                ),
                axis=0
            ),
            [batch_size, max_tree_size, 1]
        ), mask, name='child_indices')

        # weights for every tree node in the case that num_siblings = 0
        # shape is (batch_size x max_tree_size x max_children + 1)
        singles = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.fill((batch_size, max_tree_size, 1), 0.5),
             tf.zeros((batch_size, max_tree_size, max_children - 1))],
            axis=2, name='singles')

        # eta_r is shape (batch_size x max_tree_size x max_children + 1)
        return tf.where(
            tf.equal(num_siblings, 1.0),
            # avoid division by 0 when num_siblings == 1
            singles,
            # the normal case where num_siblings != 1
            tf.multiply((1.0 - t_coef), tf.divide(child_indices, num_siblings - 1.0)),
            name='coef_r'
        )


def eta_l(children, coef_t, coef_r):
    """Compute weight matrix for how much each vector belongs to the 'left'"""
    with tf.name_scope('coef_l'):
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2,
            name='mask'
        )

        # eta_l is shape (batch_size x max_tree_size x max_children + 1)
        return tf.multiply(
            tf.multiply((1.0 - coef_t), (1.0 - coef_r)), mask, name='coef_l'
        )


def conv_step(nodes, children, feature_size, w_t, w_r, w_l, b_conv):
    """Convolve a batch of nodes and children.

    Lots of high dimensional tensors in this function. Intuitively it makes
    more sense if we did this work with while loops, but computationally this
    is more efficient. Don't try to wrap your head around all the tensor dot
    products, just follow the trail of dimensions.
    """
    with tf.name_scope('conv_step'):
        # nodes is shape (batch_size x max_tree_size x feature_size)
        # children is shape (batch_size x max_tree_size x max_children)

        with tf.name_scope('trees'):
            # children_vectors will have shape
            # (batch_size x max_tree_size x max_children x feature_size)
            children_vectors = children_tensor(nodes, children, feature_size)

            # add a 4th dimension to the nodes tensor
            nodes = tf.expand_dims(nodes, axis=2)
            # tree_tensor is shape
            # (batch_size x max_tree_size x max_children + 1 x feature_size)
            tree_tensor = tf.concat([nodes, children_vectors], axis=2, name='trees')

        with tf.name_scope('coefficients'):
            # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
            c_t = eta_t(children)
            c_r = eta_r(children, c_t)
            c_l = eta_l(children, c_t, c_r)

            # concatenate the position coefficients into a tensor
            # (batch_size x max_tree_size x max_children + 1 x 3)
            coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')

        with tf.name_scope('weights'):
            # stack weight matrices on top to make a weight tensor
            # (3, feature_size, output_size)
            weights = tf.stack([w_t, w_r, w_l], axis=0)

        with tf.name_scope('combine'):
            batch_size = tf.shape(children)[0]
            max_tree_size = tf.shape(children)[1]
            max_children = tf.shape(children)[2]

            # reshape for matrix multiplication
            x = batch_size * max_tree_size
            y = max_children + 1
            result = tf.reshape(tree_tensor, (x, y, feature_size))
            coef = tf.reshape(coef, (x, y, 3))
            result = tf.matmul(result, coef, transpose_a=True)
            result = tf.reshape(result, (batch_size, max_tree_size, 3, feature_size))

            # output is (batch_size, max_tree_size, output_size)
            result = tf.tensordot(result, weights, [[2, 3], [0, 1]])

            # output is (batch_size, max_tree_size, output_size)
            return tf.nn.tanh(result + b_conv, name='conv')


def pooling_layer(nodes):
    """Creates a max dynamic pooling layer from the nodes."""
    with tf.name_scope("pooling"):
        pooled = tf.reduce_max(nodes, axis=1)
        return pooled


def position_encoding(inputs, position_size):
    batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
    position_j = 1. / tf.pow(10000.,
                             2 * tf.range(position_size / 2, dtype=tf.float32
                                          ) / position_size)
    position_j = tf.expand_dims(position_j, 0)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    position_i = tf.expand_dims(position_i, 1)
    position_ij = tf.matmul(position_i, position_j)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)

    position_embedding = tf.expand_dims(position_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_size))
    return tf.add(inputs, position_embedding)


# 双向循环神经网络
def bi_directional_rnn_layer(batch_subtree_embeddings, subtree_len):
    with tf.variable_scope("encoder"):
        encoder_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(rnn_size / 2)
        encoder_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(rnn_size / 2)

        # 添加dropout
        encoder_cell_fw = tf.contrib.rnn.DropoutWrapper(encoder_cell_fw)
        encoder_cell_bw = tf.contrib.rnn.DropoutWrapper(encoder_cell_bw)

        encoder_outputs, encode_state = tf.nn.bidirectional_dynamic_rnn(
            encoder_cell_fw, encoder_cell_bw, batch_subtree_embeddings, subtree_len, dtype=tf.float32)
        # 将两个LSTM的输出拼接为一个向量
        encoder_outputs = tf.concat([encoder_outputs[0], encoder_outputs[1]], -1)

        code_vector = tf.reduce_max(encoder_outputs, axis=1)

    return code_vector

# 单向RNN


def multi_rnn_layer(batch_subtree_embeddings, subtree_len):
    stacked_rnn = []
    for i in range(2):
        stacked_rnn.append(tf.contrib.rnn.BasicLSTMCell(rnn_size))
    lstm_multi = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
    output, state = tf.nn.dynamic_rnn(lstm_multi, batch_subtree_embeddings,
                                      sequence_length=subtree_len,
                                      dtype=tf.float32)

    code_vector = tf.reduce_max(output, axis=1)

    return code_vector

# 直接最大层池化


def max_pooling(batch_subtree_embeddings):
    code_vector = tf.reduce_max(batch_subtree_embeddings, axis=1)
    with tf.name_scope("max_pooling_layer"):
        weights = tf.Variable(
            tf.truncated_normal(
                [num_conv*tree_embedding_size, code_vector_size], stddev=1.0 / math.sqrt(code_vector_size)
            ),
            name='weights'
        )

        init = tf.truncated_normal([code_vector_size, ], stddev=math.sqrt(2.0 / code_vector_size))
        biases = tf.Variable(init, name='biases')

        return tf.nn.tanh(tf.matmul(code_vector, weights) + biases)
    
def softmax_layer(code_vector, code_vector_size, label_size):
    """Create a hidden feedforward layer."""
    with tf.name_scope("softmax_hidden_layer"):
        weights = tf.Variable(
            tf.truncated_normal(
                [code_vector_size, label_size], stddev=1.0 / math.sqrt(code_vector_size)
            ),
            name='weights'
        )

        init = tf.truncated_normal([label_size, ], stddev=math.sqrt(2.0 / code_vector_size))
        biases = tf.Variable(init, name='biases')

        return tf.nn.tanh(tf.matmul(code_vector, weights) + biases)


def loss_layer(logits_node, label_size):
    """Create a loss layer for training."""

    labels = tf.placeholder(tf.int32, (None, label_size,))

    with tf.name_scope('loss_layer'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits_node, name='cross_entropy'
        )

        loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        return labels, loss


def out_layer(logits_node):
    """Apply softmax to the output layer."""
    with tf.name_scope('output'):
        return tf.nn.softmax(logits_node)
