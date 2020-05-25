"""Functions to help with sampling trees."""

import pickle
import numpy as np
from cvRnn.extractStatement import extractSTBaseRoot
import random
from cvRnn.config import label_size
import gc
from tqdm import tqdm
'''
trees: Statement AST list
vector:node embedding matrix
vector_lookup:search the node num
'''

def traverse(root):
    if root.tail != None:
        if root.text != None:
            print(root.text, end="")
        for node in root:
            traverse(node)
        print(root.tail, end="")
    else:
        if root.text != None:
            print(root.text, end="")
        for node in root:
            traverse(node)

# 统计节点类型的数目
def node_type_num(root, node_type_set):
    node_type_set.add(root.tag)
    for child in root:
        node_type_num(child, node_type_set)

    return len(node_type_set)


# 统计树中非叶子节点的个数
def statistic_tree_nodes(root):
    queue = []
    queue.append(root)

    count = 0
    while len(queue) > 0:
        root = queue.pop(0)
        count += 1
        for node in root:
            queue.append(node)
    return count


def processTree(trees, vectors, vector_lookup):
    nodes_batch = []
    children_batch = []
    for tree in trees:
        nodes = []
        children = []

        queue = [(tree, -1)]

        # level visit
        while queue:
            node, parent_ind = queue.pop(0)

            str = "http://www.srcML.org/srcML/cpp"
            if node.tag.find(str) > 0:
                continue

            # neglect the comment node
            if node.tag == "comment":
                continue

            node_index = len(nodes)
            # add children and the parent index to the queue
            queue.extend([(child, node_index) for child in node])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_index)
            nodes.append(vectors[vector_lookup[node.tag]])

        nodes_batch.append(nodes)
        children_batch.append(children)

    return _pad(nodes_batch, children_batch)

def generatePositiveSample(path, vectors, vector_lookup):
    with open(path, "rb") as f:
        datas = pickle.load(f)
        random.shuffle(datas)

        for data in tqdm(datas):
            label = data[0]
            tree = data[1]

            subtrees = extractSTBaseRoot(tree)
            # process one tree which has been splited into statement trees
            nodes, children,max_children_size = processTree(subtrees, vectors, vector_lookup)

            # 根据标记获得对应于该标记的向量
            label_vector = np.eye(label_size, dtype=int)[int(label) - 1]

            yield [nodes], [children], [len(subtrees)], [label_vector]

def generateNegativeSample(path, vectors, vector_lookup):
    with open(path, "rb") as f:
        datas = pickle.load(f)
        random.shuffle(datas)

        for data in tqdm(datas):
            label = 101 #!!!负样本的标记均设为101
            tree = data[1]

            subtrees = extractSTBaseRoot(tree)
            # process one tree which has been splited into statement trees
            nodes, children,max_children_size = processTree(subtrees, vectors, vector_lookup)

            # 根据标记获得对应于该标记的向量
            label_vector = np.eye(label_size, dtype=int)[int(label) - 1]

            yield [nodes], [children], [len(subtrees)], [label_vector]



#
def _pad(nodes, children):
    if not nodes:
        return [], [], []
    # 树的最多节点个数
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    feature_len = len(nodes[0][0])

    # 最大孩子节点的个数
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children,max_children
