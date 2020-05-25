import sys
sys.path.append('./')
from tqdm import tqdm
import _pickle as pkl
import numpy as np
from cvRnn.sampling import processTree, _pad
import random
from cvRnn.config import label_size



def generatePositiveSample(path, vectors, vector_lookup):
    with open(path, "rb") as f:
        datas = pkl.load(f)
        random.shuffle(datas)

    for data in tqdm(datas):
        label = data[0]
        tree = data[1]
        nodes, children, max_children_size = processTree([tree], vectors, vector_lookup)
        # 根据标记获得对应于该标记的向量
        label_vector = np.eye(label_size, dtype=int)[int(label) - 1]

        yield [nodes], [children], [label_vector]


def generateNegativeSample(path, vectors, vector_lookup):  # 提取负样本时所有label设为101
    with open(path, "rb") as f:
        datas = pkl.load(f)
        random.shuffle(datas)

    for data in tqdm(datas):
        label = 101
        tree = data[1]
        nodes, children, max_children_size = processTree([tree], vectors, vector_lookup)
        # 根据标记获得对应于该标记的向量
        label_vector = np.eye(label_size, dtype=int)[int(label) - 1]

        yield [nodes], [children], [label_vector]
