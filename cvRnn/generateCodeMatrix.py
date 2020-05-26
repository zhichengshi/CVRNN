from utils import generateCodeMatrix
from sampling import generatePositiveSample,generateNegativeSample
import _pickle as pkl 
import numpy as np 
import tensorflow as tf
import os 
if __name__ == "__main__":
    negative_path1='dataset/104/train/train1.pkl'
    negative_path2='dataset/104/train/train2.pkl'
    negative_dump_path='dataset/matrix/cvrnn/negative.pkl'
    negative_dump_path1='dataset/matrix/cvrnn/negative1.pkl'
    negative_dump_path2='dataset/matrix/cvrnn/negative2.pkl'


    postive_path='dataset/leetcodeCorpus.pkl'
    postive_dump_path='dataset/matrix/cvrnn/positive.pkl'
    
    log_dir='log/cvrnn'
    node_path = "dataset/embeddings/word2vec_embeddings.pkl"

    node_path = "dataset/embeddings/word2vec_embeddings.pkl"
    with open(node_path, "rb") as f:
        data = pkl.load(f)
        vector = data[0]
        vector_lookup = data[1]

    #!!!因为验证集中的数据量很少，仅有1000条，这里仅使用训练集中的数据生成负样本中的代码向量，所以将训练集拆分为两部分生成负样本的代码向量，由于电脑内存原因，训练集数据不能
    #!!!一次全装进内存，因此分两批生成代码向量，最终合并这两个负样本代码向量数据库

    # 根据leetcode中的数据生成正样本代码向量库
    generateCodeMatrix(postive_path,log_dir,vector,vector_lookup,postive_dump_path,generatePositiveSample)

    # 根据104中的数据生成负样本代码库
    generateCodeMatrix(negative_path1,log_dir,vector,vector_lookup,negative_dump_path1,generateNegativeSample)
    generateCodeMatrix(negative_path2,log_dir,vector,vector_lookup,negative_dump_path2,generateNegativeSample)

    # 合并两个负样本代码库
    with open(negative_dump_path1,'rb') as f:
        negative1=pkl.load(f)

    with open(negative_dump_path2,'rb') as f:
        negative2=pkl.load(f)

    vectors=list(negative1[0])+list(negative2[0])
    labels=negative1[1]+negative2[1]

    os.remove(negative_dump_path1)
    os.remove(negative_dump_path2)

    #将合并后的负样本代码库写出
    with open(negative_dump_path,'wb') as f:
        pkl.dump((vectors,labels),f)

 