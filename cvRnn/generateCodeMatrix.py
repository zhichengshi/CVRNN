from utils import generateCodeMatrix
from sampling import generatePositiveSample,generateNegativeSample
import _pickle as pkl 
import numpy as np 
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

    generateCodeMatrix(postive_path,log_dir,vector,vector_lookup,postive_dump_path,generatePositiveSample)

    # generateCodeMatrix(negative_path1,log_dir,vector,vector_lookup,negative_dump_path1,generateNegativeSample) #!!!内存太小，只能分两批
    # generateCodeMatrix(negative_path2,log_dir,vector,vector_lookup,negative_dump_path2,generateNegativeSample)

    # with open(negative_dump_path1,'rb') as f:
    #     negative1=pkl.load(f)

    # with open(negative_dump_path2,'rb') as f:
    #     negative2=pkl.load(f)

    # vectors=list(negative1[0])+list(negative2[0])
    # labels=negative1[1]+negative2[1]

    # with open(negative_dump_path,'wb') as f:
    #     pkl.dump((vectors,labels),f)

 