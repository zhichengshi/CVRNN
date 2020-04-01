import _pickle as pkl
from utils import generateCodeMatrix,draw2DPicture,sampleSearchVector,searchCode
import numpy as np 
if __name__ == "__main__":
    node_embedding_path = "dataset/embeddings.pkl"
    with open(node_embedding_path, "rb") as f:
        data = pkl.load(f)
        vector = data[0]

        vector_lookup = data[1]

    dataset_path = "dataset/104/test.pkl"
    log_dir = "log/train"
    
    generateCodeMatrix(dataset_path,log_dir,vector,vector_lookup)

