import _pickle as pkl 
import os 
import shutil
from utils import generateCodeMatrix,draw2DPicture

node_embedding_path = "dataset/matrix/embeddings.pkl"
with open(node_embedding_path, "rb") as f:
    data = pkl.load(f)
    vector = data[0]
    vector_lookup = data[1]

with open('dataset/matrix/positive.pkl','rb') as f:
    dataset=pkl.load(f)
    vectors=dataset[0]
    labels=dataset[1]


draw2DPicture(vectors,labels,len(labels))


