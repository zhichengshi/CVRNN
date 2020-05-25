from utils import draw2DPicture
import _pickle as pkl

path='dataset/matrix/cvrnn/positive.pkl'
with open(path,'rb') as f:
    dataset=pkl.load(f)
    labels=dataset[1]
    embeddings=dataset[0]

draw2DPicture(embeddings,labels,len(embeddings))
