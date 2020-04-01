import _pickle as pkl
from utils import generateCodeMatrix,draw2DPicture,sampleSearchVector,searchCode
import numpy as np 
if __name__ == "__main__":
    node_embedding_path = "dataset/embeddings.pkl"
    with open(node_embedding_path, "rb") as f:
        data = pkl.load(f)
        vector = data[0]

        vector_lookup = data[1]

    log_dir = "log/train"

    with open('dataset/code_vector.pkl','rb') as f:
        dataset=pkl.load(f)
        code_db=dataset[0]
        label_db=dataset[1]

    
    code_db=np.vstack(code_db)

    query_vector,query_label=sampleSearchVector(code_db,label_db)

    top1,top3,top5,top10=searchCode(query_vector,query_label,code_db,label_db)
    
    print(top1)
    print(top3)
    print(top5)
    print(top10)

    



