import sys
sys.path.append('./')

import _pickle as pkl
from cvRnn.utils import searchCode,printMetric,mixNegativePositive
import numpy as np

if __name__ == "__main__":

    with open('dataset/matrix/cvrnn/positive.pkl', "rb") as f:
        dataset = pkl.load(f)
        positive_vectors = dataset[0]
        positive_labels = dataset[1]

    with open('dataset/matrix/tbcnn/negative.pkl', 'rb') as f:
        dataset = pkl.load(f)
        negative_vectors = dataset[0]
        negative_labels = dataset[1]

    query_vectors, query_labels, db_vectors, db_labels = mixNegativePositive(positive_vectors, positive_labels, negative_vectors, negative_labels)

    predict_label=searchCode(query_vectors,query_labels,db_vectors,db_labels)
    printMetric(query_labels,predict_label)


