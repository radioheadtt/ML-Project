from base import Operator
import numpy as np
import torch.nn.functional as F
glove_dir = 'C:/Users/Radiohead/Desktop/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector =  embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
class embedding():
    def __init__(self,weight,input_dim,embedding_dim):
        self.weight=weight
        self.input_dim=input_dim
        self.embedding_dim=embedding_dim
    def __call__(self,x):
        assert x.shape[1]=input_dim
        vec=torch.zeros((self.shape[0],input_dim,embedding_dim))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i][j]!=0:
                    vec[i][j]=self.weight[j]
        return vec

        
    