from base import Operator
import numpy as np
import torch.nn.functional as F
import torch
import os
import pre_processor
from pre_processor import data_clean,load_corpus
from torch import Tensor
def load_GloVe(vocab):
    glove_dir = 'C:/Users/Radiohead/Desktop/glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),'rb')
    for line in f:
        values = line.split()
        word = values[0].decode('UTF-8')
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_dim = 100
    max_words=10000
    word_index=vocab.idx_to_token
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for i, word in enumerate(word_index):
        if i < max_words:
            embedding_vector =  embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix,embeddings_index
class embedding():
    def __init__(self,weight,input_dim=16,embedding_dim=100):
        self.weight=weight
        self.input_dim=input_dim
        self.embedding_dim=embedding_dim
    def __call__(self,x):
        assert x.shape[1]==self.input_dim
        vec=torch.zeros((x.shape[0],self.input_dim,self.embedding_dim)).cuda()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i][j]!=0:
                    vec[i][j]=self.weight[j]
        return vec

if __name__ == '__main__':
    df=data_clean()
    corpus,vocab=load_corpus(df)
    embedding_matrix,embedding_index=load_GloVe(vocab)
        
    