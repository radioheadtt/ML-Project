# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:04:19 2021

@author: 孙亚非
"""

import pandas as pd
import os
import re
import collections
import numpy as np
import torch
import nltk
import torch.nn.functional as F

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "ve", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]     


class Vocab:
    """token is word, ukn is unkown, """
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens=[]
        #listed in order by frequency
        counter=count_corpus(tokens)
        self.token_freqs=sorted(counter.items(),key=lambda x: x[1],reverse=True)
        
        #index for unk token is 0
        self.unk,uniq_tokens=0,['<unk>'] + reserved_tokens
        uniq_tokens+=[token for token,freq in self.token_freqs
                      if (freq>=min_freq and token not in uniq_tokens)]
                          
        #a map from index to token, and a map from token to index
        self.idx_to_token,self.token_to_index=[],dict()        
        for token in uniq_tokens:
            self.idx_to_token.append(token)     #e.g.，idx_to_token[1]='the'
            self.token_to_index[token]=len(self.idx_to_token)-1     #e.g.，token_to_idx['the']='1'
    
    #class function                    
    def __len__(self):  #测量token的数量
        return len(self.idx_to_token)
    
    def __getitem__(self,tokens):
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_index.get(tokens,self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
                

    
def count_corpus(tokens):   #统计token的频率
    #tokens is a 1D or 2D list
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]
    return collections.Counter(tokens)

def data_clean():
    path=r'.\train_data_all'
    path_w=r'.\cleaned_data'
    file_dir = os.listdir(path)
    oneletter='[^a-zA-Z]'
    contents=[]  #此列表中元素为一条条评价
    df_final = pd.DataFrame(columns=['CommentsStars', 'CommentsContent'])
    for file in file_dir:
        if not os.path.isdir(file):     #make sure it's a file not a directory
            file_name=os.path.join(path,file)
            cleaned_file_name=os.path.join(path_w,file)
            raw_data=pd.read_csv(file_name,encoding= 'utf-8').dropna()
            stars=raw_data['CommentsStars'].tolist()            
            content=list(map(lambda x:re.sub(oneletter," ",x).lower(),raw_data['CommentsContent'].astype(str).tolist()))                 
            contents.append(content)
            dic_data={'CommentsStars':stars,'CommentsContent':content}
            data=pd.DataFrame(dic_data)
            data.to_csv(cleaned_file_name)
            df_final=pd.concat([df_final,data],ignore_index=True)
        else:
            print('Error: directory found')    
    return df_final     #一个包含所有评价的大df，star的数据类型为int
    
 
    
def tokenize(df):   #返回一个列表，其中元素是分词化的各条评价
    comments=[]        
    for line in df['CommentsContent']:        
        words=line.split(' ')
        words = [word for word in words if(len(str(word))!=0) and word not in stop_words]
        if len(words) < 12:
            words=words+['<unk>']*(12-len(words))
        comments.append(words)    
    return comments    #e.g.,[['very','good'],['not','bad']]


def tokenize_star(df):   #返回一个列表，其中元素是分词化的各条评价以及星级
    comments_with_stars=[]
    for row in df.iterrows():
        temp=[]
        temp.append([i for i in row['CommentsContent'].split(' ') if(len(str(i))!=0 and i not in stop_words)][:12])
        if len(temp) < 12:
            temp=temp+['<unk>']*(12-len(temp))
        temp.append(row['CommentsStars'])
        comments_with_stars.append(temp)
    return comments_with_stars     #e.g.,[[['very','good'],5],[['not','bad'],3]]


def load_corpus(df,max_tokens=-1):     #整合
    tokens=tokenize(df)
    vocab=Vocab(tokens)
    corpus=[vocab[token] for line in tokens for token in line]
    if max_tokens>0:
        corpus=corpus[:max_tokens]
    return corpus,vocab


def comment_to_idx(lis,dic):   #the list is comments_with_stars and the dict is token_to_index
    tens_with_5, tens_with_4, tens_with_3, tens_with_2, tens_with_1 = [], [], [], [], []
    for line in lis:
        temp=[dic[word] for word in line[0] if word in dic]    
        temp=torch.tensor(temp)
        if line[1]==1:
            tens_with_1.append([temp,line[1]])   
        elif line[1]==2:
            tens_with_2.append([temp,line[1]])
        elif line[1]==3:
            tens_with_3.append([temp,line[1]])
        elif line[1]==4:
            tens_with_4.append([temp,line[1]])
        elif line[1]==5:
            tens_with_5.append([temp,line[1]])
    return tens_with_5, tens_with_4, tens_with_3, tens_with_2, tens_with_1

if __name__ == '__main__':
    df=data_clean()
    corpus,vocab=load_corpus(df)
    comments_with_stars=tokenize_star(df)
    token_to_index=vocab.token_to_index
    tens_with_5, tens_with_4, tens_with_3, tens_with_2, tens_with_1=comment_to_idx(comments_with_stars,token_to_index)
    
    