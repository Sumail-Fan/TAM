# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:08:34 2021

@author: TUBA
"""

import numpy as np
import math
from math import *

import torch

from decimal import Decimal
# 余弦距离
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



# 欧式距离
def euclidean_distance(x,y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
# 曼哈顿距离
def manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))

def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)
#
def minkowski_distance(x,y):
    p_value=3
    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)

def createGraph(sentence_embeddings):


    sentenceGraph =np.zeros((len(sentence_embeddings), len(sentence_embeddings)))
    temp = np.arange(len(sentence_embeddings))
    # print(sentence_embeddings[1].shape)
    for x in range(len(sentence_embeddings)):
        newTemp= np.delete(temp, x)
        for y in newTemp:
            similarity= cosine(sentence_embeddings[x],sentence_embeddings[y]) # You can change the vector similarity measurement method used when creating graphs. Cosine, euclidean, manhattan and minkowski methods are defined.
            sentenceGraph[x][y]=similarity
    return sentenceGraph


def findHighestSimilarityRank(similarityMatrix, initialRank,direction=False):
    newRank=[0] * len(similarityMatrix)
    temp=0
    lambda1 = -1.0
    lambda2 = 2.0
    for i in range(len(similarityMatrix)):
        for j in range(len(similarityMatrix)):
            if(direction):
                temp=temp+similarityMatrix[i][j]*(lambda1 if i>=j else lambda2)
            else:
                temp=temp+similarityMatrix[i][j] # sum of total similarity of sentences
        newRank[i]=temp*initialRank[i]
        temp=0

    return newRank

def findHighestSimilarityRankTopic(similarityMatrix, initialRank,weights):
    newRank=[0] * len(similarityMatrix)
    temp=0
    lambda1 = 0.2
    lambda2 = 0.8
    for i in range(len(similarityMatrix)):
        for j in range(len(similarityMatrix)):
            temp=temp+similarityMatrix[i][j]*(lambda1 if weights[i]<=weights[j] else lambda2)

        newRank[i]=temp*initialRank[i]
        temp=0

    return newRank