from model import createGraph,findHighestSimilarityRank
import h5py
from tqdm import tqdm
from compute_rouge import get_score,get_score_all
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# file_path = "../dataset/nyt.test.h5df"
# f = h5py.File(file_path,'r')
# for b in tqdm(f['dataset']):
#     string = str(b)
#     data = eval(string)
#     article = data["article"]
#     abstract = data["abstract"]
# 不同的sbert 采取mean-tokens
# bert-base-nli-mean-tokens
# roberta-base-nli-stsb-mean-tokens
# distilbert-base-nli-stsb-mean-tokens
# bert-base-nli-stsb-mean-tokens
model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens',device=torch.device('cuda'))  # The sentence transform models mentioned above can be used.

if __name__ == '__main__':
    article_list = []
    abstract_list = []
    ans_list = []

    file_path = "../dataset/cnndm.test.h5df"
    f = h5py.File(file_path,'r')
    for b in tqdm(f['dataset']):

        string = str(b,'utf-8')
        data = eval(string)
        article = data["article"]
        abstract = data["abstract"]
        asjdoidaj = data['oracle_sens']
        if (len(article) < 3):
            continue
        article_list.append(article)
        abstract_list.append(abstract)
        sentence_embeddings = model.encode(article)
        sentenceGraph = createGraph(model,sentence_embeddings)
        newRank = findHighestSimilarityRank(sentenceGraph,[1.0]*len(article),True)
        ans = np.argpartition(newRank, -3)[-3:]
        score = get_score(article, abstract, ans)
        ans_list.append(ans)

    print(get_score_all(article_list,abstract_list,ans_list))


