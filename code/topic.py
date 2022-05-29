from model import createGraph,findHighestSimilarityRankTopic
import h5py
from tqdm import tqdm
from compute_rouge import get_score,get_score_all
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os
from umap import UMAP
from sklearn.cluster import KMeans
from model import cosine
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
        if (len(article) <= 8):
            continue
        article_list.append(article)
        abstract_list.append(abstract)
        sentence_embeddings = model.encode(article)
        sentenceGraph = createGraph(model, sentence_embeddings)
        umap_model = UMAP(n_neighbors=15, n_components=3, min_dist=0.0, metric='cosine')

        reduced_embedding = umap_model.fit_transform(sentence_embeddings)

        clu_num = 8 if len(article)>=8 else len(article)
        k_means = KMeans(n_clusters=clu_num)
        k_means.fit(reduced_embedding)

        y_predict = k_means.predict(reduced_embedding)

        count = [0] * clu_num
        for i in y_predict:
            count[i] += 1
        index_list = np.argpartition(count, -3)[-3:]
        centers = k_means.cluster_centers_

        centers_less = [centers[i] for i in index_list]
        weights = []
        for i in range(len(article)):
            # dis_list = []
            # for center in centers_less:
            #     dis_list.append(cosine(np.array(reduced_embedding[i]), np.array(center)))
            # weights.append(max(dis_list))

            weights.append(cosine(np.array(reduced_embedding[i]),np.array(centers[index_list[-1]])))


        newRank = findHighestSimilarityRankTopic(sentenceGraph,[1.0]*len(article),weights)
        ans = np.argpartition(newRank, -3)[-3:]
        # score = get_score(article,abstract,ans)
        ans_list.append(ans)

    print(get_score_all(article_list,abstract_list,ans_list))


