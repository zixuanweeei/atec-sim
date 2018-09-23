# coding: utf-8
import pandas as pd
import jieba
from tqdm import tqdm
from collections import defaultdict
jieba.load_userdict('../data/user_dict.txt')

#%% load training set 
data1 = pd.read_csv('../data/atec_nlp_sim_train.csv', header=None, sep='\t', index_col=0)
data2 = pd.read_csv('../data/atec_nlp_sim_train_add.csv', header=None, sep='\t', index_col=0)
data = pd.concat([data1, data2], axis=0)
data.columns = ['s1', 's2', 'label']
vector_dim = 300

#%% read baidubaike word2vec
# wv = defaultdict(lambda : [0 for _ in range(vector_dim)])
# with open("../data/sgns.baidubaike.bigram-char", "r", encoding="utf-8") as reader:
#     line = reader.readline().split(" ")
#     num = int(line[0])
#     dim = int(line[1])
#     for line in tqdm(reader.readlines(), total=num, desc="Embedding reading:", ascii=True):
#         ele = line.split(" ")
#         word = ele[0]
#         vector = list(map(float, ele[1:-1]))
#         if len(vector) == dim:
#             wv[word] = vector
#         else:
#             raise RuntimeError("Dimension doesn't match.")

#%% concat question pairs to individual sentences, do segmentatin and create corporas
s1 = data.loc[:, 's1']
s2 = data.loc[:, 's2']
strs = pd.concat([s1, s2])
strs.reset_index(drop=True, inplace=True)
del s1, s2

words = {}
num_words = 0
new_words = []
with open("../data/corpus.txt", "w+", encoding="utf-8") as writer:
    for idx, line in tqdm(strs.iteritems(), desc="Corporas making:", total=strs.shape[0], ascii=True):
        seg_list = jieba.cut(line.strip().replace(" ", ""))
        writer.write(" ".join(seg_list) + "\n")
#         for item in seg_list:
#             if item not in words:
#                 if item not in wv:
#                     new_words.append(item)
#                 words[item] = wv[item]
#                 num_words += 1

# with open("../data/local_corpora.txt", "w+", encoding="utf-8") as corpora:
#     corpora.write("{} {}\n".format(num_words, vector_dim))
#     for word, vector in tqdm(words.items(), desc="Writing corpora:", total=num_words, ascii=True):
#         line = word + " " + " ".join(map(str, vector))
#         corpora.write(line + "\n")

# with open("../data/new_words.txt", "w+", encoding="utf-8") as writer:
#     for word in new_words:
#         writer.write(word + "\n")
