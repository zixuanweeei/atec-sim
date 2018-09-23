# coding: utf-8
import os
import jieba
import pandas as pd
from tqdm import tqdm
jieba.load_userdict('../data/user_dict.txt')

data1 = pd.read_csv("../data/train.csv", header=None, index_col=0, sep="\t")
data2 = pd.read_csv("../data/eval.csv", header=None, index_col=0, sep="\t")
data = pd.concat([data1, data2])
data.columns = ["s1", "s2", "label"]
strs = pd.concat([data.loc[:, "s1"], data.loc[:, "s2"]])

words = set()
for idx, string in tqdm(strs.iteritems(), desc="Creating dictionary", ascii=True, total=strs.shape[0]):
    words |= set(list(string))
    words |= set(jieba.cut(string.strip(), cut_all=True, HMM=False))

words.remove(" ")
all_words = words.copy()
# DIR = r"E:\WZXWork\wordvector"
# pretrained_vocab = "cc.zh.300.vec"
# target_vocab = []
# with open(os.path.join(DIR, pretrained_vocab), "r", encoding="utf-8") as p:
#     info = p.readline().split()
#     vocab_size = int(info[0])
#     embed_dim = int(info[1])
#     for _ in tqdm(range(vocab_size), desc="Reading vocab", ascii=True, total=vocab_size):
#         line = p.readline()
#         word = line.split(" ")[0]
#         if word in words:
#             words.remove(word)
#             target_vocab.append(line)
#         elif len(word) == 1:
#             target_vocab.append(line)
    
# with open("../data/pretrianed_vocab.txt", "w+", encoding="utf-8") as t:
#     t.write("{0} {1}\n".format(len(target_vocab), embed_dim))
#     for line in target_vocab:
#         t.write(line)

# with open("../data/oov.txt", "w+", encoding="utf-8") as o:
#     for ele in words:
#         o.write(ele + "\n")

with open("../data/all_words.txt", "w+", encoding="utf-8") as all_w:
    for word in words:
        all_w.write(word + "\n")