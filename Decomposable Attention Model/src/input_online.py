# coding:utf-8
import pandas as pd
import numpy as np
import sys

def topai(result):
    print(result)
print(sys.argv)
if "run_transformer.py" in sys.argv[0]:
    model_dir = "./model_ckp"
elif "run.py" in sys.argv[0]:
    model_dir = "./model_test"
elif "input_test.py" in sys.argv[0]:
    model_dir = "test"
print("Save model to ", model_dir)
df1 = pd.read_csv("./data/train.csv", header=None, sep="\t")
df1.columns = ["id", "sent1", "sent2", "label"]
wv_cols = ["sent1"]
wv_cols.extend(["f" + str(idx) for idx in range(300)])
word = []
with open("./data/local_wv.txt", "r", encoding="utf-8") as reader:
    info = reader.readline().split(" ")
    words_num = int(info[0])
    vector_dim = int(info[1])
    W = np.zeros((words_num, vector_dim))
    for idx, line in enumerate(reader.readlines()):
        ele = line.rstrip().split(" ")
        W[idx, ] = list(map(float, ele[1:]))
        word.append(ele[0])
word = pd.DataFrame({"sent1": word})
vectors = pd.DataFrame(W)
df2 = pd.concat([word, vectors], axis=1)
df2.columns= wv_cols
df3 = pd.read_csv("./data/pos_tag.txt", header=None)
df3.columns = ["pos"]

if __name__ == "__main__":
    print(df1.head(2))
    print(df2.head(2))
    print(df3.head(2))