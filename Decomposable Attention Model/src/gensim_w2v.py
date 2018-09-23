# coding: utf-8
import argparse
import os

import pandas as pd
import jieba
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors

def main():
    jieba.load_userdict(os.path.join(FLAGS.dataspace, FLAGS.user_dict))

    # load training set
    datafile = os.path.join(FLAGS.dataspace, FLAGS.train_filename)
    data = pd.read_csv(datafile, header=None, sep='\t', index_col=0)
    data.columns = ['s1', 's2', 'label']

    # concat question pairs to individual sentences, do segmentatin and create corporas
    s1 = data.loc[:, 's1']
    s2 = data.loc[:, 's2']
    strs = pd.concat([s1, s2])
    strs.reset_index(drop=True, inplace=True)
    del s1, s2

    sentences = []
    for idx, line in tqdm(strs.iteritems(), desc="Corporas making:", total=strs.shape[0], ascii=True):
        seg_list = list(jieba.cut(line.strip().replace(" ", "")))
        sentences.append(seg_list)

    # word_vectors = KeyedVectors.load_word2vec_format("../data/local_corpora.txt", binary=False)

    model = Word2Vec(size=300, window=5, min_count=1, workers=20)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=10000)
    model.save(os.path.join(FLAGS.dataspace, FLAGS.wordvector_name + ".bin"))
    model.wv.save_word2vec_format(os.path.join(FLAGS.dataspace, FLAGS.wordvector_name + ".txt"), binary=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--dataspace",
        type=str,
        default="./data",
        help="The directory to store the data.")
    parser.add_argument("--user_dict",
        type=str,
        default="user_dict.txt",
        help="Custom dict passed to jieba.")
    parser.add_argument("--train_filename",
        type=str,
        default="train.csv",
        help="Raw training dataset.")
    parser.add_argument("--wordvector_name",
        type=str,
        default="local_wv",
        help="File to store word2vector result.")

    FLAGS, unparsed = parser.parse_known_args()
    main()