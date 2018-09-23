# coding: utf-8
""" 对数据进行预处理

包括清洗数据、连接、预处理、分割等步骤
> 清洗数据：人工对一些错别字、同义词、无用字符进行处理
> 连接：将多个数据文件组合到一起
> 预处理：目前还不知道怎么预处理
> 分割：划分训练集和验证集
"""
import argparse
import codecs
import io
import os

import pandas as pd
from sklearn.utils import shuffle

def concat():
    if len(FLAGS.file_list) == 1:
        return pd.read_csv(os.path.join(FLAGS.dataspace, FLAGS.file_list[0]), header=None, sep='\t', index_col=0)
    else:
        data_frames = []
        for data_file in FLAGS.file_list:
            data_frames.append(pd.read_csv(os.path.join(FLAGS.dataspace, data_file), header=None, sep='\t', index_col=0))
        data = pd.concat(data_frames, axis=0).reset_index(drop=True)
        if data.shape[1] == 3:
            data.columns = ['s1', 's2', 'label']
        else:
            data.columns = ["s1", "s2"]
        data.index += 1
    
        return data

def clean():
    for file in FLAGS.file_list:
        with io.open(os.path.join(FLAGS.dataspace, file), "r", encoding="utf-8") as wr:
            strings = wr.read()
            strings = strings.replace(codecs.BOM_UTF8.decode("utf-8"), "")\
                             .replace(u"蚂蚁", u"")\
                             .replace(u"花别", u"花呗")\
                             .replace(u"花被", u"花呗")\
                             .replace(u"借别", u"借呗")\
                             .replace(u"余利宝", u"余额宝")\
                             .replace(u"小蓝车", u"小蓝")\
                             .replace(u"唄", u"呗")\
                             .replace(u"花贝", u"花呗")\
                             .replace(u"花臂", u"花呗")\
                             .replace(u"花坝", u"花呗")\
                             .replace(u"卷", u"券")\
                             .replace(u"考拉海购", u"考拉")\
                             .replace(u"网易考拉", u"考拉")\
                             .replace(u"哈罗单车", u"哈罗")\
                             .replace(u"hellobike", u"哈罗")\
                             .replace(u"单车", "")\
                             .replace(u"天猫超市", u"天猫")\
                             .replace(u"天猫商城", u"天猫")
        with io.open(os.path.join(FLAGS.dataspace, file), "w", encoding="utf-8") as wr:
            wr.write(strings)

def preprocess(DataFrame):
    return DataFrame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("file_list", 
        nargs="+",
        metavar="N",
        type=str,
        help="The files to be concatenated and preprocessed.")
    parser.add_argument("-f", "--filename",
        type=str,
        default="train.csv",
        help="The file to store all of the data.")
    parser.add_argument("-o", "--eval_filename",
        type=str,
        default="eval.csv",
        help="The file to store all of the data.")
    parser.add_argument("-d", "--dataspace",
        type=str,
        default="./data",
        help="The file to store all of the data.")
    parser.add_argument("-t", "--task",
        type=str,
        default="normal",
        help="The file to store all of the data.")
    parser.add_argument("-k", "--kfold",
        type=int,
        default=5,
        help="The file to store all of the data.")
    
    FLAGS, unparsed = parser.parse_known_args()
    clean()
    if FLAGS.task is not "predict":
        data = concat()
        data = preprocess(data)
        if FLAGS.task == "normal":
            print("ALL ...")
            data.to_csv(os.path.join(FLAGS.dataspace, FLAGS.filename), header=False, index=True, sep="\t")
        elif FLAGS.task == "split":
            data = shuffle(data, random_state=47)
            split_idx = data.shape[0] * 4//5
            train = data.iloc[:split_idx, :].reset_index(drop=True)
            evaluation = data.iloc[split_idx:, :].reset_index(drop=True)
            train.index += 1
            evaluation.index += 1

            train.to_csv(os.path.join(FLAGS.dataspace, FLAGS.filename), header=False, index=True, sep="\t")
            evaluation.to_csv(os.path.join(FLAGS.dataspace, FLAGS.eval_filename), header=False, index=True, sep="\t")
        elif FLAGS.task == "kfold":
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=FLAGS.kfold, random_state=47)
            for idx, (train_index, test_index) in enumerate(kf.split(data)):
                data.iloc[train_index, :].to_csv("./data/train_{}.csv".format(idx), header=False, index=True, sep="\t")
                data.iloc[test_index, :].to_csv("./data/eval_{}.csv".format(idx), header=False, index=True, sep="\t")
        