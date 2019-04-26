# conding: utf-8
import argparse
import os
import codecs
import io
import sys
from collections import defaultdict

import pandas as pd
import numpy as np
import tensorflow as tf
# from tqdm import tqdm
import jieba
import jieba.posseg as pseg

def word2idx(in_file, out_file):
    with codecs.open(out_file, "w+", "utf-8") as writer, \
        codecs.open(in_file, "r", "utf-8") as reader:
        info = reader.readline().split(" ")
        words_num = int(info[0])
        vector_dim = int(info[1])
        for idx, line in enumerate(reader.readlines()):
            item = line.split(" ")[0]
            writer.write("{} {}\n".format(item, idx + 1))

def get_wi():
    wi = defaultdict(lambda : 0)
    wi_file = os.path.join(FLAGS.dataspace, FLAGS.wi)
    with io.open(wi_file, "r", encoding="utf-8") as reader:
        for line in reader.readlines():
            info = line.split(" ")
            wi[info[0]] = int(info[1])
    pos = defaultdict(lambda : 0)
    with io.open(os.path.join(FLAGS.dataspace, "POS_tag.txt"), "r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader.readlines()):
            pos[line.rstrip()] = idx + 1

    return wi, pos

def string2idxs(string, wi):
    s = []
    for word in jieba.cut(string):
        if word in wi:
            s.append(wi[word])
        else:
            for char in list(word):
                if char in wi:
                    s.append(wi[char])
                else:
                    s.append(0)
    
    return np.array(s)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def sequence2tfrecords(csv_file, records_file, train=False):
    data = pd.read_csv(csv_file, header=None, sep='\t', index_col=0)
    if train:
        data.columns = ['s1', 's2', 'label']
    else:
        if data.shape[1] == 3:
            data = data.iloc[:, :2]
        data.columns = ['s1', 's2']
    
    wi, posi = get_wi()
    with tf.python_io.TFRecordWriter(records_file) as writer:
        if train:
            for idx, row in data.iterrows():
                words1 = pseg.cut(row['s1'], HMM=False)
                words2 = pseg.cut(row['s2'], HMM=False)
                s1 = [(wi[word], posi[flag]) for word, flag in words1 if word is not " "]
                s2 = [(wi[word], posi[flag]) for word, flag in words2 if word is not " "]
                # s1 = string2idxs(row["s1"], wi)
                # s2 = string2idxs(row["s2"], wi)
                len1 = len(s1)
                len2 = len(s2)
                label = row['label']

                ex = tf.train.SequenceExample()
                ex.context.feature['label'].int64_list.value.append(label)
                ex.context.feature['len1'].int64_list.value.append(len1)
                ex.context.feature['len2'].int64_list.value.append(len2)

                s1_tokens = ex.feature_lists.feature_list['s1']
                s2_tokens = ex.feature_lists.feature_list['s2']
                s1_pos = ex.feature_lists.feature_list['pos1']
                s2_pos = ex.feature_lists.feature_list['pos2']

                for token, pos in s1:
                    s1_tokens.feature.add().int64_list.value.append(token)
                    s1_pos.feature.add().int64_list.value.append(pos)
                for token, pos in s2:
                    s2_tokens.feature.add().int64_list.value.append(token)
                    s2_pos.feature.add().int64_list.value.append(pos)

                writer.write(ex.SerializeToString())
        else:
            for idx, row in data.iterrows():
                words1 = pseg.cut(row['s1'], HMM=False)
                words2 = pseg.cut(row['s2'], HMM=False)
                s1 = [(wi[word], posi[flag]) for word, flag in words1 if word is not " "]
                s2 = [(wi[word], posi[flag]) for word, flag in words2 if word is not " "]
                # s1 = string2idxs(row["s1"], wi)
                # s2 = string2idxs(row["s2"], wi)
                len1 = len(s1)
                len2 = len(s2)

                ex = tf.train.SequenceExample()
                ex.context.feature['len1'].int64_list.value.append(len1)
                ex.context.feature['len2'].int64_list.value.append(len2)

                s1_tokens = ex.feature_lists.feature_list['s1']
                s2_tokens = ex.feature_lists.feature_list['s2']
                s1_pos = ex.feature_lists.feature_list['pos1']
                s2_pos = ex.feature_lists.feature_list['pos2']

                for token, pos in s1:
                    s1_tokens.feature.add().int64_list.value.append(token)
                    s1_pos.feature.add().int64_list.value.append(pos)
                for token, pos in s2:
                    s2_tokens.feature.add().int64_list.value.append(token)
                    s2_pos.feature.add().int64_list.value.append(pos)

                writer.write(ex.SerializeToString())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--task',
        type=str,
        default='test',
        help='Choose tasks')
    parser.add_argument('--filename',
        type=str,
        default='input',
        help='The filename that used to convert to training and test set')
    parser.add_argument('--wi',
        type=str,
        default='word2idx.txt',
        help='The word2idx filename')
    parser.add_argument('--dataspace',
        type=str,
        default='./data',
        help='Where the data stored')
    
    FLAGS, unparsed = parser.parse_known_args()
    jieba.load_userdict(os.path.join(FLAGS.dataspace, 'user_dict.txt'))
    
    if FLAGS.task == 'test':
        in_file = FLAGS.filename
        out_file = FLAGS.filename.split()[0] + '.tfrecords'
        sequence2tfrecords(os.path.join(FLAGS.dataspace, in_file), os.path.join(FLAGS.dataspace, out_file))
    elif FLAGS.task == 'train':
        in_file = FLAGS.filename + '.csv'
        out_file = FLAGS.filename + '.tfrecords'
        sequence2tfrecords(os.path.join(FLAGS.dataspace, in_file), os.path.join(FLAGS.dataspace, out_file), train=True)
    elif FLAGS.task == 'w2i':
        in_file = os.path.join(FLAGS.dataspace, FLAGS.filename)
        word2idx(in_file, os.path.join(FLAGS.dataspace, 'word2idx.txt'))
