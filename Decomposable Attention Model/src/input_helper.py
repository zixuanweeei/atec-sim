# coding: utf-8
import functools
import math
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from convert import get_wi

wi, posi = get_wi()
with io.open("./data/local_wv.txt", "r", encoding="utf-8") as reader:
    info = reader.readline().split(" ")
    words_num = int(info[0])
    vector_dim = int(info[1])
    W = np.zeros((words_num + 1, vector_dim))
    for idx, line in enumerate(reader.readlines()):
        ele = line.rstrip().split(" ")
        W[idx + 1, ] = list(map(float, ele[1:]))
W = tf.constant(W, name="word_embeddings", dtype=tf.float32)
print("Word Embeddings shape:", W.shape)


def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
        length: Sequence length.
        hidden_size: Size of the
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position

    Returns:
        Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length[0]))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * \
        tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

    return signal


def get_input_fn(mode, tfrecord_pattern, batch_size):
    """Creates an input_fn that stores all the data in memory.

    Args:
     mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
     tfrecord_pattern: path to a TF record file created using create_dataset.py.
     batch_size: the batch size to output.

    Returns:
      A valid input_fn for the model estimator.
    """

    def _parse_tfexample_fn(example_proto, mode):
        """Parse a single record which is expected to be a tensorflow.Example."""
        sequence_features = {
            "s1": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "s2": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "pos1": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "pos2": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }
        context_features = {
            "len1": tf.FixedLenFeature([1], dtype=tf.int64),
            "len2": tf.FixedLenFeature([1], dtype=tf.int64)
        }
        if mode != tf.estimator.ModeKeys.PREDICT:
            # The labels won't be available at inference time, so don't add them
            # to the list of feature_columns to be read.
            context_features["label"] = tf.FixedLenFeature(
                [1], dtype=tf.int64)

        context, sequences = tf.parse_single_sequence_example(
            example_proto,
            context_features=context_features,
            sequence_features=sequence_features)
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     context["label"] = None
        features = context.copy()
        features.update(sequences)

        return features

    def _input_fn():
        """Estimator `input_fn`.

        Returns:
          A tuple of:
          - Dictionary of string feature name to `Tensor`.
          - `Tensor` of target labels.
        """
        dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10)
        if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.repeat()
        # Preprocesses 10 files concurrently and interleaves records from each file.
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)
        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, mode=mode),
            num_parallel_calls=10)
        dataset = dataset.prefetch(10000)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000000)
        # Our inputs are variable length, so pad them.
        padded_shapes={
            "len1": 1,
            "len2": 1,
            "s1": tf.TensorShape([None]),
            "s2": tf.TensorShape([None]),
            "pos1": tf.TensorShape([None]),
            "pos2": tf.TensorShape([None]),
        }
        if mode != tf.estimator.ModeKeys.PREDICT:
            padded_shapes["label"] = 1
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        features = dataset.make_one_shot_iterator().get_next()
        if mode == tf.estimator.ModeKeys.PREDICT:
            return features, None
        return features, features["label"]

    return _input_fn


def split_word_pstg(x, columns=["sent1", "pstg1", "len1"]):
    word_pstg_pair_list = [word.split("/") for word in x.split()]
    word_pstg_pair_list.append(["</s>", "end"])
    word_str_token = [str(wi[pair[0].strip()]) for pair in word_pstg_pair_list\
                        if (pair[0] and (pair[0] is not " ") and pair[1])]
    pstg_str_token = [str(posi[pair[1].strip().lower()]) for pair in word_pstg_pair_list\
                        if (pair[0] and (pair[0] is not " ") and pair[1])]
    if (len(posi) > 64):
        print(word_str_token)
        print(x)
        print(word_pstg_pair_list)
        raise ValueError("hahahaha")
    assert len(word_str_token) == len(pstg_str_token)
    word_str = " ".join(word_str_token)
    pstg_str = " ".join(pstg_str_token)

    return pd.Series(dict(zip(columns, [word_str, pstg_str, len(word_str_token)])))


def pandas_input_fn(mode, pandas_pattern, batch_size):
    """Creates an input_fn that stores all the data in memory.

    Args:
     mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
     pandas_pattern: path to a TF record file created using create_dataset.py.
     batch_size: the batch size to output.

    Returns:
      A valid input_fn for the model estimator.
    """
    def to_ids(raw, hashtable):
        raw_str = tf.string_split([raw]).values
        return tf.string_to_number(raw_str, tf.int32)

    def _parse_tfexample_fn(pandas_row, mode):
        """Parse a single record which is expected to be a tensorflow.Example."""
        # global wi, posi, vector_dim
        s1 = to_ids(pandas_row["sent1"], wi)
        s2 = to_ids(pandas_row["sent2"], wi)
        pos1 = to_ids(pandas_row["pstg1"], posi)
        pos2 = to_ids(pandas_row["pstg2"], posi)
        len1 = pandas_row["len1"]
        len2 = pandas_row["len2"]

        s1_pos_ = get_position_encoding(len1, vector_dim + len(posi))
        s2_pos_ = get_position_encoding(len2, vector_dim + len(posi))
        s1_pe = tf.concat([s1_pos_, s2_pos_], axis=0)
        s2_pe = tf.concat([s2_pos_, s1_pos_], axis=0)

        features = {
            "s1": s1,
            "s2": s2,
            "len1": len1,
            "len2": len2,
            "pos1": pos1,
            "pos2": pos2,
            "pe1": s1_pe,
            "pe2": s2_pe}
        if mode != tf.estimator.ModeKeys.PREDICT:
            features["label"] = pandas_row["label"]
        
        return features

    def _input_fn():
        """Estimator `input_fn`.

        Returns:
          A tuple of:
          - Dictionary of string feature name to `Tensor`.
          - `Tensor` of target labels.
        """
        dataset = pd.read_csv(pandas_pattern, header=None, sep='\t', index_col=0)
        if dataset.shape[-1] == 3:
            dataset.columns = ["s1", "s2", "label"]
        elif dataset.shape[-1] == 2:
            dataset.columns = ["s1", "s2"]
        wp1 = dataset["s1"].apply(lambda x: split_word_pstg(x, ["sent1", "pstg1", "len1"]))
        wp2 = dataset["s2"].apply(lambda x: split_word_pstg(x, ["sent2", "pstg2", "len2"]))
        # print("wwi==============", len(wi))
        # print("posi==============", len(posi))
        # print(posi)
        dataset = pd.concat([dataset, wp1, wp2], axis=1)
        tensor_slice = {"sent1": dataset["sent1"].values + " " + dataset["sent2"].values,
            "sent2": dataset["sent2"].values + " " + dataset["sent1"].values,
            "pstg1": dataset["pstg1"].values + " " + dataset["pstg2"].values,
            "pstg2": dataset["pstg2"].values + " " + dataset["pstg1"].values,
            "len1": dataset["len1"].values[:, np.newaxis],
            "len2": dataset["len2"].values[:, np.newaxis]}
        if mode != tf.estimator.ModeKeys.PREDICT:
            tensor_slice["label"] = dataset["label"].values[:, np.newaxis]
        dataset = tf.data.Dataset.from_tensor_slices(tensor_slice)

        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10000)
        if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.repeat()

        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, mode=mode),
            num_parallel_calls=10)
        dataset = dataset.prefetch(10000)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=100000)
        # Our inputs are variable length, so pad them.
        padded_shapes={
            "len1": 1,
            "len2": 1,
            "s1": tf.TensorShape([None]),
            "s2": tf.TensorShape([None]),
            "pos1": tf.TensorShape([None]),
            "pos2": tf.TensorShape([None]),
            "pe1": tf.TensorShape([None, vector_dim + len(posi)]),
            "pe2": tf.TensorShape([None, vector_dim + len(posi)])
        }
        if mode != tf.estimator.ModeKeys.PREDICT:
            padded_shapes["label"] = 1
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        features = dataset.make_one_shot_iterator().get_next()
        if mode == tf.estimator.ModeKeys.PREDICT:
            return features, None
        return features, features["label"]

    return _input_fn


def get_input_tensors(features, labels):
    len1 = tf.squeeze(features["len1"])
    len2 = tf.squeeze(features["len2"])
    
    s1 = features["s1"]
    s2 = features["s2"]
    pos1 = features["pos1"]
    pos2 = features["pos2"]
    pe1 = features["pe1"]
    pe2 = features["pe2"]
    

    if labels is not None:
        labels = tf.squeeze(labels)
    # s1 = tf.cast(s1, tf.int64)
    # s2 = tf.cast(s2, tf.int64)
    
    return s1, s2, pos1, pos2, pe1, pe2, len1, len2, labels
    # return s1, s2, len1, len2, labels


if __name__ == "__main__":
    input_fn = pandas_input_fn(tf.estimator.ModeKeys.EVAL, "data/train.csv", 3)
    feature, label = input_fn()
    s1, s2, pos1, pos2, pe1, pe2, len1, len2, labels = get_input_tensors(feature, label)
    sess = tf.Session()
    for idx in range(2):
        if labels is None:
            s1_, len1_, pe1_ = sess.run([s1, len1, pe1])
            label_ = labels
        else:
            s1_, len1_, label_, pe1_ = sess.run([s1, len1, labels, pe1])
        print(s1_)
        print(len1_)
        print(label_)
        print(pe1_.shape)