# coding: utf-8
import tensorflow as tf
import numpy as np
import math
import functools
from run_transformer import dataset, wi, posi, vector_dim, pos_size

class FLAGS_(object):
    def __init__(self):
        self.allow_pe = True
FLAGS = FLAGS_()

def get_input_fn(mode, pandas_pattern, batch_size):
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
        
        if FLAGS.allow_pe:
            s1_pos_ = get_position_encoding(len1, vector_dim + pos_size)
            s2_pos_ = get_position_encoding(len2, vector_dim + pos_size)
            s1_pe = tf.concat([s1_pos_, s2_pos_], axis=0)
            s2_pe = tf.concat([s2_pos_, s1_pos_], axis=0)

        features = {
            "s1": s1,
            "s2": s2,
            "len1": len1,
            "len2": len2,
            "pos1": pos1,
            "pos2": pos2}
        if FLAGS.allow_pe:
            features["pe1"] = s1_pe
            features["pe2"] = s2_pe
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
        dataset = pandas_pattern.reset_index(drop=True)
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
            dataset = dataset.shuffle(buffer_size=1000000)
        if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.repeat()

        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, mode=mode),
            num_parallel_calls=10)
        dataset = dataset.prefetch(100000)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=100000)
        # Our inputs are variable length, so pad them.
        padded_shapes={
            "len1": 1,
            "len2": 1,
            "s1": tf.TensorShape([None]),
            "s2": tf.TensorShape([None]),
            "pos1": tf.TensorShape([None]),
            "pos2": tf.TensorShape([None])}
        if FLAGS.allow_pe:
            padded_shapes["pe1"] = tf.TensorShape([None, vector_dim + len(posi)])
            padded_shapes["pe2"] = tf.TensorShape([None, vector_dim + len(posi)])
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
    if FLAGS.allow_pe:
        pe1 = features["pe1"]
        pe2 = features["pe2"]
    

    if labels is not None:
        labels = tf.squeeze(labels)
    # s1 = tf.cast(s1, tf.int64)
    # s2 = tf.cast(s2, tf.int64)
    if FLAGS.allow_pe:
        return s1, s2, pos1, pos2, pe1, pe2, len1, len2, labels
    else:
        return s1, s2, pos1, pos2, len1, len2, labels 


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


input_fn = get_input_fn(tf.estimator.ModeKeys.EVAL, dataset, 8)
feature, label = input_fn()
if FLAGS.allow_pe:
    s1, s2, pos1, pos2, pe1, pe2, len1, len2, labels = get_input_tensors(feature, label)
else:
    s1, s2, pos1, pos2, len1, len2, labels = get_input_tensors(feature, label)
sess = tf.Session()
for idx in range(3):
    if labels is None:
        if FLAGS.allow_pe:
            s1_, len1_, pe1_ = sess.run([s1, len1, pe1] if FLAGS.allow_pe else [s1, len1])
        else:
            s1_, len1_ = sess.run([s1, len1, pe1] if FLAGS.allow_pe else [s1, len1])
    else:
        if FLAGS.allow_pe:
            s1_, len1_, label_, pe1_ = sess.run([s1, len1, labels, pe1] if FLAGS.allow_pe else [s1, len1, labels])
        else:
            s1_, len1_, label_ = sess.run([s1, len1, labels, pe1] if FLAGS.allow_pe else [s1, len1, labels])
    # pe = get_position_encoding([idx], idx)
    # pe = sess.run(pe)
    print(s1_)
    print(len1_)
    print(label_)
    # print(pe1_)

# pe, cpe, c = get_position_encoding([20], 364)
# pe, cpe, c = sess.run([pe, cpe, c])

# print(pe)
# print(cpe)
# print(c)