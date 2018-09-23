# coding: utf-8
import argparse
import sys
import os
import glob
import functools
import codecs
import ast
import io
import math
import time
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from input_online import *
except ImportError:
    print("We are online!")

# INPUT
data = df1
pos_i = df3
pos_size = df3.shape[0]
wordvector = df2.loc[:, "f0":"f299"].values
local_wv = df2["sent1"]
print("Reading input completed.")
vector_dim = wordvector.shape[-1]

def split_word_pstg(x, columns=["sent1", "pstg1", "len1"]):
    word_pstg_pair_list = [word.split("/") for word in x.split()]
    word_pstg_pair_list.append(["</s>", "end"])
    word_str_token = [str(wi[pair[0].strip()]) for pair in word_pstg_pair_list\
                        if (pair[0] and (pair[0] is not " ") and pair[1] and posi[pair[1].lower()])]
    pstg_str_token = [str(posi[pair[1].strip().lower()]) for pair in word_pstg_pair_list\
                        if (pair[0] and (pair[0] is not " ") and pair[1] and posi[pair[1].lower()])]
    # print(word_str_token)
    # print(pstg_str_token)
    assert len(word_str_token) == len(pstg_str_token)
    word_str = " ".join(word_str_token)
    pstg_str = " ".join(pstg_str_token)

    return pd.Series(dict(zip(columns, [word_str, pstg_str, len(word_str_token)])))


def get_wi(local_wv, pos_i):
    print("Dictize word with id...", end="")
    wi = defaultdict(lambda : 0)
    vocab_size = local_wv.shape[0]
    local_wv_ = pd.Series([idx + 1 for idx in range(vocab_size)],
                        index=pd.Index(local_wv.values.squeeze(), name="word"))
    wi = local_wv_.to_dict(wi)
    print("Done!")

    print("Dictize pos tag with id...", end="")
    pos = defaultdict(lambda : 0)
    pos_size = pos_i.shape[0]
    pos_i = pd.Series([idx + 1 for idx in range(pos_size)],
                    index=pd.Index(pos_i.values.squeeze(), name="pos"))
    pos = pos_i.to_dict(pos)
    print("Done!")

    return wi, pos
wi, posi = get_wi(local_wv, pos_i)
dataset = data.reset_index(drop=True)
wp1 = dataset["sent1"].apply(lambda x: split_word_pstg(x, ["sent1", "pstg1", "len1"]))
wp2 = dataset["sent2"].apply(lambda x: split_word_pstg(x, ["sent2", "pstg2", "len2"]))
# print("wwi==============", len(wi))
# print("posi==============", len(posi))
# print(posi)
dataset = pd.concat([dataset["label"], wp1, wp2], axis=1)


def get_input_fn(mode, pandas_pattern, batch_size):
    """Creates an input_fn that stores all the data in memory.

    Args:
     mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
     pandas_pattern: path to a TF record file created using create_dataset.py.
     batch_size: the batch size to output.

    Returns:
      A valid input_fn for the model estimator.
    """
    def to_ids(raw):
        raw_str = tf.string_split([raw]).values
        return tf.string_to_number(raw_str, tf.int32)

    def _parse_tfexample_fn(pandas_row, mode):
        """Parse a single record which is expected to be a tensorflow.Example."""
        # global wi, posi, vector_dim
        s1 = to_ids(pandas_row["sent1"])
        s2 = to_ids(pandas_row["sent2"])
        pos1 = to_ids(pandas_row["pstg1"])
        pos2 = to_ids(pandas_row["pstg2"])
        len1 = pandas_row["len1"]
        len2 = pandas_row["len2"]

        features = {
            "s1": s1,
            "s2": s2,
            "len1": len1,
            "len2": len2,
            "pos1": pos1,
            "pos2": pos2}
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
        tensor_slice = {"sent1": dataset["sent1"].values,
            "sent2": dataset["sent2"].values,
            "pstg1": dataset["pstg1"].values,
            "pstg2": dataset["pstg2"].values,
            "len1": dataset["len1"].values[:, np.newaxis],
            "len2": dataset["len2"].values[:, np.newaxis]}
        if mode != tf.estimator.ModeKeys.PREDICT:
            tensor_slice["label"] = dataset["label"].values[:, np.newaxis]
        dataset = tf.data.Dataset.from_tensor_slices(tensor_slice)

        if mode == tf.estimator.ModeKeys.TRAIN:
            print("Train")
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat()
        elif mode == tf.estimator.ModeKeys.EVAL:
            print("Evaluating ...")

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
            "pos2": tf.TensorShape([None])
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
    # s1 = tf.decode_raw(features["s1"], tf.int32)
    # s2 = tf.decode_raw(features["s2"], tf.int32)

    if labels is not None:
        labels = tf.squeeze(labels)
    # s1 = tf.cast(s1, tf.int64)
    # s2 = tf.cast(s2, tf.int64)
    
    return s1, s2, pos1, pos2, len1, len2, labels

def inference_v2(features1, features2, lengths1, lengths2, mode, params):
    # with tf.name_scope("LSTM"):
    #     features1 = stack_bidirectional_LSTM(features1, lengths1, 'sen_embeddings', mode, params)
    #     features2 = stack_bidirectional_LSTM(features2, lengths2, 'sen_embeddings', mode, params)

    with tf.name_scope("Embeddings_Projection"):
        projection_layers = []
        if params.projection_hidden > 0:
            projection_layers.extend([
                tf.layers.Dense(params.projection_hidden, activation=tf.nn.relu, name="projection_hidden", _reuse=tf.AUTO_REUSE),
                Dropout(params.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
            ])
        projection_layers.extend([
            tf.layers.Dense(params.projection_dim, activation=tf.nn.relu, name="projection_output", _reuse=tf.AUTO_REUSE),
            Dropout(params.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
        ])    
        s1_encoded = features1
        s2_encoded = features2
        for idx, layer in enumerate(projection_layers):
            s1_encoded = layer(s1_encoded)
            s2_encoded = layer(s2_encoded)
        s1_encoded = _masked_seq(s1_encoded, lengths1)
        s2_encoded = _masked_seq(s2_encoded, lengths2)
    
    with tf.name_scope("soft_attention_alignment"):
        s1_aligned, s2_aligned = soft_attention_alignment(features1, features2, s1_encoded, s2_encoded, lengths1, lengths2)

    with tf.name_scope("self_attention"):
        pe_1 = get_position_encoding(tf.reduce_max(lengths1), features1.shape[2])
        pe_2 = get_position_encoding(tf.reduce_max(lengths2), features2.shape[2])
        s1_self = _masked_seq(pe_1 + features1, lengths1)
        s2_self = _masked_seq(pe_2 + features2, lengths2)
        self_projection_layers = []
        self_projection_layers.extend([
            tf.layers.Dense(params.projection_dim, activation=tf.nn.relu, name="self_project", _reuse=tf.AUTO_REUSE),
            Dropout(params.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
        ])
        for idx, layer in enumerate(self_projection_layers):
            s1_self = layer(s1_self)
            s2_self = layer(s2_self)
        s1_self = self_attention_alignment(features1, s1_self, lengths1)
        s2_self = self_attention_alignment(features2, s2_self, lengths2)
    
    with tf.name_scope("Concat_EAs"):
        s1_combined = tf.concat([s1_encoded, s1_self, s2_aligned, s1_encoded*s2_aligned], -1, name="s1_concat")
        s2_combined = tf.concat([s2_encoded, s2_self, s1_aligned, s2_encoded*s1_aligned], -1, name="s2_concat")

    with tf.name_scope("Comparision_Projection"):
        compare_layers = []
        for idx, compare_dim in enumerate(params.compare_dim):
            USE_BIAS = (idx != (len(params.compare_dim) - 1))
            compare_layers.extend([
                tf.layers.Dense(compare_dim, activation=tf.nn.relu, name="compare_" + str(idx), _reuse=tf.AUTO_REUSE, use_bias=USE_BIAS),
                Dropout(params.dropout, training=(mode == tf.estimator.ModeKeys.TRAIN))
            ])
        s1_compare = s1_combined
        s2_compare = s2_combined
        for idx, layer in enumerate(compare_layers):
            s1_compare = layer(s1_compare)
            s2_compare = layer(s2_compare)
        # s1_compare = s1_compare*tf.expand_dims(tf.sequence_mask(lengths1, tf.reduce_max(lengths1), dtype=tf.float32), axis=-1)
        # s2_compare = s2_compare*tf.expand_dims(tf.sequence_mask(lengths2, tf.reduce_max(lengths2), dtype=tf.float32), axis=-1)
        s1_compare = _masked_seq(s1_compare, lengths1)
        s2_compare = _masked_seq(s2_compare, lengths2)
        s1_compare = tf.reshape(s1_compare, [params.batch_size, -1, params.compare_dim[-1]])
        s2_compare = tf.reshape(s2_compare, [params.batch_size, -1, params.compare_dim[-1]])
        
    with tf.name_scope("Pooling"):
        s1_rep = Reducesum(axis=1)(s1_compare)
        s2_rep = Reducesum(axis=1)(s2_compare)
    
    with tf.name_scope("FC"):
        merged = tf.concat([s1_rep, s2_rep], axis=-1, name="rep_concat")
        dense = tf.layers.batch_normalization(merged, training=(mode == tf.estimator.ModeKeys.TRAIN), name="PairedRepBN")
        for idx, dense_dim in enumerate(params.dense_dim):
            dense = tf.layers.dense(dense, dense_dim, activation=tf.nn.elu,
                                    name="fc_hidden1_" + str(idx), reuse=tf.AUTO_REUSE, use_bias=False)
            dense = tf.layers.dropout(dense, rate=params.dropout,
                                    training=(mode == tf.estimator.ModeKeys.TRAIN), name="dropout_hidden_" + str(idx))
            dense = tf.layers.batch_normalization(dense, training=(mode == tf.estimator.ModeKeys.TRAIN), name="BN_hidden_" + str(idx))

        logits = tf.layers.dense(dense, 2, activation=None, name="fc", reuse=tf.AUTO_REUSE)
    
    return logits


def soft_attention_alignment(features1, features2, projected_1, projected_2, len1, len2, projected_attention=True):
    attention = tf.matmul(projected_1, tf.transpose(projected_2, perm=[0, 2, 1]), name="attention")
    w_att_1 = _masked_softmax(tf.transpose(attention, perm=[0, 2, 1]), len1, axis=-1)
    w_att_2 = _masked_softmax(attention, len2, axis=-1)
    
    if projected_attention:
        in1_aligned = tf.matmul(w_att_1, projected_1, name="s1_align")
        in2_aligned = tf.matmul(w_att_2, projected_2, name="s2_align")
    else:
        in1_aligned = tf.matmul(w_att_1, features1, name="s1_align")
        in2_aligned = tf.matmul(w_att_2, features2, name="s2_align")

    # in1_aligned = _masked_seq(in1_aligned, len1)
    # in2_aligned = _masked_seq(in2_aligned, len2)

    return in1_aligned, in2_aligned

def self_attention_alignment(features, projected, lens, projected_attention=True):
    attention = tf.matmul(projected, tf.transpose(projected, perm=[0, 2, 1]), name="intra_attention")
    att = _masked_softmax(attention, lens, axis=-1)
    
    if projected_attention:
        features_aligned = tf.matmul(att, projected, name="intra_align")
    else:
        features_aligned = tf.matmul(att, features, name="intra_align")
    # features_aligned = _masked_seq(features_aligned, lens)

    return features_aligned

def _masked_softmax(values, lengths, axis=None):
    mask = tf.expand_dims(tf.sequence_mask(lengths, tf.reduce_max(lengths), dtype=tf.float32), -2)
    inf_mask = (1 - mask) * -np.inf
    inf_mask = tf.where(tf.is_nan(inf_mask), tf.zeros_like(inf_mask), inf_mask)

    return tf.nn.softmax(values*mask + inf_mask, axis=axis)

def _masked_seq(values, lengths, axis=None):
    mask = tf.tile(tf.expand_dims(tf.sequence_mask(lengths, tf.reduce_max(lengths), dtype=tf.bool), -1),
                [1, 1, values.shape[-1]])
    
    return tf.where(mask, values, tf.zeros_like(values, dtype=values.dtype))

class Dropout(tf.layers.Dropout):
    def __init__(self, rate, training=False, **kwargs):
        super(Dropout, self).__init__(rate=rate, **kwargs)
        self.training = training
    
    def call(self, inputs):
        return tf.layers.dropout(inputs, self.rate, 
                                noise_shape=self._get_noise_shape(inputs),
                                seed=self.seed,
                                training=self.training)

    def compute_output_shape(self, input_shape):
        return input_shape


class Reducesum(tf.layers.Layer):
    def __init__(self, axis=None, **kwargs):
        super(Reducesum, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class Reducemax(tf.layers.Layer):
    def __init__(self, axis=None, **kwargs):
        super(Reducemax, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=self.axis)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = tf.concat(agg_, axis=-1)
    return out_

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
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal

def focal_loss(prediction_tensor, labels, weights=None, alpha=0.25, gamma=2, return_mean=True):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: A float tensor of shape [batch_size, num_anchors]
        alpha: A scalar tensor for focal loss alpha hyper-parameter
        gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    with tf.name_scope("focal_loss"):
        target_tensor = tf.one_hot(labels, 2, dtype=prediction_tensor.dtype)
        sigmoid_p = tf.nn.softmax(prediction_tensor, axis=-1)
        zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        
        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = tf.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
        
        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(target_tensor > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                            - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        if return_mean:
            return tf.reduce_mean(per_entry_cross_ent)
        else:
            return tf.reduce_sum(per_entry_cross_ent)

class SiameseModel(object):
    def __init__(self, FLAGS):
        self.W = np.zeros((wordvector.shape[0] + 1, 300))
        self.W[1:, :] = wordvector
        self.FLAGS = FLAGS

    def model_fn(self, features, labels, mode, params):
        s1, s2, pos1, pos2, len1, len2, labels = get_input_tensors(features, labels)
        # s1, s2, len1, len2, labels = get_input_tensors(features, labels)
        with tf.name_scope("Input_embedding"):
            embeddings = tf.constant(self.W, dtype=tf.float32, name="fastText")
            embedded_s1 = tf.nn.embedding_lookup(embeddings, s1)
            embedded_s2 = tf.nn.embedding_lookup(embeddings, s2)

            # pos_embeddings = tf.get_variable("POS_embeddings", shape=[56], initializer=tf.initializers.ones)
            # embedded_pos1 = tf.nn.embedding_lookup(pos_embeddings, pos1)
            # embedded_pos2 = tf.nn.embedding_lookup(pos_embeddings, pos2)
            embedded_pos1 = tf.one_hot(pos1, pos_size, name="POS_embedding_1")
            embedded_pos2 = tf.one_hot(pos2, pos_size, name="POS_embedding_2")
            
            embedded_s1 = tf.reshape(embedded_s1, [params.batch_size, -1, 300])
            embedded_s2 = tf.reshape(embedded_s2, [params.batch_size, -1, 300])
            embedded_pos1 = tf.reshape(embedded_pos1, [params.batch_size, -1, pos_size])
            embedded_pos2 = tf.reshape(embedded_pos2, [params.batch_size, -1, pos_size])

            embedded_s1 = tf.concat([embedded_s1, embedded_pos1], axis=-1, name="s1_concat_pos1")
            embedded_s2 = tf.concat([embedded_s2, embedded_pos2], axis=-1, name="s2_concat_pos2")

        # logits = inference_v2(embedded_s1, embedded_s2, embedded_pos1, embedded_pos2, len1, len2, mode, params)
        logits = inference_v2(embedded_s1, embedded_s2, len1, len2, mode, params)

        predictions = tf.argmax(logits, axis=1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"logits": tf.nn.softmax(logits), "predictions": predictions})

        cross_entropy = focal_loss(logits, labels, return_mean=False)

        train_op = tf.contrib.layers.optimize_loss(
            loss=cross_entropy,
            global_step=tf.train.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer="Adam",
            clip_gradients=params.gradient_clipping_norm,
            summaries=["learning_rate", "loss", "gradients", "gradient_norm"]
        )
        
        precision = tf.metrics.precision(labels, predictions)
        recall = tf.metrics.recall(labels, predictions)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"logits": logits, "predictions": predictions},
            loss=cross_entropy,
            train_op=train_op,
            eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions),
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score(precision, recall)}
        )

    def create_estimator_and_specs(self, run_config):
        """Creates an Experiment configuration based on the estimator and input fn."""
        model_params = tf.contrib.training.HParams(
            batch_size=self.FLAGS.batch_size,
            num_classes=2,
            learning_rate=self.FLAGS.learning_rate,
            gradient_clipping_norm=self.FLAGS.gradient_clipping_norm,
            batch_norm=self.FLAGS.batch_norm,
            dropout=self.FLAGS.dropout,
            pos_weight=self.FLAGS.pos_weight,
            projection_hidden=self.FLAGS.projection_hidden,
            projection_dim=self.FLAGS.projection_dim,
            compare_dim=ast.literal_eval(self.FLAGS.compare_dim),
            dense_dim=ast.literal_eval(self.FLAGS.dense_dim))

        estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=run_config,
            params=model_params)

        train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(
            mode=tf.estimator.ModeKeys.TRAIN,
            pandas_pattern=dataset,
            batch_size=self.FLAGS.batch_size), max_steps=self.FLAGS.steps)

        eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(
            mode=tf.estimator.ModeKeys.EVAL,
            pandas_pattern=dataset,
            batch_size=self.FLAGS.batch_size), steps=1, throttle_secs=1000, start_delay_secs=300)

        return estimator, train_spec, eval_spec


def f1_score(predictions=None, recalls=None, weights=None):
    P, update_op1 = predictions
    R, update_op2 = recalls
    eps = 1e-5
    return (2*(P*R)/(P+R+eps), tf.group(update_op1, update_op2))

def main(unused_args):
    if not os.path.exists(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)
    print("Model_dir", os.listdir(FLAGS.model_dir))
    if FLAGS.task == "retrain":
        files = glob.glob(os.path.join(FLAGS.model_dir, "*"))
        for f in files:
            if os.path.isdir(f):
                shutil.rmtree(f, ignore_errors=True)
            else:
                os.remove(f)
    
    model = SiameseModel(FLAGS)
    print("Start to build model ...", end="")
    estimator, train_spec, eval_spec = model.create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=300,
            save_summary_steps=100))
    print("Done!")
    start_time = time.time()
    if FLAGS.task == "train":
        print("Start to train ...")
        estimator.train(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.TRAIN,
                pandas_pattern=dataset,
                batch_size=FLAGS.batch_size), max_steps=FLAGS.steps)
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print(os.listdir(FLAGS.model_dir))
    elif FLAGS.task == "retrain":
        print("Start to train ...")
        estimator.train(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.TRAIN,
                pandas_pattern=dataset,
                batch_size=FLAGS.batch_size), max_steps=FLAGS.steps)
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print(os.listdir(FLAGS.model_dir))
    elif FLAGS.task == "eval":
        print("Start to train and evaluate ...")
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        print(os.listdir(FLAGS.model_dir))
    elif FLAGS.task == "predict":
        print("Start to predict ...")
        predictions = estimator.predict(
            input_fn=get_input_fn(mode=tf.estimator.ModeKeys.PREDICT,
                pandas_pattern=dataset,
                batch_size=FLAGS.batch_size)
        )
        logits = pd.DataFrame(np.array([predict["logits"] for predict in predictions]))
        logits.columns = ["prob0", "prob1"]
        result = (logits.prob1 > 0.29).astype(int)
        result = pd.DataFrame({"id": [idx + 1 for idx in range(dataset.shape[0])],
                            "label": result.values,
                            "proba": logits.prob1.values})
        topai(1, result)
    elif FLAGS.task == "predict_proba":
        print("Start to predict probability...")
        predictions = estimator.predict(
            input_fn=get_input_fn(mode=tf.estimator.ModeKeys.PREDICT,
                pandas_pattern=dataset,
                batch_size=FLAGS.batch_size)
        )
        predictions = list(predictions)
        logits = pd.DataFrame(np.array([predict["logits"] for predict in predictions]))
        logits.columns = ["prob0", "prob1"]
        result = logits.prob1
        result.index += 1
        result.to_csv(FLAGS.result_name, header=None, sep="\t")
    duration = time.time() - start_time
    print("Time usage: ", duration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument( "--training_data", type=str, default=os.path.join(model_dir, "train.tfrecords"))
    parser.add_argument( "--eval_data", type=str, default=os.path.join(model_dir, "train.tfrecords"))
    parser.add_argument( "--test_data", type=str, default=os.path.join(model_dir, "train.tfrecords"))
    parser.add_argument( "--batch_norm", type="bool", default="False")
    parser.add_argument( "--learning_rate", type=float, default=0.0001)
    parser.add_argument( "--gradient_clipping_norm", type=float, default=9.0)
    parser.add_argument( "--dropout", type=float, default=0.2)
    parser.add_argument( "--steps", type=int, default=50000)
    parser.add_argument( "--batch_size", type=int, default=256)
    parser.add_argument( "--model_dir", type=str, default=model_dir)
    parser.add_argument( "--result_name", type=str, default="./result.csv")
    parser.add_argument( "--pos_weight", type=float, default=1.0)
    parser.add_argument( "--projection_hidden", type=int, default=0)
    parser.add_argument( "--projection_dim", type=int, default=200)
    parser.add_argument( "--compare_dim", type=str, default="[800, 400]")
    parser.add_argument( "--dense_dim", type=str, default="[400, 200]")
    parser.add_argument( "--task", type=str, default="eval")
    
    print("Parse command ...", end="")
    FLAGS, unparsed = parser.parse_known_args()
    print("Done!")

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

