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

# INPUT
data = df1
pos_i = df3
pos_size = df3.shape[0]
wordvector = df2.loc[:, "f0":"f299"].values
local_wv = df2["sent1"]
print("Reading input completed.")
vector_dim = wordvector.shape[-1]

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
dataset = pd.concat([dataset["label"], wp1, wp2], axis=1)
# print(dataset.head(5))
# raise RuntimeError("hahaha")
assert data.shape[0] == dataset.shape[0]
print("Data rows:", dataset.shape[0])

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
            padded_shapes["pe1"] = tf.TensorShape([None, vector_dim + len(posi)]),
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


def inference_v2(features1, features2, lengths1, lengths2, mode, params):
    with tf.name_scope("ProjectionLayer"):
        projection_layer = tf.layers.Dense(params.hidden_size)
        features1 = projection_layer(features1)
        features2 = projection_layer(features2)
    with tf.name_scope("SequencePadding"):
        lengths = lengths1 + lengths2
        s1_mask = tf.sequence_mask(lengths, tf.reduce_max(lengths), tf.float32)
        s2_mask = tf.sequence_mask(lengths, tf.reduce_max(lengths), tf.float32)
        s1_padding = get_padding(s1_mask)
        s2_padding = get_padding(s2_mask)
        s1_attention_bias = get_padding_bias(s1_mask)
        s2_attention_bias = get_padding_bias(s2_mask)

    encoder_stack = EncoderStack(params, mode == tf.estimator.ModeKeys.TRAIN)

    s1_encoded = encoder_stack(features1, s1_attention_bias, s1_padding)
    s2_encoded = encoder_stack(features2, s2_attention_bias, s2_padding)

    with tf.name_scope("Sum2Rep"):
        s1_rep = tf.reduce_sum(s1_encoded, axis=-2)
        s2_rep = tf.reduce_sum(s2_encoded, axis=-2)
        # merged = tf.concat([s1_rep, s2_rep], axis=-1, name="rep_concat")
        merged = s1_rep + s2_rep
    with tf.name_scope("Linear"):
        dense = tf.layers.batch_normalization(merged, training=(mode == tf.estimator.ModeKeys.TRAIN), name="PairedRepBN")
        for idx, dense_dim in enumerate(params.dense_dim):
            dense = tf.layers.dense(dense, dense_dim, activation=tf.nn.elu, name="fc_hidden1_" + str(idx), reuse=tf.AUTO_REUSE, use_bias=False)
            dense = tf.layers.dropout(dense, rate=params.dropout, training=(mode == tf.estimator.ModeKeys.TRAIN), name="dropout_hidden_" + str(idx))
            dense = tf.layers.batch_normalization(dense, training=(mode == tf.estimator.ModeKeys.TRAIN), name="BN_hidden_" + str(idx))

        logits = tf.layers.dense(dense, 2, activation=None, name="fc", reuse=tf.AUTO_REUSE)

    return logits


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.

    Args:
      x: int tensor with any shape
      padding_value: int value that

    Returns:
      flaot tensor with same shape as x containing values 0 or 1.
        0 -> non-padding, 1 -> padding
    """
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:
      x: int tensor with shape [batch_size, length]

    Returns:
      Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        padding = padding > 0.5
        padding = tf.expand_dims(padding, -1)   # --> [batch_size, length, 1]
        padding = tf.logical_or(padding, tf.transpose(padding, [0, 2, 1]))   # --> [batch_size, length, length]
        attention_bias = tf.to_float(padding) * -1e9
        attention_bias = tf.expand_dims(attention_bias, axis=1) # --> [batch_size, 1, length, length]
    return attention_bias


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


class Attention(tf.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, train, params):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train
        self.params = params

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
          x: A tensor with shape [batch_size, length, hidden_size]

        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            # batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [self.params.batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            # --> [batch, length, num_heads, depth]
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None):
        """Apply attention mechanism to x and y.

        Args:
          x: a tensor with shape [batch_size, length_x, hidden_size]
          y: a tensor with shape [batch_size, length_y, hidden_size]
          bias: attention bias that will be added to the result of the dot product.
          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, key_channels],
                 "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.

        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads. [batch_size, num_heads, length, hidden_dim//num_heads]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)  # --> [batch_size, num_heads, length, length]
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        # weights = _masked_softmax(logits, length)
        if self.train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)  # --> [batch_size, num_heads, length, hidden_size//num_heads]

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)  # --> [batch_size, length, hidden_size]

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)


class FeedFowardNetwork(tf.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad):
        super(FeedFowardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(
            filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
        self.output_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, name="output_layer")

    def call(self, x, padding=None):
        """Return outputs of the feedforward network.

        Args:
          x: tensor with shape [batch_size, length, hidden_size]
          padding: (optional) If set, the padding values are temporarily removed
            from x (provided self.allow_pad is set). The padding values are placed
            back in the output tensor in the same locations.
            shape [batch_size, length]

        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """
        padding = None if not self.allow_pad else padding

        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        if padding is not None:
            with tf.name_scope("remove_padding"):
                # Flatten padding to [batch_size*length]
                pad_mask = tf.reshape(padding, [-1])

                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

                # Reshape x to [batch_size*length, hidden_size] to remove padding
                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                # Reshape x from 2 dimensions to 3 dimensions.
                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)

        output = self.filter_dense_layer(x)
        if self.train:
            output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
        output = self.output_dense_layer(output)

        if padding is not None:
            with tf.name_scope("re_add_padding"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                    indices=nonpad_ids,
                    updates=output,
                    shape=[batch_size * length, self.hidden_size]
                )
                output = tf.reshape(
                    output, [batch_size, length, self.hidden_size])
        return output


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(
            tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params, train):
        self.layer = layer
        self.postprocess_dropout = params.layer_postprocess_dropout
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(params.hidden_size)

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y


class EncoderStack(tf.layers.Layer):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, params, train):
        super(EncoderStack, self).__init__()
        self.layers = []
        for _ in range(params.num_hidden_layers):
            # Create sublayers for each layer.
            self_attention_layer = SelfAttention(
                params.hidden_size, params.num_heads,
                params.attention_dropout, train, params)
            feed_forward_network = FeedFowardNetwork(
                params.hidden_size, params.filter_size,
                params.relu_dropout, train, params.allow_ffn_pad)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params.hidden_size)

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer.
            [batch_size, 1, 1, input_length]
          inputs_padding: P

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
                with tf.variable_scope("ffn"):
                    encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

        return self.output_normalization(encoder_inputs)


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
        if self.FLAGS.allow_pe:
            s1, s2, pos1, pos2, pe1, pe2, len1, len2, labels = get_input_tensors(features, labels)
        else:
            s1, s2, pos1, pos2, len1, len2, labels = get_input_tensors(features, labels)
        with tf.name_scope("Input_embedding"):
            embeddings = tf.constant(self.W, dtype=tf.float32, name="fastText")
            embedded_s1 = tf.nn.embedding_lookup(embeddings, s1)
            embedded_s2 = tf.nn.embedding_lookup(embeddings, s2)

            embedded_pos1 = tf.one_hot(pos1, 64, name="POS_embedding_1")
            embedded_pos2 = tf.one_hot(pos2, 64, name="POS_embedding_2")
            
            embedded_s1 = tf.reshape(embedded_s1, [params.batch_size, -1, 300])
            embedded_s2 = tf.reshape(embedded_s2, [params.batch_size, -1, 300])
            embedded_pos1 = tf.reshape(embedded_pos1, [params.batch_size, -1, pos_size])
            embedded_pos2 = tf.reshape(embedded_pos2, [params.batch_size, -1, pos_size])

            embedded_s1 = tf.concat([embedded_s1, embedded_pos1], axis=-1, name="s1_concat_pos1")
            embedded_s2 = tf.concat([embedded_s2, embedded_pos2], axis=-1, name="s2_concat_pos2")

        # logits = inference_v2(embedded_s1, embedded_s2, embedded_pos1, embedded_pos2, len1, len2, mode, params)
        if self.FLAGS.allow_pe:
            logits = inference_v2(embedded_s1 + pe1, embedded_s2 + pe2, len1, len2, mode, params)
        else:
            logits = inference_v2(embedded_s1, embedded_s2, len1, len2, mode, params)

        predictions = tf.argmax(logits, axis=1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"logits": tf.nn.softmax(logits), "predictions": predictions})

        cross_entropy = focal_loss(logits, labels, return_mean=True)

        train_op = tf.contrib.layers.optimize_loss(
            loss=cross_entropy,
            global_step=tf.train.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer="Adam",
            clip_gradients=params.gradient_clipping_norm,
            summaries=["learning_rate", "loss", "gradients", "gradient_norm"]
        )
        
        with tf.name_scope("Metrics"):
            precision = tf.metrics.precision(labels, predictions)
            recall = tf.metrics.recall(labels, predictions)
            accuracy = tf.metrics.accuracy(labels, predictions)
            f1_score_ = f1_score(precision, recall)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"logits": logits, "predictions": predictions},
            loss=cross_entropy,
            train_op=train_op,
            eval_metric_ops={"accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score_}
        )

    def create_estimator_and_specs(self, run_config):
        """Creates an Experiment configuration based on the estimator and input fn."""
        model_params = tf.contrib.training.HParams(
            batch_size=self.FLAGS.batch_size,
            num_classes=2,
            num_heads=self.FLAGS.num_heads,
            learning_rate=self.FLAGS.learning_rate,
            gradient_clipping_norm=self.FLAGS.gradient_clipping_norm,
            dropout=self.FLAGS.dropout,
            layer_postprocess_dropout=self.FLAGS.layer_postprocess_dropout,
            attention_dropout=self.FLAGS.attention_dropout,
            relu_dropout=self.FLAGS.relu_dropout,
            pos_weight=self.FLAGS.pos_weight,
            hidden_size=self.FLAGS.hidden_size,
            filter_size=self.FLAGS.filter_size,
            dense_dim=ast.literal_eval(self.FLAGS.dense_dim),
            allow_ffn_pad=self.FLAGS.allow_ffn_pad,
            num_hidden_layers=self.FLAGS.num_hidden_layers)

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
            batch_size=self.FLAGS.batch_size), steps=10000//FLAGS.batch_size, throttle_secs=300, start_delay_secs=300)

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
        print("Start to train ...")
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
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--gradient_clipping_norm", type=float, default=9.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--model_dir", type=str, default=model_dir)
    parser.add_argument("--task", type=str, default="eval")
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dense_dim", type=str, default="[400, 100]")
    parser.add_argument("--num_hidden_layers", type=str, default=2)
    parser.add_argument("--print_logits", type="bool", default="false")
    parser.add_argument("--layer_postprocess_dropout", type=float, default=0.05)
    parser.add_argument("--relu_dropout", type=float, default=0.05)
    parser.add_argument("--attention_dropout", type=float, default=0.05)
    parser.add_argument("--filter_size", type=int, default=512)
    parser.add_argument("--allow_ffn_pad", type="bool", default="true")
    parser.add_argument("--allow_pe", type="bool", default="false")
    
    print("Parse command ...", end="")
    FLAGS, unparsed = parser.parse_known_args()
    print("Done!")

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

