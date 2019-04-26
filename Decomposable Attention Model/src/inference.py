# coding: utf-8
import math
import numpy as np
import tensorflow as tf

def inference(features1, features2, lengths1, lengths2, mode, params):
    s1 = stack_bidirectional_LSTM(features1, lengths1, 'embeddings', mode, params)
    s2 = stack_bidirectional_LSTM(features2, lengths2, 'embeddings', mode, params)
    
    final_state = tf.concat([s1, s2, tf.abs(s1 - s2), s1*s2], 1)
    logits = tf.layers.dense(final_state, 2, name="fc", reuse=tf.AUTO_REUSE)

    return logits

def stack_bidirectional_LSTM(features, lengths, scope, mode, params):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        cell = tf.contrib.rnn.LSTMBlockCell

        cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        if mode == tf.estimator.ModeKeys.TRAIN:
            cells_fw = [tf.contrib.rnn.DropoutWrapper(cell,
                input_keep_prob=1.0,
                output_keep_prob=1. - params.dropout,
                state_keep_prob=1. - params.dropout) for cell in cells_fw]
            cells_bw = [tf.contrib.rnn.DropoutWrapper(cell,
                input_keep_prob=1.0,
                output_keep_prob=1. - params.dropout,
                state_keep_prob=1. - params.dropout) for cell in cells_bw]
        
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=features,
            sequence_length=lengths,
            dtype=tf.float32)
        # batch_range = tf.cast(tf.range(0, params.batch_size), tf.int64)
        # indices = tf.stack([batch_range, lengths - 1], axis=1)
    # return tf.gather_nd(outputs, indices)
    return outputs


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
        s1_combined = tf.concat([tf.abs(s1_encoded - s2_aligned), s1_encoded*s2_aligned], -1, name="s1_concat")
        s2_combined = tf.concat([tf.abs(s2_encoded - s1_aligned), s2_encoded*s1_aligned], -1, name="s2_concat")
        s1_combined_ = tf.concat([s1_encoded, s2_aligned, s1_self], -1, name="s1_concat_")
        s2_combined_ = tf.concat([s2_encoded, s1_aligned, s2_self], -1, name="s2_concat_")

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
    
    with tf.name_scope("Comparision_Projection_"):
        compare_layers_ = []
        for idx, compare_dim in enumerate(params.compare_dim_):
            USE_BIAS = (idx != (len(params.compare_dim_) - 1))
            compare_layers_.extend([
                tf.layers.Dense(compare_dim, activation=tf.nn.relu, name="compare2_" + str(idx), _reuse=tf.AUTO_REUSE, use_bias=USE_BIAS),
                Dropout(params.dropout, training=(mode == tf.estimator.ModeKeys.TRAIN))
            ])
        s1_compare_ = s1_combined_
        s2_compare_ = s2_combined_
        for idx, layer in enumerate(compare_layers_):
            s1_compare_ = layer(s1_compare_)
            s2_compare_ = layer(s2_compare_)
        # s1_compare_ = s1_compare_*tf.expand_dims(tf.sequence_mask(lengths1, tf.reduce_max(lengths1), dtype=tf.float32), axis=-1)
        # s2_compare_ = s2_compare_*tf.expand_dims(tf.sequence_mask(lengths2, tf.reduce_max(lengths2), dtype=tf.float32), axis=-1)
        s1_compare_ = _masked_seq(s1_compare_, lengths1)
        s2_compare_ = _masked_seq(s2_compare_, lengths2)
        s1_compare_ = tf.reshape(s1_compare_, [params.batch_size, -1, params.compare_dim_[-1]])
        s2_compare_ = tf.reshape(s2_compare_, [params.batch_size, -1, params.compare_dim_[-1]])
    
    with tf.name_scope("CNN"):
        s1_conv = cnn_encoder(features1, lengths1, params)
        s2_conv = cnn_encoder(features2, lengths2, params)
        
    with tf.name_scope("Pooling"):
        s1_rep = Reducesum(axis=1)(s1_compare)
        s2_rep = Reducesum(axis=1)(s2_compare)
        s1_rep_ = Reducesum(axis=1)(s1_compare_)
        s2_rep_ = Reducesum(axis=1)(s2_compare_)
    
    with tf.name_scope("FC"):
        merged = tf.concat([s1_rep, s2_rep, s1_rep_, s2_rep_, s1_conv, s2_conv], axis=-1, name="rep_concat")
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

def cnn_encoder(inputs, lens, params):
    inputs_ = tf.expand_dims(inputs, -1)
    conv_0 = tf.layers.conv2d(inputs_, 2, [2, 356], name="conv_2h", reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
    conv_1 = tf.layers.conv2d(inputs_, 2, [3, 356], name="conv_3h", reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
    conv_2 = tf.layers.conv2d(inputs_, 2, [4, 356], name="conv_4h", reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

    maxpool_0 = tf.reduce_max(tf.reshape(conv_0, [params.batch_size, -1, 2]), axis=1)
    maxpool_1 = tf.reduce_max(tf.reshape(conv_1, [params.batch_size, -1, 2]), axis=1)
    maxpool_2 = tf.reduce_max(tf.reshape(conv_2, [params.batch_size, -1, 2]), axis=1)

    concat_tensor = tf.concat([maxpool_0, maxpool_1, maxpool_2], axis=1)

    return concat_tensor