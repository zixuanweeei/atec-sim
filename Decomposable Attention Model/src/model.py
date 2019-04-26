# coding: utf-8
from __future__ import absolute_import, division, print_function
import argparse
import codecs
import ast
import io
import tensorflow as tf
import numpy as np
# from tqdm import tqdm

from inference import inference_v2
from loss import focal_loss
from input_helper import get_input_fn, get_input_tensors

class SiameseModel(object):
    def __init__(self, FLAGS):
        with io.open(FLAGS.wordvector, "r", encoding="utf-8") as reader:
            info = reader.readline().split(" ")
            words_num = int(info[0])
            vector_dim = int(info[1])
            self.W = np.zeros((words_num + 1, vector_dim))
            for idx, line in enumerate(reader.readlines()):
                ele = line.rstrip().split(" ")
                self.W[idx + 1, ] = list(map(float, ele[1:]))
        
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
            embedded_pos1 = tf.one_hot(pos1, 56, name="POS_embedding_1")
            embedded_pos2 = tf.one_hot(pos2, 56, name="POS_embedding_2")
            
            embedded_s1 = tf.reshape(embedded_s1, [params.batch_size, -1, 300])
            embedded_s2 = tf.reshape(embedded_s2, [params.batch_size, -1, 300])
            embedded_pos1 = tf.reshape(embedded_pos1, [params.batch_size, -1, 56])
            embedded_pos2 = tf.reshape(embedded_pos2, [params.batch_size, -1, 56])

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
            num_layers=self.FLAGS.num_layers,
            num_nodes=self.FLAGS.num_nodes,
            batch_size=self.FLAGS.batch_size,
            num_conv=ast.literal_eval(self.FLAGS.num_conv),
            conv_len=ast.literal_eval(self.FLAGS.conv_len),
            num_classes=2,
            learning_rate=self.FLAGS.learning_rate,
            gradient_clipping_norm=self.FLAGS.gradient_clipping_norm,
            cell_type=self.FLAGS.cell_type,
            batch_norm=self.FLAGS.batch_norm,
            dropout=self.FLAGS.dropout,
            pos_weight=self.FLAGS.pos_weight,
            projection_hidden=self.FLAGS.projection_hidden,
            projection_dim=self.FLAGS.projection_dim,
            compare_dim=ast.literal_eval(self.FLAGS.compare_dim),
            compare_dim_=ast.literal_eval(self.FLAGS.compare_dim_),
            dense_dim=ast.literal_eval(self.FLAGS.dense_dim))

        estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=run_config,
            params=model_params)

        train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(
            mode=tf.estimator.ModeKeys.TRAIN,
            tfrecord_pattern=self.FLAGS.training_data,
            batch_size=self.FLAGS.batch_size), max_steps=self.FLAGS.steps)

        eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(
            mode=tf.estimator.ModeKeys.EVAL,
            tfrecord_pattern=self.FLAGS.eval_data,
            batch_size=self.FLAGS.batch_size), steps=10000//self.FLAGS.batch_size, throttle_secs=300)

        return estimator, train_spec, eval_spec


def f1_score(predictions=None, recalls=None, weights=None):
    P, update_op1 = predictions
    R, update_op2 = recalls
    eps = 1e-5
    return (2*(P*R)/(P+R+eps), tf.group(update_op1, update_op2))