# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from model import SiameseModel
from input_helper import get_input_fn

def main(unused_args):
    model = SiameseModel(FLAGS)
    print("Start to build model ...", end="")
    estimator, train_spec, eval_spec = model.create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=300,
            save_summary_steps=100))
    print("Done!")
    if FLAGS.task == "train":
        print("Start to train and evaluate ...")
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif FLAGS.task == "predict":
        print("Start to predict ...")
        predictions = estimator.predict(
            input_fn=get_input_fn(mode=tf.estimator.ModeKeys.PREDICT,
                tfrecord_pattern=FLAGS.test_data,
                batch_size=FLAGS.batch_size))
        logits = pd.DataFrame(np.array([predict["logits"] for predict in predictions]))
        logits.columns = ["prob0", "prob1"]
        result = (logits.prob1 > 0.5).astype(int)
        eval_ptr = FLAGS.test_data.split(".")[0]
        eval_file = eval_ptr + ".csv"
        pair_stc = pd.read_csv(eval_file, header=None, sep="\t", index_col=0)
        pair_stc.iloc[:, -1] = result.values
        pair_stc.to_csv(FLAGS.result_name, header=None, sep="\t")
    elif FLAGS.task == "predict_proba":
        print("Start to predict probability...")
        predictions = estimator.predict(
            input_fn=get_input_fn(mode=tf.estimator.ModeKeys.PREDICT,
                tfrecord_pattern=FLAGS.test_data,
                batch_size=FLAGS.batch_size))
        predictions = list(predictions)
        logits = pd.DataFrame(np.array([predict["logits"] for predict in predictions]))
        logits.columns = ["prob0", "prob1"]
        result = logits.prob1
        result.index += 1
        result.to_csv(FLAGS.result_name, header=None, sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--training_data",
        type=str,
        default="./data/train.tfrecords",
        help="Path to training data (tf.Example in TFRecord format)")
    parser.add_argument(
        "--eval_data",
        type=str,
        default="./data/eval.tfrecords",
        help="Path to evaluation data (tf.Example in TFRecord format)")
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/atec_nlp_sim_test.tfrecords",
        help="Path to test data (tf.Example in TFRecord format)")
    parser.add_argument(
        "--submission_dir",
        type=str,
        default="./submission",
        help="Path to store the result")
    parser.add_argument(
        "--classes_file",
        type=str,
        default="",
        help="Path to a file with the classes - one class per line")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of recurrent neural network layers.")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=300,
        help="Number of node per recurrent network layer.")
    parser.add_argument(
        "--num_conv",
        type=str,
        default="[48, 64, 96]",
        help="Number of conv layers along with number of filters per layer.")
    parser.add_argument(
        "--conv_len",
        type=str,
        default="[5, 5, 3]",
        help="Length of the convolution filters.")
    parser.add_argument(
        "--cell_type",
        type=str,
        default="lstm",
        help="Cell type used for rnn layers: cudnn_lstm, lstm or block_lstm.")
    parser.add_argument(
        "--batch_norm",
        type="bool",
        default="False",
        help="Whether to enable batch normalization or not.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate used for training.")
    parser.add_argument(
        "--gradient_clipping_norm",
        type=float,
        default=9.0,
        help="Gradient clipping norm used during training.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout used for convolutions and bidi lstm layers.")
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Number of training steps.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size to use for training/evaluation.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model_ckp",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--self_test",
        type="bool",
        default="False",
        help="Whether to enable batch normalization or not.")
    parser.add_argument(
        "--wordvector",
        type=str,
        default="data/local_wv.txt",
        help="The file contains the word2vector config.")
    parser.add_argument(
        "--task",
        type=str,
        default="predict",
        help="Prediction or train")
    parser.add_argument(
        "--result_name",
        type=str,
        default="./result.csv",
        help="Filename to store the result.")
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Positive samples weights in loss function.")
    parser.add_argument(
        "--projection_hidden",
        type=int,
        default=0,
        help="Dimension of hidden projection layer.")
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=300,
        help="Dimension of projection layer.")
    parser.add_argument(
        "--compare_dim",
        type=str,
        default="[300, 32]",
        help="Sentence compare layers dimension.")
    parser.add_argument(
        "--compare_dim_",
        type=str,
        default="[300, 32]",
        help="Sentence compare layers dimension.")
    parser.add_argument(
        "--dense_dim",
        type=str,
        default="[100, 32]",
        help="Sentence compare layers dimension.")
    parser.add_argument(
        "--print_logits",
        type="bool",
        default="false",
        help="Sentence compare layers dimension.")
    

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
