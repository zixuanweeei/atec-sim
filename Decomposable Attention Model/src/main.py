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
                batch_size=FLAGS.batch_size)
        )
        # result = pd.DataFrame({"label": [predict["predictions"] for predict in predictions]})
        # if FLAGS.print_logits:
        #     logits = pd.DataFrame(np.array([predict["logits"] for predict in predictions]))
        #     logits.columns = ["prob0", "prob1"]
        #     result = pd.concat([result, logits], axis=1)
        logits = pd.DataFrame(np.array([predict["logits"] for predict in predictions]))
        logits.columns = ["prob0", "prob1"]
        result = (logits.prob1 > 0.29).astype(int)
        result.index += 1
        result.to_csv(FLAGS.result_name, header=None, sep="\t")
    elif FLAGS.task == "predict_proba":
        print("Start to predict probability...")
        predictions = estimator.predict(
            input_fn=get_input_fn(mode=tf.estimator.ModeKeys.PREDICT,
                tfrecord_pattern=FLAGS.test_data,
                batch_size=FLAGS.batch_size)
        )
        predictions = list(predictions)
        logits = pd.DataFrame(np.array([predict["logits"] for predict in predictions]))
        logits.columns = ["prob0", "prob1"]
        result = logits.prob1
        result.index += 1
        result.to_csv(FLAGS.result_name, header=None, sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--training_data", type=str, default="./data/train.csv")
    parser.add_argument("--eval_data", type=str, default="./data/train.csv")
    parser.add_argument("--test_data", type=str, default="./data/atec_nlp_sim_test.tfrecords")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--gradient_clipping_norm", type=float, default=9.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_dir", type=str, default="model_ckp")
    parser.add_argument("--task", type=str, default="predict")
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dense_dim", type=str, default="[400, 100]")
    parser.add_argument("--num_hidden_layers", type=str, default=6)
    parser.add_argument("--print_logits", type="bool", default="false")
    parser.add_argument("--layer_postprocess_dropout", type=float, default=0.05)
    parser.add_argument("--relu_dropout", type=float, default=0.05)
    parser.add_argument("--attention_dropout", type=float, default=0.05)
    parser.add_argument("--filter_size", type=int, default=2048)
    parser.add_argument("--allow_ffn_pad", type="bool", default="true")

    parser.add_argument("--wordvector", type=str, default="data/local_wv.txt")
    

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
