# coding: utf-8

import os
import argparse
import pandas as pd

def main():
    result_fold = []
    for one_fold_idx in range(FLAGS.kfold):
        one_fold_filepath = FLAGS.result_pattern + "_" + str(one_fold_idx)
        result_fold.append(pd.read_csv(one_fold_filepath, header=None, index_col=0, sep="\t"))

    all_fold_result = pd.concat(result_fold, axis=1)
    avg_fold_result = all_fold_result.mean(axis=1)
    label = (avg_fold_result > 0.4478).astype(int)
    label.to_csv(FLAGS.result_name, header=None, index=True, sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--result_pattern", 
        type=str,
        default="",
        help="The files to be concatenated and ensembling.")
    parser.add_argument("--workspace", 
        type=str,
        default="./data/",
        help="The file to store all of the data.")
    parser.add_argument("--result_name", 
        type=str,
        default="",
        help="The file to store result.")
    parser.add_argument("--kfold", 
        type=int,
        default=5,
        help="CV fold.")

    FLAGS, unparsed = parser.parse_known_args()
    main()
    