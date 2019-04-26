# coding: utf-8
import functools
import tensorflow as tf

def expand(x, mode):
    x["len1"] = tf.expand_dims(tf.convert_to_tensor(x["len1"]), 0)
    x["len2"] = tf.expand_dims(tf.convert_to_tensor(x["len2"]), 0)
    if mode != tf.estimator.ModeKeys.PREDICT:
        x["label"] = tf.expand_dims(tf.convert_to_tensor(x["label"]), 0)
    return x

def deflate(x, mode):
    x["len1"] = tf.squeeze(x["len1"])
    x["len2"] = tf.squeeze(x["len2"])
    if mode != tf.estimator.ModeKeys.PREDICT:
        x["label"] = tf.squeeze(x["label"])
    return x

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
            "len1": tf.FixedLenFeature([], dtype=tf.int64),
            "len2": tf.FixedLenFeature([], dtype=tf.int64)
        }
        if mode != tf.estimator.ModeKeys.PREDICT:
            # The labels won't be available at inference time, so don't add them
            # to the list of feature_columns to be read.
            context_features["label"] = tf.FixedLenFeature(
                [], dtype=tf.int64)

        context, sequences = tf.parse_single_sequence_example(
            example_proto,
            context_features=context_features,
            sequence_features=sequence_features)
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     context["label"] = None
        context.update(sequences)

        return context

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
        # if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.repeat()
        # Preprocesses 10 files concurrently and interleaves records from each file.
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)
        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, mode=mode),
            num_parallel_calls=10)
        dataset = dataset.map(functools.partial(expand, mode=mode))
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
        }
        if mode != tf.estimator.ModeKeys.PREDICT:
            padded_shapes["label"] = 1
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        dataset = dataset.map(functools.partial(deflate, mode=mode))
        features = dataset.make_one_shot_iterator().get_next()
        if mode == tf.estimator.ModeKeys.PREDICT:
            return features, None
        return features, features["label"]

    return _input_fn


def get_input_tensors(features, labels):
    len1 = features["len1"]
    len2 = features["len2"]
    
    s1 = features["s1"]
    s2 = features["s2"]
    pos1 = features["pos1"]
    pos2 = features["pos2"]
    
    return s1, s2, pos1, pos2, len1, len2, labels


if __name__ == "__main__":
    input_fn = get_input_fn(tf.estimator.ModeKeys.EVAL, "data/train.tfrecords", 3)
    feature, label = input_fn()
    s1, s2, pos1, pos2, len1, len2, labels = get_input_tensors(feature, label)
    sess = tf.Session()
    for idx in range(5):
        if labels is None:
            s1_, len1_ = sess.run([s1, len1])
            label_ = labels
        else:
            s1_, len1_, label_ = sess.run([s1, len1, labels])
        print(s1_)
        print(len1_)
        print(label_)