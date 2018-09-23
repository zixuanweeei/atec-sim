# coding: utf-8
import tensorflow as tf

def loss(output, label, params):
    output = tf.squeeze(output)
    label = tf.squeeze(label)
    if params.pos_weight > 1.0:
        output = tf.nn.softmax(output, axis=-1)
        logits = output[:, 1]
        label = tf.cast(label, logits.dtype)
        cross_entropy = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                label, logits, params.pos_weight))
    else:
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label, logits=output))
    return cross_entropy


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


def focal_loss_v2(logits, labels, weights=None, alpha=0.25, gamma=2.):
    with tf.name_scope("focal_loss"):
        output = tf.nn.softmax(logits, axis=-1)
        logits = output[:, 1]
        labels = tf.cast(labels, logits.dtype)
        FL = -alpha*((1 - labels)**tf.constant(gamma))*tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
            - (1 - alpha)*(labels**tf.constant(gamma))*tf.log(tf.clip_by_value(1. - logits, 1e-8, 1.0))
        return tf.reduce_mean(FL)