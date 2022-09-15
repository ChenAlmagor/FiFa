import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from os.path import join
import logging
import pickle
import tensorflow_addons as tfa

def dense_to_sparse(dense_tensor, n_dim):
    zero_t = tf.zeros_like(dense_tensor)
    dense_final = tf.where(tf.greater_equal(dense_tensor, tf.ones_like(dense_tensor) * n_dim), zero_t, dense_tensor)
    indices = tf.to_int64(
        tf.transpose([tf.range(tf.shape(dense_tensor)[0]), tf.reshape(dense_final, [-1])]))
    values = tf.ones_like(dense_tensor, dtype=tf.float32)
    shape = [tf.shape(dense_tensor)[0], n_dim]
    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=shape
    )

def FiFa(features, labels, mode, params):
    V = []
    xv = {}

    feat_cnt = params["categorical_feature_counts"]
    feat_dims = params['feat_dims']
    print('Field Dim - {}'.format(feat_dims))
    feat_dim_sum = sum(feat_dims.values())

    for f_name in features.keys():
        if f_name in {'label', 'tag'} or params["feature_types"][f_name] == 'NUMERIC':
            continue
        sparse_t = dense_to_sparse(features[f_name], feat_cnt[f_name] + 1)
        v = tf.get_variable('v_%s' % f_name, dtype=tf.float32, initializer=tf.random_normal(
            shape=[feat_cnt[f_name] + 1, feat_dims[f_name]], mean=0.0, stddev=0.2))
        V.append(v)
        xv[f_name] = tf.sparse_tensor_dense_matmul(sparse_t, v)

    # linear term
    w_l = tf.get_variable('w_l', dtype=tf.float32,
                          initializer=tf.random_normal(shape=[feat_dim_sum, 1], mean=0.0, stddev=1.0))

    b = tf.get_variable('b', shape=[1], dtype=tf.float32)

    # concact to matrix [N, M * F]
    l_cat = tf.concat(list(xv.values()), 1)

    dim_factor = 5
    emb_dim = feat_dim_sum * dim_factor

    M1 = tf.get_variable('M1', dtype=tf.float32,
                         initializer=tf.random_normal(shape=[feat_dim_sum, emb_dim], mean=0.0, stddev=0.2))
    D = tf.get_variable('D', dtype=tf.float32, initializer=tf.random_normal(shape=[emb_dim, 1], mean=0.0, stddev=0.2))

    M1_bias = tf.get_variable('M1_b', dtype=tf.float32,
                              initializer=tf.random_normal(shape=[emb_dim], mean=0.0, stddev=0.2))

    xvM1 = tf.matmul(l_cat, M1) + M1_bias

    # xvM1_non_lin = tf.nn.relu(xvM1)
    xvM1_non_lin = tfa.activations.gelu(xvM1)


    D_xvM1_non_lin = tf.matmul(xvM1_non_lin, D)
    p = D_xvM1_non_lin


    # # Add the linear part and bias
    logits = tf.matmul(l_cat, w_l) + b + p

    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)

    y_prob = tf.sigmoid(logits)
    pred_class = tf.cast((y_prob >= 0.5), tf.bool)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_class,
            'probabilities': y_prob,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(loss)


    l2_loss = tf.nn.l2_loss(w_l) * params['l2_linear'] \
              + sum([tf.nn.l2_loss(v) * params['l2_latent'] for v in V]) \
              + tf.nn.l2_loss(M1) * params['l2_r'] \
              + tf.nn.l2_loss(D) * params['l2_linear']


    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=pred_class,
                                   name='acc_op1')
    auc = tf.metrics.auc(labels=labels,
                         predictions=y_prob,
                         name='auc_op1')

    metric_orig_loss = tf.metrics.mean(loss, name='orig_loss_op')
    metric_l2_loss = tf.metrics.mean(l2_loss, name='l2_loss_op')
    metrics = {'accuracy': accuracy, 'auc': auc, 'orig_loss': metric_orig_loss, 'l2_loss': metric_l2_loss}

    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


