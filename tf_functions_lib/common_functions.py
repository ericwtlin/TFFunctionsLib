# -*- coding: utf-8 -*-
import tensorflow as tf


def elementwise_walking_operation_2d(vectors_a, vectors_b, op):
    """ element-wise walking operation between every two element of vectors_a and vectors_b

    Args:
        vectors_a: 2d Tensor with shape, [batch_size, vector_len]
        vectors_b: 2d Tensor with shape, [batch_size, vector_len]
        op: operation, e.g.  tf.subtract, tf.add, or some customized operations

    Returns:
        3d Tensor with shape [batch_size, vector_len, vector_len]
        result[k, i, j] indicates the result of some operation between vectors_a[k, i] and
            vectors_b[k, j]
    """
    shape = tf.shape(vectors_a)
    vectors_a_list = tf.unstack(vectors_a, axis=1)
    results = []
    for vector_a in vectors_a_list:
        vector_a_tiled = tf.reshape(tf.tile(vector_a, [shape[1]]), [-1, shape[1]])
        results.append(op(vector_a_tiled, vectors_b))

    result = tf.stack(results, axis=1)
    return result


def elementwise_walking_operation_3d(matrices_a, matrices_b, op):
    """ element wise operation between every two element of matrices_a and matrices_a

    Args:
        matrices_a: 3d Tensor with shape. E.g. [batch_size, vector_num, vector_len]
        matrices_b: 3d Tensor with shape. E.g. [batch_size, vector_num, vector_len]
        op: operation, e.g.  tf.subtract, tf.add, or some customized operations

    Returns:
        the result of some operation between matrices_a[k, i, :] and matrices_b[k, j, :]
        the result dimension depends on the parameter op.
    """
    shape = tf.shape(matrices_a)
    matrics_a_list = tf.unstack(matrices_a, axis=1)
    results = []
    for matrix_a in matrics_a_list:
        matrix_a_tiled = tf.reshape(tf.tile(matrix_a, [shape[1], 1]), [-1, shape[1], shape[2]])
        results.append(op(matrix_a_tiled, matrices_b))

    result = tf.stack(results, axis=1)
    return result
