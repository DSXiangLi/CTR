import tensorflow as tf
from utils import add_layer_summary

def sparse_embedding(feature_size, embedding_size, field_size, feat_ids, feat_vals, add_summary):
    with tf.variable_scope('Sparse_Embedding'):
        v = tf.get_variable( shape=[feature_size, embedding_size],
                             initializer=tf.truncated_normal_initializer(),
                             name='embedding_weight' )

        embedding_matrix = tf.nn.embedding_lookup( v, feat_ids ) # batch * field_size * embedding_size
        embedding_matrix = tf.multiply( embedding_matrix, tf.reshape(feat_vals, [-1, field_size,1] ) )

        if add_summary:
            add_layer_summary( 'embedding_matrix', embedding_matrix )

    return embedding_matrix


def sparse_linear(feature_size, feat_ids, feat_vals, add_summary):
    with tf.variable_scope('Linear_output'):
        weight = tf.get_variable( shape=[feature_size],
                             initializer=tf.truncated_normal_initializer(),
                             name='linear_weight' )
        bias = tf.get_variable( shape=[1],
                             initializer=tf.glorot_uniform_initializer(),
                             name='linear_bias' )

        linear_output = tf.nn.embedding_lookup( weight, feat_ids )
        linear_output = tf.reduce_sum( tf.multiply( linear_output, feat_vals ), axis=1, keepdims=True )
        linear_output = tf.add( linear_output, bias )

        if add_summary:
            add_layer_summary('linear_output', linear_output)

    return linear_output


def stack_dense_layer(dense, hidden_units, dropout_rate, batch_norm, mode, add_summary):
    with tf.variable_scope('Dense'):
        for i, unit in enumerate(hidden_units):
            dense = tf.layers.dense(dense, units = unit, activation = 'relu',
                                    name = 'dense{}'.format(i))
            if batch_norm:
                dense = tf.layers.batch_normalization(dense, center = True, scale = True,
                                                      trainable = True,
                                                      training = (mode == tf.estimator.ModeKeys.TRAIN))
            if dropout_rate > 0:
                dense = tf.layers.dropout(dense, rate = dropout_rate,
                                          training = (mode == tf.estimator.ModeKeys.TRAIN))

            if add_summary:
                add_layer_summary(dense.name, dense)

    return dense
