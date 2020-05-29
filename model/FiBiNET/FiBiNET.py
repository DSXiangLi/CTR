"""
paper

Tongwen Huang, 2019, FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction

"""

import tensorflow as tf
import numpy as np
from config import *
from model.DeepFM.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper
from layers import stack_dense_layer, sparse_embedding, sparse_linear


def Bilinear_layer(embedding_matrix, field_size, emb_size, type, name):
    # Bilinear_layer: combine inner and element-wise product
    interaction_list = []
    with tf.variable_scope('BI_interaction_{}'.format(name)):
        if type == 'field_all':
            weight = tf.get_variable( shape=(emb_size, emb_size), initializer=tf.truncated_normal_initializer(),
                                      name='Bilinear_weight_{}'.format(name) )
        for i in range(field_size):
            if type == 'field_each':
                weight = tf.get_variable( shape=(emb_size, emb_size), initializer=tf.truncated_normal_initializer(),
                                          name='Bilinear_weight_{}_{}'.format(i, name) )
            for j in range(i+1, field_size):
                if type == 'field_interaction':
                    weight = tf.get_variable( shape=(emb_size, emb_size), initializer=tf.truncated_normal_initializer(),
                                          name='Bilinear_weight_{}_{}_{}'.format(i,j, name) )
                vi = tf.gather(embedding_matrix, indices = i, axis =1, batch_dims =0, name ='v{}'.format(i)) # batch * emb_size
                vj = tf.gather(embedding_matrix, indices = j, axis =1, batch_dims =0, name ='v{}'.format(j)) # batch * emb_size
                pij = tf.matmul(tf.multiply(vi,vj), weight) # bilinear : vi * wij \odot vj
                interaction_list.append(pij)

        combination = tf.stack(interaction_list, axis =1 ) # batch * emb_size * (Field_size * (Field_size-1)/2)
        combination = tf.reshape(combination, shape = [-1, int(emb_size * (field_size * (field_size-1) /2)) ]) # batch * ~
        add_layer_summary( 'bilinear_output', combination )

    return combination


def SENET_layer(embedding_matrix, field_size, emb_size, pool_op, ratio):
    with tf.variable_scope('SENET_layer'):
        # squeeze embedding to scaler for each field
        with tf.variable_scope('pooling'):
            if pool_op == 'max':
                z = tf.reduce_max(embedding_matrix, axis=2) # batch * field_size * emb_size -> batch * field_size
            else:
                z = tf.reduce_mean(embedding_matrix, axis=2)
            add_layer_summary('pooling scaler', z)

        # excitation learn the weight of each field from above scaler
        with tf.variable_scope('excitation'):
            z1 = tf.layers.dense(z, units = field_size//ratio, activation = 'relu')
            a = tf.layers.dense(z1, units= field_size, activation = 'relu') # batch * field_size
            add_layer_summary('exciitation weight', a )

        # re-weight embedding with weight
        with tf.variable_scope('reweight'):
            senet_embedding = tf.multiply(embedding_matrix, tf.expand_dims(a, axis = -1)) # (batch * field * emb) * ( batch * field * 1)
            add_layer_summary('senet_embedding', senet_embedding) # batch * field_size * emb_size

        return senet_embedding

@tf_estimator_model
def model_fn_dense(features, labels, mode, params):
    dense_feature, sparse_feature = build_features()
    dense_input = tf.feature_column.input_layer(features, dense_feature)
    sparse_input = tf.feature_column.input_layer(features, sparse_feature)

    # Linear part
    with tf.variable_scope('Linear_component'):
        linear_output = tf.layers.dense( sparse_input, units=1 )
        add_layer_summary( 'linear_output', linear_output )

    field_size = len(dense_feature)
    emb_size = dense_feature[0].variable_shape.as_list()[-1]
    embedding_matrix = tf.reshape(dense_input, [-1, field_size, emb_size])

    # SENET_layer to get new embedding matrix
    senet_embedding_matrix = SENET_layer(embedding_matrix, field_size, emb_size,
                                         pool_op = params['pool_op'], ratio= params['senet_ratio'])

    # combination layer & BI_interaction
    BI_org = Bilinear_layer(embedding_matrix, field_size, emb_size, type = params['bilinear_type'], name = 'org')
    BI_senet = Bilinear_layer(senet_embedding_matrix, field_size, emb_size, type = params['bilinear_type'], name = 'senet')

    combination_layer = tf.concat([BI_org, BI_senet] , axis =1)

    # Deep part
    dense_output = stack_dense_layer(combination_layer, params['hidden_units'],
                               params['dropout_rate'], params['batch_norm'],
                               mode, add_summary=True )

    with tf.variable_scope('output'):
        y = dense_output + linear_output
        add_layer_summary( 'output', y )

    return y


@tf_estimator_model
def model_fn_sparse(features, labels, mode, params):
    # hyper parameter
    field_size = FRAPPE_PARAMS['field_size']
    feature_size = FRAPPE_PARAMS['feature_size']
    embedding_size = FRAPPE_PARAMS['embedding_size']

    # extract feature
    feat_ids = tf.reshape(features['feat_ids'], shape = [-1, field_size]) # batch * field_size
    feat_vals = tf.reshape(features['feat_vals'], shape = [-1, field_size]) # batch * field_size

    # extract embedding
    with tf.variable_scope('extract_embedding'):
        embedding_matrix = sparse_embedding( feature_size, embedding_size, field_size,
                                             feat_ids, feat_vals, add_summary =True) # (batch, field_size, embedding_size)

    # linear part
    linear_output = sparse_linear( feature_size, feat_ids, feat_vals, add_summary = True )

    # SENET_layer to get new embedding matrix
    senet_embedding_matrix = SENET_layer(embedding_matrix, field_size, embedding_size,
                                         pool_op = params['pool_op'], ratio= params['senet_ratio'])

    # combination layer & BI_interaction
    BI_org = Bilinear_layer(embedding_matrix, field_size, embedding_size, type = params['bilinear_type'], name = 'org')
    BI_senet = Bilinear_layer(senet_embedding_matrix, field_size, embedding_size, type = params['bilinear_type'], name = 'senet')

    combination_layer = tf.concat([BI_org, BI_senet] , axis =1)

    # Deep part
    dense_output = stack_dense_layer(combination_layer, params['hidden_units'],
                               params['dropout_rate'], params['batch_norm'],
                               mode, add_summary=True )

    with tf.variable_scope('output'):
        y = dense_output + linear_output
        add_layer_summary( 'output', y )

    return y


build_estimator = build_estimator_helper(
    model_fn = {
        'dense' : model_fn_dense,
        'sparse': model_fn_sparse
    },
    params = {
         'dense': {
            'dropout_rate': 0.2,
            'learning_rate' : 0.001,
            'hidden_units': [20,10,1],
            'batch_norm': True,
            'cin_layer_size': [8,4,4],
             'pool_op': 'avg',
             'senet_ratio': 2,
             'bilinear_type': 'field_all' # support field_all / field_each / field_interaction
            },
        'sparse': {
            'dropout_rate': 0.2,
            'learning_rate': 0.002,
            'hidden_units': [128, 64, 32, 1],
            'batch_norm': True,
            'cin_layer_size': [32,16,8],
            'pool_op': 'avg',
            'senet_ratio': 2,
            'bilinear_type': 'field_all'  # support field_all / field_each / field_interaction

        }
    }
)

