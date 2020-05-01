"""
paper

Jun Xiao, Hao Ye ,2017, Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks
"""


import tensorflow as tf
import numpy as np
from config import *
from model.AFM.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper
from layers import sparse_embedding, sparse_linear, stack_dense_layer

@tf_estimator_model
def model_fn_dense(features, labels, mode, params):
    dense_feature, sparse_feature = build_features()
    dense = tf.feature_column.input_layer(features, dense_feature) # lz linear concat of embedding
    sparse = tf.feature_column.input_layer(features, sparse_feature)

    field_size = len( dense_feature )
    embedding_size = dense_feature[0].variable_shape.as_list()[-1]
    embedding_matrix = tf.reshape( dense, [-1, field_size, embedding_size] )  # batch * field_size *emb_size

    with tf.variable_scope('Linear_part'):
        linear_output = tf.layers.dense(sparse, units=1)
        add_layer_summary( 'linear_output', linear_output )

    with tf.variable_scope('Elementwise_Interaction'):
        elementwise_list = []
        for i in range(field_size):
            for j in range(i+1, field_size):
                vi = tf.gather(embedding_matrix, indices=i, axis=1, batch_dims=0,name = 'vi') # batch * emb_size
                vj = tf.gather(embedding_matrix, indices=j, axis=1, batch_dims=0,name = 'vj')
                elementwise_list.append(tf.multiply(vi,vj)) # batch * emb_size
        elementwise_matrix = tf.stack(elementwise_list) # (N*(N-1)/2) * batch * emb_size
        elementwise_matrix = tf.transpose(elementwise_matrix, [1,0,2]) # batch * (N*(N-1)/2) * emb_size

    with tf.variable_scope('Attention_Net'):
        # 2 fully connected layer
        dense = tf.layers.dense(elementwise_matrix, units = params['attention_factor'], activation = 'relu') # batch * (N*(N-1)/2) * t
        add_layer_summary( dense.name, dense )
        attention_weight = tf.layers.dense(dense, units=1, activation = 'softmax') # batch *(N*(N-1)/2) * 1
        add_layer_summary( attention_weight.name, attention_weight)

    with tf.variable_scope('Attention_pooling'):
        interaction_output = tf.reduce_sum(tf.multiply(elementwise_matrix, attention_weight), axis=1) # batch * emb_size
        interaction_output = tf.layers.dense(interaction_output, units=1) # batch * 1

    with tf.variable_scope('output'):
        y = interaction_output + linear_output
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
    embedding_matrix = sparse_embedding(feature_size, embedding_size, field_size,
                                        feat_ids, feat_vals, add_summary = True)

    # linear output
    linear_output = sparse_linear(feature_size, feat_ids, feat_vals, add_summary = True)

    with tf.variable_scope('Elementwise_Interaction'):
        elementwise_list = []
        for i in range(field_size):
            for j in range(i+1, field_size):
                vi = tf.gather(embedding_matrix, indices=i, axis=1, batch_dims=0,name = 'vi') # batch * emb_size
                vj = tf.gather(embedding_matrix, indices=j, axis=1, batch_dims=0,name = 'vj')
                elementwise_list.append(tf.multiply(vi,vj)) # batch * emb_size
        elementwise_matrix = tf.stack(elementwise_list) # (N*(N-1)/2) * batch * emb_size
        elementwise_matrix = tf.transpose(elementwise_matrix, [1,0,2]) # batch * (N*(N-1)/2) * emb_size

    with tf.variable_scope('Attention_Net'):
        # 2 fully connected layer
        dense = tf.layers.dense(elementwise_matrix, units = params['attention_factor'], activation = 'relu') # batch * (N*(N-1)/2) * t
        add_layer_summary( dense.name, dense )
        attention_weight = tf.layers.dense(dense, units=1, activation = 'softmax') # batch *(N*(N-1)/2) * 1
        add_layer_summary( attention_weight.name, attention_weight)

    with tf.variable_scope('Attention_pooling'):
        interaction_output = tf.reduce_sum(tf.multiply(elementwise_matrix, attention_weight), axis=1) # batch * k
        interaction_output = tf.layers.dense(interaction_output, units=1) # batch * 1

    with tf.variable_scope('output'):
        y = interaction_output + linear_output
        add_layer_summary( 'output', y )

    return y



build_estimator = build_estimator_helper(
    model_fn = {
        'dense' : model_fn_dense,
        'sparse' : model_fn_sparse
    },
     params = {
         'dense':{
             'attention_factor':3,
             'dropout_rate':0.2,
             'learning_rate' :0.002
            },
         'sparse':{
             'attention_factor': 16,
             'dropout_rate': 0.2,
             'learning_rate': 0.002,
             'hidden_units': [128, 64, 1]
         }
     }
)