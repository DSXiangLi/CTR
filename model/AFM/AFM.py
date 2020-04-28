"""
paper

Jun Xiao,H ao Ye ,2017, Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks
"""


import tensorflow as tf
import numpy as np
from config import *
from model.AFM.preprocess import build_features
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper

@tf_estimator_model
def model_fn(features, labels, mode, params):
    dense_feature, sparse_feature = build_features()
    dense = tf.feature_column.input_layer(features, dense_feature) # lz linear concat of embedding
    sparse = tf.feature_column.input_layer(features, sparse_feature)

    feature_size = len( dense_feature )
    embedding_size = dense_feature[0].variable_shape.as_list()[-1]
    embedding_matrix = tf.reshape( dense, [-1, feature_size, embedding_size] )  # batch * feature_size *emb_size

    with tf.variable_scope('Linear_part'):
        linear_output = tf.layers.dense(sparse, units=1)
        add_layer_summary( 'linear_output', linear_output )

    with tf.variable_scope('Elementwise_Interaction'):
        elementwise_list = []
        for i in range(feature_size):
            for j in range(i+1, feature_size):
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
    {'dense':model_fn},
     params = {'attention_factor':3,
                'dropout_rate':0.2,
               'learning_rate' :0.002,
               'hidden_units':[5,5]
            }
)