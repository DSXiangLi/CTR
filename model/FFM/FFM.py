"""
Paper

Yuchin Juan，Yong Zhuang，Wei-Sheng Chin，Field-aware Factorization Machines for CTR Prediction

"""

from model.FFM.preprocess import build_features
import tensorflow as tf
import numpy as np
from utils import tf_estimator_model, add_layer_summary, build_estimator_helper

@tf_estimator_model
def model_fn(features, labels, mode, params):
    """
    Field_aware factorication machine for 2 classes classification
    """
    feature_columns, field_dict = build_features()

    field_dim = len(np.unique(list(field_dict.values())))

    input = tf.feature_column.input_layer(features, feature_columns)

    input_dim = input.get_shape().as_list()[-1]

    with tf.variable_scope('linear'):
        linear_term = tf.layers.dense( input, units=1 )
        add_layer_summary( 'linear_output', linear_term )

    with tf.variable_scope('field_aware_interaction'):
        init = tf.truncated_normal(shape = (input_dim, field_dim, params['factor_dim']))
        v = tf.get_variable('v', dtype = tf.float32, initializer = init, validate_shape = False)

        interaction_term = tf.constant(0, dtype =tf.float32)
        # iterate over all the combination of features
        for i in range(input_dim):
            for j in range(i+1, input_dim):
                interaction_term += tf.multiply(
                    tf.reduce_mean(tf.multiply(v[i, field_dict[j],: ], v[j, field_dict[i],:])) ,
                    tf.multiply(input[:,i], input[:,j])
                )
        interaction_term = tf.reshape(interaction_term, [-1,1])
        add_layer_summary( 'interaction_term', interaction_term )

    with tf.variable_scope('output'):
        y = tf.math.add(interaction_term, linear_term)
        add_layer_summary( 'output', y )

    return y

build_estimator = build_estimator_helper(
    model_fn = {
        'census':model_fn
    },
    params = {
         'census':{
            'learning_rate' :0.01,
            'factor_dim': 3
         }
    }
)
