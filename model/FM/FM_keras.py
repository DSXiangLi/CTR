"""
Paper

S. Rendle, “Factorization machines,” in Proceedings of IEEE International Conference on Data Mining (ICDM), pp. 995–1000, 2010

"""

from tensorflow.keras.layers import Layer, InputSpec, Input, Dense
from tensorflow.keras import activations, Model

import tensorflow.keras.backend as K
from config import *
from model.FM.preprocess import build_features

class FM_Layer( Layer ):
    """
    Input:
        factor_dim: latent vector size
        input_shape: raw feature size
        activation
    output:
        FM layer output
    """

    def __init__(self, factor_dim,  **kwargs):
        self.factor_dim = factor_dim
        self.InputSepc = InputSpec( ndim=2 )  # Specifies input layer attribute. one Inspec for each input
        super( FM_Layer, self ).__init__( **kwargs )

    def build(self, input_shape):
        """
        input:
            tuple of input_shape
        func:
            define all the necessary variable here
        """
        assert len( input_shape ) >= 2
        input_dim = int( input_shape[-1] )

        self.w = self.add_weight( name='w0', shape=(input_dim, 1),
                                  initializer='truncated_normal',
                                  trainable=True )

        self.b = self.add_weight( name='bias', shape=(1,),
                                  initializer='zeros',
                                  trainable=True )

        self.v = self.add_weight( name='hidden_vector', shape=(input_dim, self.factor_dim),
                                  initializer='truncated_normal',
                                  trainable=True )

        super( FM_Layer, self ).build( input_shape )  # set self.built=True

    def call(self, x):
        """
        input:
            x(previous layer output)
        func:
            core calculcation of layer goes here
        """
        linear_term = K.dot( x, self.w ) + self.b

        # Embedding之和，Embedding内积： (1, input_dim) * (input_dim, factor_dim) = (1, factor_dim)
        sum_square = K.pow( K.dot( x, self.v ), 2 )
        square_sum = K.dot( K.pow( x, 2 ), K.pow( self.v, 2 ) )

        # (1, factor_dim) -> (1)
        quad_term = K.mean( (sum_square - square_sum), axis=1, keepdims=True )

        tf.summary.histogram('quad_term', quad_term)
        output = linear_term + quad_term
        tf.summary.histogram('output', output)

        return output

    def compute_output_shape(self, input_shape):
        # Attention: tf.keras回传input_shape是tf.dimension而不是tuple, 所以要cast成int
        return (int(input_shape[0]), 1)

    def get_config(self):
        """
        for custom Layer to be serializable
        """
        config = super( FM_Layer, self ).get_config()
        config.update( {'factor_dim': self.factor_dim} )
        return config

def model_fn():
    # build Keras Model

    # use feature_column as keras input
    input = {}
    for f in FEATURE_NAME:
        if f != TARGET:
            input[f] = Input(shape=(1,), name = f, dtype = DTYPE[f])

    feature_columns = build_features()
    feature_layer = tf.keras.layers.DenseFeatures( feature_columns )

    dense_feature = feature_layer(input)

    fm = FM_Layer(name = 'fm_layer',  factor_dim = 8)(dense_feature)

    tf.summary.histogram('fm_output', fm)

    output = Dense(1, activation='sigmoid', name = 'output')(fm)

    model = Model(inputs = [i for i in input.values()], outputs = output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics=['binary_accuracy','AUC']
    )
    print( model.summary())

    return model

def build_estimator(model_dir, **kwargs):
    # keras model -> tf.estimator

    model = model_fn()

    run_config = tf.estimator.RunConfig(
        save_summary_steps=10,
        log_step_count_steps=10,
        keep_checkpoint_max = 3,
        save_checkpoints_steps = 10
    )
    # Avoid checkpoint
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir= model_dir, config = run_config )

    return estimator