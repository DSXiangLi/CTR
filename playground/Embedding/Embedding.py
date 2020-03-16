"""
SVD matrix decomposition

"""

from tensorflow.keras.layers import Layer, InputSpec, Input, Dense, Embedding, Reshape, Dot
from tensorflow.keras import activations, Model
# df =pd.read_csv('./data/netflix_movie_rating/train.csv')

def fm_embedding(num_user, num_movie, k):
    input_user = Input(name = 'user_input', shape = [None, ], dtype = 'int32')
    embedding_user = Embedding(name = 'user_embedding', input_dim = num_user,
                                  output_dim = k, input_length = 1)(input_user)
    embedding_user = Reshape((k,))(embedding_user)

    input_movie = Input(name = 'movie_input', shape= [None, ], dtype = 'int32')
    embedding_movie = Embedding(name = 'movie_embedding', input_dim = num_movie,
                                  output_dim = k, input_length = 1)(input_movie)
    embedding_movie = Reshape((k,))(embedding_movie)

    out = Dot(name = 'inner_product', axes=1, normalize= False)([embedding_user, embedding_movie ])

    model = Model(inputs = [input_user, input_movie], outputs= out)
    model.compile(loss = 'mse', optimizer= 'Adam')

    model.summary()

    return model


if __name__ == '__main__':
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from tensorflow.keras.callbacks import TensorBoard

    df = pd.read_csv('./data/movie_len/ratings.csv')

    df = df.loc[:, ['userId', 'movieId', 'rating']]
    df.columns = ['user_id', 'movie_id', 'rating']
    target = 'rating'

    num_user = len(np.unique(df['user_id'] )) + 1
    num_movie = len(np.unique(df['movie_id'] )) + 1

    model = fm_embedding(num_user, num_movie, k=8)

    ## Train goes here

    train, test = train_test_split( df,
                                    test_size=0.2, random_state=1234 )

    model.fit( x = [train.loc[:,'user_id'].values,
                    train.loc[:,'movie_id'].values],
               y = train[target].values,
                  batch_size=256,
                  epochs=1,
                  validation_split=0.1,
                  shuffle=True)

    tensorboard = TensorBoard( log_dir='./log/Embedding_autoencoder',
                               histogram_freq=1,
                               write_graph=True,
                               write_grads=True,
                               write_image=True,
                               embeddings_freq=0,
                               embeddings_layer_names=None )

    ## Measure model performance
    y_pred = model.predict([test.loc[:,'user_id'].values,
                            test.loc[:,'movie_id'].values])
    y_true = test['rating'].values

    rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))

    print ('Mean Squared Error = {:.2f}'.format(rmse))