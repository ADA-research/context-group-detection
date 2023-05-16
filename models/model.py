import keras as keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, LSTM, concatenate
from keras.models import Model


def conv(filters, reg, name=None):
    return Conv2D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
                  use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)


def build_model(context_size, features, consecutive_frames):
    context_inputs = keras.layers.Input(shape=(context_size, consecutive_frames, features), name='context')
    pair_inputs = keras.layers.Input(shape=(2, consecutive_frames, features), name='pair')

    # LSTM branch 1
    lstm1 = LSTM(64)(pair_inputs)
    dense1 = Dense(32)(lstm1)

    # LSTM branch 2
    lstm2 = LSTM(64)(context_inputs)
    dense2 = Dense(32)(lstm2)

    # Concatenate the outputs of the two branches
    concatenated = concatenate([dense1, dense2])

    # Output layer
    output = Dense(1)(concatenated)

    # Create the model with two inputs and one output
    model = Model(inputs=[pair_inputs, context_inputs], outputs=output)

    # Compile the model
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model


if __name__ == '__main__':
    context_size, consecutive_frames, features = 8, 10, 4

    data_filename = '../datasets/reformatted/eth_10_10_data.npy'
    data = np.load(data_filename)
    X_train_pair = data[:, :2]
    X_train_context = data[:, 2:]
    labels_filename = '../datasets/reformatted/eth_10_10_labels.npy'
    Y_train = np.load(labels_filename)

    model = build_model(context_size, features, consecutive_frames)

    model.fit([X_train_pair, X_train_context], Y_train,
              epochs=10, batch_size=1024,
              # validation_data=(X_val, Y_val)
              )
