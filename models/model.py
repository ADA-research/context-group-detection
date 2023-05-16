import keras as keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, LSTM, concatenate
from keras.models import Model


def conv(filters, reg, name=None):
    return Conv2D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
                  use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)


def build_model(context_size, features, consecutive_frames):
    inputs = []
    pair_first_input = keras.layers.Input(shape=(consecutive_frames, features), name='pair_first')
    inputs.append(pair_first_input)
    pair_second_input = keras.layers.Input(shape=(consecutive_frames, features), name='pair_second')
    inputs.append(pair_second_input)
    context_inputs = []
    for i in range(context_size):
        context_input = keras.layers.Input(shape=(consecutive_frames, features), name='context_{}'.format(i))
        context_inputs.append(context_input)
        inputs.append(context_input)

    denses = []
    # LSTM branch 1
    pair_first_lstm = LSTM(64, batch_input_shape=(10, 4))(pair_first_input)
    dense1 = Dense(32)(pair_first_lstm)
    denses.append(dense1)

    # LSTM branch 2
    pair_second_lstm = LSTM(64, batch_input_shape=(10, 4))(pair_second_input)
    dense2 = Dense(32)(pair_second_lstm)
    denses.append(dense2)

    for context_input in context_inputs:
        lstm = LSTM(64, batch_input_shape=(10, 4))(context_input)
        dense = Dense(32)(lstm)
        denses.append(dense)

    # Concatenate the outputs of the two branches
    concatenated = concatenate(denses)

    # Output layer
    output = Dense(1)(concatenated)

    # Create the model with two inputs and one output
    model = Model(inputs=[inputs], outputs=output)

    # Compile the model
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model


if __name__ == '__main__':
    context_size, consecutive_frames, features = 8, 10, 4

    data_filename = '../datasets/reformatted/eth_10_10_data.npy'
    data = np.load(data_filename)
    X_train = []
    for i in range(context_size + 2):
        X_train.append(data[:, i])

    labels_filename = '../datasets/reformatted/eth_10_10_labels.npy'
    Y_train = np.load(labels_filename)

    model = build_model(context_size, features, consecutive_frames)

    model.fit(X_train, Y_train,
              epochs=10, batch_size=1024,
              # validation_data=(X_val, Y_val)
              )
