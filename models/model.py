import keras as keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, LSTM, concatenate
from keras.models import Model
from sklearn.model_selection import KFold


def conv(filters, reg, name=None):
    return Conv2D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
                  use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)


def build_model(context_size, consecutive_frames, features, units):
    inputs = []
    pair_inputs = []
    context_inputs = []

    pair_first_input = keras.layers.Input(shape=(consecutive_frames, features), name='pair_1')
    pair_inputs.append(pair_first_input)
    inputs.append(pair_first_input)

    pair_second_input = keras.layers.Input(shape=(consecutive_frames, features), name='pair_2')
    pair_inputs.append(pair_second_input)
    inputs.append(pair_second_input)

    for i in range(context_size):
        context_input = keras.layers.Input(shape=(consecutive_frames, features), name='context_{}'.format(i))
        context_inputs.append(context_input)
        inputs.append(context_input)

    denses = []
    for pair_input in pair_inputs:
        lstm = LSTM(units, batch_input_shape=(consecutive_frames, features))(pair_input)
        dense = Dense(32)(lstm)
        denses.append(dense)

    for context_input in context_inputs:
        lstm = LSTM(64, batch_input_shape=(consecutive_frames, features))(context_input)
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
    X = []
    for i in range(context_size + 2):
        X.append(data[:, i])

    labels_filename = '../datasets/reformatted/eth_10_10_labels.npy'
    Y_train = np.load(labels_filename)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf.split(X[0])):
        print("Fold {}:".format(i))
        print("\tTrain: index={}".format(train_index))
        print("\tTest:  index={}".format(test_index))

        model = build_model(context_size, consecutive_frames, features, units=64)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        model.fit([x[train_index] for x in X], Y_train[train_index], epochs=20, batch_size=1024,
                  validation_data=([x[test_index] for x in X], Y_train[test_index]),
                  callbacks=[early_stop]
                  )
