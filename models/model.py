import argparse

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv1D, LSTM, concatenate, Reshape, Dropout, BatchNormalization, MaxPooling1D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import KFold


def conv(filters, reg, name=None):
    return Conv1D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
                  use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)


def build_model(context_size, consecutive_frames, features, units, reg_amount, drop_amount, learning_rate):
    inputs = []

    # pair branch
    # create input layers
    pair_inputs = []
    for i in range(2):
        pair_input = Input(shape=(consecutive_frames, features), name='pair_{}'.format(i))
        pair_inputs.append(pair_input)
        inputs.append(pair_input)

    pair_layers = []
    for pair_input in pair_inputs:
        lstm = LSTM(units, batch_input_shape=(consecutive_frames, features))(pair_input)
        pair_layers.append(lstm)

    reg = l2(reg_amount)

    pair_concatenated = concatenate(pair_layers)
    pair_reshaped = Reshape((pair_concatenated.shape[1], 1))(pair_concatenated)
    pair_conv = Conv1D(filters=64, kernel_size=3, kernel_regularizer=reg, activation=tf.nn.relu)(pair_reshaped)
    drop = Dropout(drop_amount)(pair_conv)
    batch_norm = BatchNormalization()(drop)
    max_pool = MaxPooling1D()(batch_norm)
    drop = Dropout(drop_amount)(max_pool)
    batch_norm = BatchNormalization()(drop)
    pair_layer = batch_norm

    # context branch
    context_inputs = []
    for i in range(context_size):
        context_input = Input(shape=(consecutive_frames, features), name='context_{}'.format(i))
        context_inputs.append(context_input)
        inputs.append(context_input)

    context_layers = []
    for context_input in context_inputs:
        lstm = LSTM(64, batch_input_shape=(consecutive_frames, features))(context_input)
        context_layers.append(lstm)

    context_concatenated = concatenate(context_layers)
    context_reshaped = Reshape((context_concatenated.shape[1], 1))(context_concatenated)
    context_conv = Conv1D(filters=64, kernel_size=3, kernel_regularizer=reg, activation=tf.nn.relu)(context_reshaped)
    drop = Dropout(drop_amount)(context_conv)
    batch_norm = BatchNormalization()(drop)
    max_pool = MaxPooling1D()(batch_norm)
    drop = Dropout(drop_amount)(max_pool)
    batch_norm = BatchNormalization()(drop)
    context_layer = batch_norm

    # Concatenate the outputs of the two branches
    combined = concatenate([pair_layer, context_layer], axis=1)
    combined_dense = Dense(32)(combined)
    # Output layer
    output = Dense(1)(combined_dense)

    # Create the model with two inputs and one output
    model = Model(inputs=[inputs], outputs=output)

    # Compile the model
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default="eth")
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-f', '--features', type=int, default=4)
    parser.add_argument('-cs', '--context_size', type=int, default=8)
    parser.add_argument('-cf', '--consecutive_frames', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('-r', '--reg', type=float, default=0.0000001)
    parser.add_argument('-drop', '--dropout', type=float, default=0.35)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    data_filename = '../datasets/reformatted/{}_{}_{}_data.npy'.format(args.dataset, args.consecutive_frames,
                                                                       args.context_size + 2)
    data = np.load(data_filename)
    X = []
    for i in range(args.context_size + 2):
        X.append(data[:, i])

    labels_filename = '../datasets/reformatted/{}_{}_{}_labels.npy'.format(args.dataset, args.consecutive_frames,
                                                                           args.context_size + 2)
    Y_train = np.load(labels_filename)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf.split(X[0])):
        print("Fold {}:".format(i))
        print("\tTrain: index={}".format(train_index))
        print("\tTest:  index={}".format(test_index))

        model = build_model(args.context_size, args.consecutive_frames, args.features, 64, args.reg, args.dropout,
                            args.learning_rate)

        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        model.fit([x[train_index] for x in X], Y_train[train_index],
                  epochs=args.epochs, batch_size=args.batch_size,
                  validation_data=([x[test_index] for x in X], Y_train[test_index]),
                  callbacks=[early_stop]
                  )
