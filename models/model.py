import argparse

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv1D, LSTM, concatenate, Reshape, Dropout, BatchNormalization, MaxPooling1D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split


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


def train_test_split_frames(frames):
    """
    Split train, test and val indices.
    :param frames: list of frames
    :return: train, test and val indices
    """
    frame_ids = [frame[0] for frame in frames]
    frame_values = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids)]
    train, test = train_test_split(frame_values, test_size=0.3, random_state=0)
    idx_train = [i for i, frame in enumerate(frame_ids) if frame in train]
    frame_ids_train = [frame[0] for frame in frames[idx_train]]
    frame_values_train = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids_train)]
    train, val = train_test_split(frame_values_train, test_size=0.2, random_state=0)
    idx_train = [i for i, frame in enumerate(frame_ids) if frame in train]
    idx_test = [i for i, frame in enumerate(frame_ids) if frame in test]
    idx_val = [i for i, frame in enumerate(frame_ids) if frame in val]
    return idx_train, idx_test, idx_val


def train_test_split_groups(groups, frames_train, frames_test, frames_val):
    """
    Split groups in train, test and val groups.
    :param groups: list of groups per frame
    :param frames_train: list of train frames
    :param frames_test: list of test frames
    :param frames_val: list of val frames
    :return:
    """
    frame_ids_train = [frame[0] for frame in frames_train]
    frame_values_train = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids_train)]
    frame_ids_test = [frame[0] for frame in frames_test]
    frame_values_test = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids_test)]
    frame_ids_val = [frame[0] for frame in frames_val]
    frame_values_val = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids_val)]
    groups_train = [group for group in groups if group[0] in frame_values_train]
    groups_test = [group for group in groups if group[0] in frame_values_test]
    groups_val = [group for group in groups if group[0] in frame_values_val]
    return groups_train, groups_test, groups_val


def load_data(path, agents):
    """
    Load dataset and reformat it to match model input.
    :param path: string of path to data
    :param agents: number of agents
    :return: train, test and val data
    """
    X = np.load(path + '_data.npy')
    y = np.load(path + '_labels.npy')
    frames = np.load(path + '_frames.npy', allow_pickle=True)
    groups = np.load(path + '_groups.npy', allow_pickle=True)

    samples = 0
    for frame in frames:
        if frame[0] == frames[0][0] and frame[1] == frames[0][1]:
            samples += 1

    idx_train, idx_test, idx_val = train_test_split_frames(frames)
    groups_train, groups_test, groups_val = \
        train_test_split_groups(groups, frames[idx_train], frames[idx_test], frames[idx_val])

    train = ([X[idx_train, :, i] for i in range(agents)], y[idx_train], frames[idx_train], groups_train)
    test = ([X[idx_test, :, i] for i in range(agents)], y[idx_test], frames[idx_test], groups_test)
    val = ([X[idx_val, :, i] for i in range(agents)], y[idx_val], frames[idx_val], groups_val)
    return train, test, val, samples


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default="eth")
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-f', '--features', type=int, default=4)
    parser.add_argument('-a', '--agents', type=int, default=10)
    parser.add_argument('-cf', '--frames', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('-r', '--reg', type=float, default=0.0000001)
    parser.add_argument('-drop', '--dropout', type=float, default=0.35)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # data_filename = '../datasets/reformatted/{}_{}_{}_data.npy'.format(args.dataset, args.frames, args.agents)
    # data = np.load(data_filename)
    # X = []
    # for i in range(args.agents + 2):
    #     X.append(data[:, i])
    # labels_filename = '../datasets/reformatted/{}_{}_{}_labels.npy'.format(args.dataset, args.frames, args.agents)
    # Y_train = np.load(labels_filename)
    # frames_filename = '../datasets/reformatted/{}_{}_{}_frames.npy'.format(args.dataset, args.frames, args.agents)
    # frames = np.load(frames_filename)
    # groups_filename = '../datasets/reformatted/{}_{}_{}_groups.npy'.format(args.dataset, args.frames, args.agents)
    # groups = np.load(groups_filename, allow_pickle=True)

    train, test, val, samples = load_data(
        '../datasets/reformatted/{}_{}_{}'.format(args.dataset, args.frames, args.agents), args.agents)

    X_train, y_train, frames_train, groups_train = train
    X_val, y_val, frames_val, groups_val = val

    model = build_model(args.agents - 2, args.frames, args.features, 64, args.reg, args.dropout, args.learning_rate)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(X_train, y_train,
              epochs=args.epochs, batch_size=args.batch_size,
              validation_data=(X_val, y_val),
              callbacks=[early_stop]
              )

    # kf = KFold(n_splits=5, shuffle=True, random_state=0)
    # for i, (train_index, test_index) in enumerate(kf.split(X[0])):
    #     print("Fold {}:".format(i))
    #     print("\tTrain: index={}".format(train_index))
    #     print("\tTest:  index={}".format(test_index))
    #
    #     model = build_model(args.agents - 2, args.frames, args.features, 64, args.reg, args.dropout,
    #                         args.learning_rate)
    #
    #     early_stop = EarlyStopping(monitor='val_loss', patience=5)
    #
    #     model.fit([x[train_index] for x in X], Y_train[train_index],
    #               epochs=args.epochs, batch_size=args.batch_size,
    #               validation_data=([x[test_index] for x in X], Y_train[test_index]),
    #               callbacks=[early_stop]
    #               )
