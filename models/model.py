import argparse

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Conv1D, LSTM, concatenate, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam

from models.utils import ValLoss, load_dataset, save_model_data


def build_model(context_size, consecutive_frames, features, units, reg_amount, drop_amount, learning_rate):
    """
    Builds model based on given parameters.
    :param context_size: size of context
    :param consecutive_frames: number of frames per scene
    :param features: features
    :param units: units to be used in filters
    :param reg_amount: regularization factor
    :param drop_amount: dropout rate
    :param learning_rate: learning rate
    :return: model
    """
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
        lstm = LSTM(64, return_sequences=True)(pair_input)
        pair_layers.append(lstm)

    # reg = l2(reg_amount)

    pair_concatenated = concatenate(pair_layers)
    # pair_reshaped = Reshape((pair_concatenated.shape[1], 1))(pair_concatenated)
    pair_conv = Conv1D(filters=32, kernel_size=3, activation='relu', name='pair_conv')(pair_concatenated)
    # drop = Dropout(drop_amount)(pair_conv)
    # batch_norm = BatchNormalization()(drop)
    # max_pool = MaxPooling1D()(batch_norm)
    # drop = Dropout(drop_amount)(max_pool)
    # batch_norm = BatchNormalization()(drop)
    pair_layer = pair_conv

    # context branch
    context_inputs = []
    for i in range(context_size):
        context_input = Input(shape=(consecutive_frames, features), name='context_{}'.format(i))
        context_inputs.append(context_input)
        inputs.append(context_input)

    context_layers = []
    for context_input in context_inputs:
        lstm = LSTM(64, return_sequences=True)(context_input)
        context_layers.append(lstm)

    context_concatenated = concatenate(context_layers)
    # context_reshaped = Reshape((context_concatenated.shape[1], 1))(context_concatenated)
    context_conv = Conv1D(filters=32, kernel_size=3, activation='relu', name='context_conv')(context_concatenated)
    # drop = Dropout(drop_amount)(context_conv)
    # batch_norm = BatchNormalization()(drop)
    # max_pool = MaxPooling1D()(batch_norm)
    # drop = Dropout(drop_amount)(max_pool)
    # batch_norm = BatchNormalization()(drop)
    context_layer = context_conv

    # Concatenate the outputs of the two branches
    combined = concatenate([pair_layer, context_layer], axis=1)
    flatten = Flatten()(combined)
    combined_dense = Dense(64)(flatten)
    # Output layer
    output = Dense(1, activation='sigmoid')(combined_dense)

    # Create the model with two inputs and one output
    model = Model(inputs=[inputs], outputs=output)

    # Compile the model
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="eth")
    parser.add_argument('--dataset_path', type=str, default="../datasets/ETH/seq_eth")
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-f', '--features', type=int, default=4)
    parser.add_argument('-a', '--agents', type=int, default=10)
    parser.add_argument('-cf', '--frames', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('-r', '--reg', type=float, default=0.0000001)
    parser.add_argument('-drop', '--dropout', type=float, default=0.35)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-gm', '--gmitre_calc', action="store_true", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)

    args = get_args()

    train, test, val = load_dataset(
        '../datasets/reformatted/{}_{}_{}'.format(args.dataset, args.frames, args.agents), args.agents,
        multi_frame=True)

    model = build_model(args.agents - 2, args.frames, args.features, 64, args.reg, args.dropout, args.learning_rate)

    tensorboard = TensorBoard(log_dir='./logs')
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = ValLoss(val, args.dataset, args.dataset_path, True, args.gmitre_calc)

    model.fit(train[0], train[1], epochs=args.epochs, batch_size=args.batch_size,
              validation_data=(val[0], val[1]), callbacks=[tensorboard, early_stop, history])

    save_model_data(args.dataset, args.reg, args.dropout, history, test, True, gmitre_calc=args.gmitre_calc)
