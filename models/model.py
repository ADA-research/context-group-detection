import argparse
import json
import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Conv1D, LSTM, concatenate, Input, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from models.utils import ValLoss, load_data, save_model_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def build_model(context_size, consecutive_frames, features, reg_amount, drop_amount, learning_rate, lstm_units=64,
                pair_filters=[32], context_filters=[32], combination_filters=[64], no_context=False):
    """
    Builds model based on given parameters.
    :param context_size: size of context
    :param consecutive_frames: number of frames per scene
    :param features: features
    :param reg_amount: regularization factor
    :param drop_amount: dropout rate
    :param learning_rate: learning rate
    :param lstm_units: units to be used in lstm layers
    :param pair_filters: filters to be used in conv1d layers for pair
    :param context_filters: filters to be used in conv1d layers for context
    :param combination_filters: units to be used in dense layer
    :param no_context: True, if no context is used, otherwise False
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
        lstm = LSTM(lstm_units, return_sequences=True)(pair_input)
        pair_layers.append(lstm)

    reg = l2(reg_amount)

    pair_concatenated = concatenate(pair_layers)

    pair_x = pair_concatenated
    for filters in pair_filters:
        pair_x = Conv1D(filters=filters, kernel_size=3, kernel_regularizer=reg, activation='relu',
                        name='pair_conv_{}'.format(filters))(pair_x)
        pair_x = Dropout(drop_amount)(pair_x)
        pair_x = BatchNormalization()(pair_x)
    pair_conv = pair_x
    # max_pool = MaxPooling1D()(batch_norm)
    # drop = Dropout(drop_amount)(max_pool)
    # batch_norm = BatchNormalization()(drop)
    pair_layer = pair_conv

    if no_context:
        flatten = Flatten()(pair_layer)
    else:
        # context branch
        context_inputs = []
        for i in range(context_size):
            context_input = Input(shape=(consecutive_frames, features), name='context_{}'.format(i))
            context_inputs.append(context_input)
            inputs.append(context_input)

        context_layers = []
        for context_input in context_inputs:
            lstm = LSTM(lstm_units, return_sequences=True)(context_input)
            context_layers.append(lstm)

        context_concatenated = concatenate(context_layers)

        context_x = context_concatenated
        for filters in context_filters:
            context_x = Conv1D(filters=filters, kernel_size=3, kernel_regularizer=reg, activation='relu',
                               name='context_conv_{}'.format(filters))(context_x)
            context_x = Dropout(drop_amount)(context_x)
            context_x = BatchNormalization()(context_x)
        context_conv = context_x
        # max_pool = MaxPooling1D()(batch_norm)
        # drop = Dropout(drop_amount)(max_pool)
        # batch_norm = BatchNormalization()(drop)
        context_layer = context_conv

        # Concatenate the outputs of the two branches
        combined = concatenate([pair_layer, context_layer], axis=1)
        flatten = Flatten()(combined)

    combination_x = flatten
    for filters in combination_filters:
        combination_x = Dense(units=filters, use_bias='True', kernel_regularizer=reg, activation='relu',
                              kernel_initializer="he_normal")(combination_x)
        combination_x = Dropout(drop_amount)(combination_x)
        combination_x = BatchNormalization()(combination_x)

    # Output layer
    output = Dense(1, activation='sigmoid')(combination_x)

    # Create the model with two inputs and one output
    model = Model(inputs=[inputs], outputs=output)

    # Compile the model
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fold', type=str, default="0")
    parser.add_argument('--dataset', type=str, default="eth")
    parser.add_argument('--dir_name', type=str, default="dir_name")
    parser.add_argument('--dataset_path', type=str, default="../datasets/ETH/seq_eth")
    parser.add_argument('-l', '--layers', type=str,
                        default="{\"pair_filters\":[32,128,256],\"context_filters\":[64,128,256],\"combination_filters\":[256,64]}")
    parser.add_argument('--train_epochs', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-a', '--agents', type=int, default=10)
    parser.add_argument('-cf', '--frames', type=int, default=10)
    parser.add_argument('-f', '--features', type=int, default=4)
    parser.add_argument('-p', '--patience', type=int, default=50)
    parser.add_argument('-bs', '--batch_size', type=int, default=1024)
    parser.add_argument('-r', '--reg', type=float, default=0.0000001)
    parser.add_argument('-drop', '--dropout', type=float, default=0.35)
    parser.add_argument('-et', '--eps_thres', type=float, default=1e-13)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-gm', '--gmitre_calc', action="store_true", default=True)
    parser.add_argument('-ds', '--dominant_sets', action="store_true", default=False)
    parser.add_argument('-nc', '--no_context', action="store_true", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)

    args = get_args()

    train, test, val = load_data(
        '../datasets/reformatted/{}_{}_{}/fold_{}'.format(args.dataset, args.frames, args.agents, args.fold),
        args.no_context)

    layers = json.loads(args.layers)

    model = build_model(args.agents - 2, args.frames, args.features, args.reg, args.dropout, args.learning_rate,
                        pair_filters=layers['pair_filters'], context_filters=layers['context_filters'],
                        combination_filters=layers['combination_filters'], no_context=args.no_context)

    tensorboard = TensorBoard(log_dir='./logs')
    early_stop = EarlyStopping(monitor='val_loss', patience=args.patience)
    history = ValLoss(val, args.dataset, args.dataset_path, args.train_epochs, True, args.gmitre_calc, args.eps_thres,
                      args.dominant_sets)

    model.fit(train[0], train[1], epochs=args.epochs, batch_size=args.batch_size,
              validation_data=(val[0], val[1]), callbacks=[tensorboard, early_stop, history])

    dir_name = '{}_{}_{}/fold_{}/{}'.format(args.dataset, args.frames, args.agents, args.fold, args.dir_name)
    save_model_data(dir_name, args.reg, args.dropout, history, test, True, gmitre_calc=args.gmitre_calc,
                    eps_thres=args.eps_thres, dominant_sets=args.dominant_sets, layers=layers,
                    no_context=args.no_context)
