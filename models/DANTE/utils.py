import argparse
import os
import pickle

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

from F1_calc import F1_calc, F1_calc_clone
from datasets.preparer import read_obsmat
from reformat_data import add_time, import_data


def load_matrix(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


# must have run build_dataset.py first
def load_data(path):
    train = load_matrix(path + '/train.p')
    test = load_matrix(path + '/test.p')
    val = load_matrix(path + '/val.p')
    return test, train, val


# creates a new directory to save the model to
def get_path(dataset, no_pointnet=False):
    path = 'models/' + dataset
    if not os.path.isdir(path):
        os.makedirs(path)

    if no_pointnet:
        path += '/no_pointnet'
        if not os.path.isdir(path):
            os.makedirs(path)

    path = path + '/pair_predictions_'
    i = 1
    while True:
        if not os.path.isdir(path + str(i)):
            path = path + str(i)
            os.makedirs(path)
            print('saving model to ' + path)
            break
        else:
            i += 1

        if i == 10000:
            raise Exception("ERROR: could not find models directory")
    return path


# gives T=1 and T=2/3 F1 scores
def predict(data, model, groups_at_time, dataset="SALSA_all", positions=None):
    if "cocktail_party" in dataset:
        n_people = 6
        n_features = 4

        X, y, frames = data
        preds = model.predict(X)

        return F1_calc(2 / 3, preds, frames, groups_at_time, positions, n_people, n_features), \
            F1_calc(1, preds, frames, groups_at_time, positions, n_people, n_features)
    elif dataset in ["eth", "hotel", "zara01", "zara02", "students03"]:
        pass
    # elif "eth" in dataset:
    #     n_people = 360
    # elif "hotel" in dataset:
    #     n_people = 390
    # elif "zara01" in dataset:
    #     n_people = 148
    # elif "zara02" in dataset:
    #     n_people = 204
    # elif "students03" in dataset:
    #     n_people = 428
    else:
        raise Exception("unknown dataset")

    X, y, frames, groups = data
    preds = model.predict(X)

    return F1_calc_clone(2 / 3, preds, frames, groups_at_time, positions), \
        F1_calc_clone(1, preds, frames, groups_at_time, positions)


class ValLoss(Callback):
    # record train and val losses and mse
    def __init__(self, val_data, dataset, dataset_path):
        super(ValLoss, self).__init__()
        self.val_data = val_data
        self.dataset = dataset
        self.dataset_path = dataset_path

        # each dataset has different params and possibly different F1 calc code
        if dataset in ["cocktail_party"]:
            self.positions, groups = import_data(dataset_path)
            self.groups_at_time = add_time(groups)
        elif dataset in ["eth", "hotel", "zara01", "zara02", "students03"]:
            self.positions = read_obsmat(dataset_path)
            self.groups_at_time = val_data[3]
        else:
            raise Exception("unrecognized dataset")

        self.best_model = None
        self.best_val_mse = float("inf")
        self.best_epoch = -1

        self.val_f1_one_obj = {"f1s": [], "best_f1": float('-inf')}
        self.val_f1_two_thirds_obj = {"f1s": [], "best_f1": float('-inf')}

        self.val_losses = []
        self.train_losses = []

        self.val_mses = []
        self.train_mses = []

    def on_epoch_end(self, epoch, logs={}):
        if logs['val_mse'] < self.best_val_mse:
            self.best_model = self.model
            self.best_val_mse = logs['val_mse']
            self.best_epoch = epoch

        (f1_two_thirds, _, _,), (f1_one, _, _) = predict(self.val_data, self.model, self.groups_at_time,
                                                         dataset=self.dataset, positions=self.positions)

        for f1, obj in [(f1_one, self.val_f1_one_obj), (f1_two_thirds, self.val_f1_two_thirds_obj)]:
            if f1 > obj['best_f1']:
                obj['best_f1'] = f1
                obj['epoch'] = epoch
                obj['model'] = self.model
            obj['f1s'].append(f1)
        self.val_losses.append(logs['val_loss'])
        self.train_losses.append(logs['loss'])
        self.val_mses.append(logs['val_mse'])
        self.train_mses.append(logs['mse'])


# saves the information in the model.history object to a .txt file
def write_history(file_name, history, test):
    file = open(file_name, 'w+')

    file.write("best_val: " + str(history.best_val_mse))
    file.write("\nepoch: " + str(history.best_epoch))

    file.write("\nbest_val_f1_1: " + str(history.val_f1_one_obj['best_f1']))
    file.write("\nepoch: " + str(history.val_f1_one_obj['epoch']))
    (f1_two_thirds, p_2_3, r_2_3), (f1_one, p_1, r_1) = predict(test, history.val_f1_one_obj['model'],
                                                                history.groups_at_time,
                                                                dataset=history.dataset, positions=history.positions)
    file.write("\ntest_f1s: " + str(f1_two_thirds) + " " + str(f1_one))
    file.write('\nprecisions: ' + str(p_2_3) + " " + str(p_1))
    file.write('\nrecalls: ' + str(r_2_3) + " " + str(r_1))

    file.write("\nbest_val_f1_2/3: " + str(history.val_f1_two_thirds_obj['best_f1']))
    file.write("\nepoch: " + str(history.val_f1_two_thirds_obj['epoch']))
    (f1_two_thirds, p_2_3, r_2_3), (f1_one, p_1, r_1) = predict(test, history.val_f1_two_thirds_obj['model'],
                                                                history.groups_at_time,
                                                                dataset=history.dataset, positions=history.positions)
    file.write("\ntest_f1s: " + str(f1_two_thirds) + " " + str(f1_one))
    file.write('\nprecisions: ' + str(p_2_3) + " " + str(p_1))
    file.write('\nrecalls: ' + str(r_2_3) + " " + str(r_1))

    file.write("\ntrain loss:")
    for loss in history.train_losses:
        file.write('\n' + str(loss))
    file.write("\nval loss:")
    for loss in history.val_losses:
        file.write('\n' + str(loss))
    file.write("\ntrain mse:")
    for loss in history.train_mses:
        file.write('\n' + str(loss))
    file.write("\nval mse:")
    for loss in history.val_mses:
        file.write('\n' + str(loss))
    file.write("\nval 1 f1:")
    for f1 in history.val_f1_one_obj['f1s']:
        file.write('\n' + str(f1))
    file.write("\nval 2/3 f1:")
    for f1 in history.val_f1_two_thirds_obj['f1s']:
        file.write('\n' + str(f1))
    file.close()


def conv(filters, reg, name=None):
    return Conv2D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
                  use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)


def build_model(reg_amt, drop_amt, max_people, d, global_filters,
                individual_filters, combined_filters, no_pointnet=False, symmetric=False):
    group_inputs = Input(shape=(1, max_people, d))
    pair_inputs = Input(shape=(1, 2, d))

    reg = l2(reg_amt)

    y = pair_inputs

    # Dyad Transform
    for filters in individual_filters:
        y = conv(filters, reg)(y)
        y = Dropout(drop_amt)(y)
        y = BatchNormalization()(y)

    y_0 = Lambda(lambda input: tf.slice(input, [0, 0, 0, 0], [-1, -1, 1, -1]))(y)
    y_1 = Lambda(lambda input: tf.slice(input, [0, 0, 1, 0], [-1, -1, 1, -1]))(y)

    if no_pointnet:
        concat = Concatenate(name='concat')([Flatten()(y_0), Flatten()(y_1)])
    else:
        x = group_inputs

        # Context Transform
        for filters in global_filters:
            x = conv(filters, reg)(x)
            x = Dropout(drop_amt)(x)
            x = BatchNormalization()(x)

        x = MaxPooling2D(name="global_pool", pool_size=[1, max_people], strides=1, padding='valid')(x)
        x = Dropout(drop_amt)(x)
        x = BatchNormalization()(x)
        x_flat = Flatten()(x)

        # enforce symmetric affinity predictions by doing pointnet on 2 people
        if symmetric:
            y = MaxPooling2D(name="symmetric_pool", pool_size=[1, 2], strides=1, padding='valid')(y)
            concat = Concatenate(name='concat')([x_flat, Flatten()(y)])
        else:
            concat = Concatenate(name='concat')([x_flat, Flatten()(y_0), Flatten()(y_1)])

    # Final MLP from paper
    for filters in combined_filters:
        concat = Dense(units=filters, use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu,
                       kernel_initializer="he_normal")(concat)
        concat = Dropout(drop_amt)(concat)
        concat = BatchNormalization()(concat)

    # final pred
    affinity = Dense(units=1, use_bias="True", kernel_regularizer=reg, activation=tf.nn.sigmoid,
                     name='affinity', kernel_initializer="glorot_normal")(concat)

    model = Model(inputs=[group_inputs, pair_inputs], outputs=affinity)

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model


# constructs a model, trains it with early stopping based on validation loss, and then
# saves the output to a .txt file.
def train_and_save_model(global_filters, individual_filters, combined_filters,
                         train, val, test, epochs, dataset, dataset_path, reg=0.0000001, dropout=.35, fold_num=0,
                         no_pointnet=False, symmetric=False):
    # ensures repeatability
    tf.random.set_seed(0)
    np.random.seed(0)

    num_train, _, max_people, d = train[0][0].shape
    # save achitecture
    path = get_path(dataset, no_pointnet)
    file = open(path + '/architecture.txt', 'w+')
    file.write("global: " + str(global_filters) + "\nindividual: " +
               str(individual_filters) + "\ncombined: " + str(combined_filters) +
               "\nreg= " + str(reg) + "\ndropout= " + str(dropout))

    best_val_mses = []
    best_val_f1s_one = []
    best_val_f1s_two_thirds = []
    X_train, Y_train, timestamps_train = train
    X_val, Y_val, timestamps_val = val

    # build model
    model = build_model(reg, dropout, max_people, d, global_filters, individual_filters, combined_filters,
                        no_pointnet=no_pointnet, symmetric=symmetric)

    # train model
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    history = ValLoss(val, dataset, dataset_path)
    print("MODEL IS IN {}".format(path))
    tensorboard = TensorBoard(log_dir='./logs')

    model.fit(X_train, Y_train, epochs=epochs, batch_size=1024,
              validation_data=(X_val, Y_val), callbacks=[tensorboard, history, early_stop])

    best_val_mses.append(history.best_val_mse)
    best_val_f1s_one.append(history.val_f1_one_obj['best_f1'])
    best_val_f1s_two_thirds.append(history.val_f1_two_thirds_obj['best_f1'])

    # save model
    name = path + '/val_fold_' + str(fold_num)
    if not os.path.isdir(name):
        os.makedirs(name)

    write_history(name + '/results.txt', history, test)

    history.val_f1_one_obj['model'].save(name + '/best_val_model.h5')
    print("saved best val model as " + '/best_val_model.h5')

    file.write("\n\nbest overall val loss: " + str(min(best_val_mses)))
    file.write("\nbest val losses per fold: " + str(best_val_mses))

    file.write("\n\nbest overall f1 1: " + str(max(best_val_f1s_one)))
    file.write("\nbest f1 1s per fold: " + str(best_val_f1s_one))

    file.write("\n\nbest overall f1 2/3: " + str(max(best_val_f1s_two_thirds)))
    file.write("\nbest f1 2/3s per fold: " + str(best_val_f1s_two_thirds))

    file.close()


# constructs a model, trains it with early stopping based on validation loss, and then
# saves the output to a .txt file.
def train_and_save_model_clone(global_filters, individual_filters, combined_filters,
                               train, test, epochs, dataset, dataset_path, reg=0.0000001, dropout=.35, fold_num=0,
                               no_pointnet=False, symmetric=False):
    # ensures repeatability
    tf.random.set_seed(0)
    np.random.seed(0)

    num_train, _, max_people, d = train[0][0].shape
    # save achitecture
    path = get_path(dataset, no_pointnet)
    file = open(path + '/architecture.txt', 'w+')
    file.write("global: " + str(global_filters) + "\nindividual: " +
               str(individual_filters) + "\ncombined: " + str(combined_filters) +
               "\nreg= " + str(reg) + "\ndropout= " + str(dropout))

    best_val_mses = []
    best_val_f1s_one = []
    best_val_f1s_two_thirds = []
    X_train, Y_train, frames_train, groups_train = train
    X_val, Y_val, frames_val, groups_val = test

    # build model
    model = build_model(reg, dropout, max_people, d, global_filters, individual_filters, combined_filters,
                        no_pointnet=no_pointnet, symmetric=symmetric)

    # train model
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    history = ValLoss(test, dataset, dataset_path)
    print("MODEL IS IN {}".format(path))
    tensorboard = TensorBoard(log_dir='./logs')

    model.fit(X_train, Y_train, epochs=epochs, batch_size=1024,
              validation_data=(X_val, Y_val),
              # callbacks=[tensorboard, early_stop]
              callbacks=[tensorboard, history, early_stop]
              )

    best_val_mses.append(history.best_val_mse)
    best_val_f1s_one.append(history.val_f1_one_obj['best_f1'])
    best_val_f1s_two_thirds.append(history.val_f1_two_thirds_obj['best_f1'])

    # save model
    name = path + '/val_fold_' + str(fold_num)
    if not os.path.isdir(name):
        os.makedirs(name)

    write_history(name + '/results.txt', history, test)

    history.val_f1_one_obj['model'].save(name + '/best_val_model.h5')
    print("saved best val model as " + '/best_val_model.h5')

    file.write("\n\nbest overall val loss: " + str(min(best_val_mses)))
    file.write("\nbest val losses per fold: " + str(best_val_mses))

    file.write("\n\nbest overall f1 1: " + str(max(best_val_f1s_one)))
    file.write("\nbest f1 1s per fold: " + str(best_val_f1s_one))

    file.write("\n\nbest overall f1 2/3: " + str(max(best_val_f1s_two_thirds)))
    file.write("\nbest f1 2/3s per fold: " + str(best_val_f1s_two_thirds))

    file.close()


def train_test_split_frames(frames):
    frame_values = np.unique(frames)
    train, test = train_test_split(frame_values)
    train_idx = [i for i, frame in enumerate(frames) if frame in train]
    test_idx = [i for i, frame in enumerate(frames) if frame in test]
    return train_idx, test_idx


def dante_load(path, agents, features):
    '''
    Load dataset and reformat it to match model input.
    :param path: string of path to data
    :param context_size: number of context size
    :param features: number of features
    :return: train and test data
    '''
    X = np.load(path + '_data.npy')
    X = X.reshape((len(X), 1, agents, features))
    y = np.load(path + '_labels.npy')
    frames = np.load(path + '_frames.npy', allow_pickle=True)
    groups = np.load(path + '_groups.npy', allow_pickle=True)
    frame_ids = [frame[0] for frame in frames]
    idx_train, idx_test = train_test_split_frames(frame_ids)
    train = ([X[idx_train, :, 2:], X[idx_train, :, :2]], y[idx_train], frames[idx_train], groups[idx_train])
    test = ([X[idx_test, :, 2:], X[idx_test, :, :2]], y[idx_test], frames[idx_test], groups[idx_test])
    return train, test


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-k', '--fold', type=str, default='0')
    parser.add_argument('-r', '--reg', type=float, default=0.0000001)
    parser.add_argument('-d', '--dropout', type=float, default=0.35)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-f', '--features', type=int, default=4)
    parser.add_argument('-a', '--agents', type=int, default=10)
    # parser.add_argument('--dataset', type=str, default="cocktail_party")
    # parser.add_argument('--dataset_path', type=str, default="../../datasets/cocktail_party")
    parser.add_argument('--dataset', type=str, default="eth")
    parser.add_argument('--dataset_path', type=str, default="../../datasets/ETH/seq_eth")
    parser.add_argument('-p', '--no_pointnet', action="store_true", default=False)
    parser.add_argument('-s', '--symmetric', action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # get data
    if args.dataset == 'cocktail_party':
        test, train, val = load_data("../../datasets/cocktail_party/fold_" + args.fold)
    else:
        train, test = dante_load(
            '../../datasets/reformatted/{}_1_{}'.format(args.dataset, args.agents),
            args.agents, args.features)

    # set model architecture
    global_filters = [64, 128, 512]
    individual_filters = [16, 64, 128]
    combined_filters = [256, 64]

    if args.dataset == 'cocktail_party':
        train_and_save_model(global_filters, individual_filters, combined_filters,
                             train, val, test, args.epochs, args.dataset, args.dataset_path,
                             reg=args.reg, dropout=args.dropout, fold_num=args.fold, no_pointnet=args.no_pointnet,
                             symmetric=args.symmetric)
    else:
        train_and_save_model_clone(global_filters, individual_filters, combined_filters,
                                   train, test, args.epochs, args.dataset, args.dataset_path,
                                   reg=args.reg, dropout=args.dropout, fold_num=args.fold, no_pointnet=args.no_pointnet,
                                   symmetric=args.symmetric)
