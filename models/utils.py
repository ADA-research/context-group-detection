import os
import pickle
import re

import numpy as np
import tensorflow as tf
import yaml
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from datasets.preparer import read_obsmat
from models.DANTE.F1_calc import F1_calc, F1_calc_clone


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_pickle_file(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def load_data(path, no_context=False):
    """
    Loads train, test and val sets
    :param path: string location of the files to be loaded
    :param no_context: True, if no context is used, otherwise False
    :return: train, test and val sets
    """
    train = load_pickle_file(path + '/train.p')
    test = load_pickle_file(path + '/test.p')
    val = load_pickle_file(path + '/val.p')
    if no_context:
        train = (train[0][:2], train[1], train[2], train[3])
        test = (test[0][:2], test[1], test[2], test[3])
        val = (val[0][:2], val[1], val[2], val[3])
    return train, test, val


def predict(data, model, groups, dataset, multi_frame=False, positions=None, eps_thres=1e-15, dominant_sets=True):
    """
    Gives T=1 and T=2/3 F1 scores.
    :param data: data to be used during prediction
    :param model: model to be used for prediction
    :param groups: groups at each scene
    :param dataset: name of dataset
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :param positions: data in raw format
    :param eps_thres: threshold to be used in vector climb of dominant sets
    :param dominant_sets: True if dominant sets algorithm will be used, otherwise False
    :return: T=1 and T=2/3 F1 scores
    """
    if "cocktail_party" in dataset:
        n_people = 6
        n_features = 4

        X, y, frames = data
        predictions = model.predict(X)

        return F1_calc([2 / 3, 1], predictions, frames, groups, positions, n_people, n_features, eps_thres=eps_thres)
    else:
        dataset_name = dataset
        if "_shifted" in dataset_name:
            dataset_name = dataset_name.replace("_shifted", "")
        if dataset_name not in ["eth", "hotel", "zara01", "zara02", "students03"]:
            raise Exception("unknown dataset")

    X, y, frames, groups = data
    predictions = model.predict(X)

    return F1_calc_clone([2 / 3, 1, None], predictions, frames, groups, positions, multi_frame=multi_frame,
                         eps_thres=eps_thres, dominant_sets=dominant_sets)


# generates feature and ground-truth group matrices from data files
def import_data(dataset_path):
    positions = np.genfromtxt(dataset_path + "/DS_utils/features.txt", dtype='str')
    groups = np.genfromtxt(dataset_path + "/DS_utils/group_names.txt", dtype='str', delimiter=',')
    return positions, groups


# run this to generate Groups_at_time, Groups is from import_gc_data()
# Groups is of the form: time < ID001 ID002 > < ID003 > etc.
# returns dictionary from time to array of group arrays
# eg. time -> [[ID001, ID002], [ID003], ...]
def add_time(Groups):
    Groups_at_time = {}
    for groups in Groups:
        groups_arr = re.split(" < | > < ", groups)
        Groups_at_time[groups_arr[0]] = []
        last_index = -1

        for group in groups_arr[1:]:
            last_index += 1
            Groups_at_time[groups_arr[0]].append(re.split(" ", group))

        # remove last > character
        if len(groups_arr[1:]) == 0:
            continue
        Groups_at_time[groups_arr[0]][last_index] = Groups_at_time[groups_arr[0]][last_index][:-1]

    return Groups_at_time


class ValLoss(Callback):
    """
    Records train and val losses and mse.
    """

    def __init__(self, val_data, dataset, dataset_path, train_epochs=0, multi_frame=False, eps_thres=1e-15,
                 dominant_sets=True):
        super(ValLoss, self).__init__()
        self.val_data = val_data
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.multi_frame = multi_frame
        self.train_epochs = train_epochs
        self.eps_thres = eps_thres
        self.dominant_sets = dominant_sets

        # each dataset has different params and possibly different F1 calc code
        if dataset in ["cocktail_party"]:
            self.positions, groups = import_data(dataset_path)
            self.groups = add_time(groups)
        else:
            dataset_name = dataset
            if "_shifted" in dataset_name:
                dataset_name = dataset_name.replace("_shifted", "")
            if dataset_name in ["eth", "hotel", "zara01", "zara02", "students03"]:
                self.positions = read_obsmat(dataset_path)
                self.groups = val_data[3]
            else:
                raise Exception("unrecognized dataset")

        self.best_model = None
        self.best_val_mse = float("inf")
        self.best_epoch = -1

        self.best_f1_avg_model = None
        self.best_f1_avg = float("-inf")
        self.best_f1_avg_epoch = -1

        self.val_f1_one_obj = {"f1s": [], "best_f1": float('-inf')}
        self.val_f1_two_thirds_obj = {"f1s": [], "best_f1": float('-inf')}
        self.val_f1_gmitre_obj = {"f1s": [], "best_f1": float('-inf')}

        self.val_losses = []
        self.train_losses = []

        self.val_mses = []
        self.train_mses = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if epoch < self.train_epochs:
            return

        if logs['val_mse'] < self.best_val_mse:
            self.best_model = self.model
            self.best_val_mse = logs['val_mse']
            self.best_epoch = epoch

        results = predict(self.val_data, self.model, self.groups, self.dataset, self.multi_frame, self.positions,
                          self.eps_thres, self.dominant_sets)

        avg = 0
        objs = [self.val_f1_two_thirds_obj, self.val_f1_one_obj, self.val_f1_gmitre_obj]
        for result, obj in zip(results, objs):
            f1 = result[0]
            avg += f1
            if f1 > obj['best_f1']:
                obj['best_f1'] = f1
                obj['epoch'] = epoch
                obj['model'] = self.model
            obj['f1s'].append(f1)

        avg = avg / len(objs)
        if avg >= self.best_f1_avg:
            self.best_f1_avg_model = self.model
            self.best_f1_avg = avg
            self.best_f1_avg_epoch = epoch

        self.val_losses.append(logs['val_loss'])
        self.train_losses.append(logs['loss'])
        self.val_mses.append(logs['val_mse'])
        self.train_mses.append(logs['mse'])


def conv(filters, reg, name=None):
    return Conv2D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
                  use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)


def build_model(reg_amt, drop_amt, max_people, d, global_filters, individual_filters, combined_filters):
    """
    Builds model based on given parameters.
    :param reg_amt: regularization factor
    :param drop_amt: dropout rate
    :param max_people: context size
    :param d: features
    :param global_filters: filters for context branch
    :param individual_filters: filters for pair branch
    :param combined_filters: filters after concatenation
    :return: model
    """
    group_inputs = Input(shape=(1, max_people, d))
    pair_inputs = Input(shape=(1, 2, d))

    reg = l2(reg_amt)

    y = pair_inputs

    # Dyad Transform
    for filters in individual_filters:
        y = conv(filters, reg)(y)
        y = Dropout(drop_amt)(y)
        y = BatchNormalization()(y)

    y_0 = Lambda(lambda inp: tf.slice(inp, [0, 0, 0, 0], [-1, -1, 1, -1]))(y)
    y_1 = Lambda(lambda inp: tf.slice(inp, [0, 0, 1, 0], [-1, -1, 1, -1]))(y)

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

    opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model


def get_path(dir_name):
    """
    creates a new directory to save the model into.
    :param dir_name: name of folder to save data
    :return: path
    """

    path = 'results/{}'.format(dir_name)
    if not os.path.isdir(path):
        os.makedirs(path)

    return path


def write_architecture(path, reg, dropout, layers, eps_thres=1e-15, dominant_sets=True, no_context=False):
    """
    Writes evaluation metrics in file.
    :param path: name of the path to the file
    :param reg: regularization factor
    :param dropout: dropout rate
    :param layers: dict with info about layers
    :param eps_thres: threshold to be used in vector climb of dominant sets
    :param dominant_sets: True if dominant sets algorithm will be used, otherwise False
    :param no_context: True if no context data will be used, otherwise False
    :return: nothing
    """
    file = open(path + '/architecture.txt', 'w+')
    file.write("reg= {}\ndropout= {}\n".format(str(reg), str(dropout)))

    file.write("layers\n")
    for key, value in layers.items():
        file.write("\t{}: {}\n".format(key, str(value)))

    file.write("dominant sets: {}\n".format('active' if dominant_sets else 'inactive'))
    if dominant_sets:
        file.write("\teps threshold: {}\n".format(eps_thres))

    file.write("context: {}\n".format('active' if not no_context else 'inactive'))

    file.close()


def write_object_history(file, history_object, history, test, multi_frame=False, eps_thres=1e-15, dominant_sets=True):
    file.write("\tepoch: {}\n".format(str(history_object['epoch'])))
    results = predict(test, history_object['model'], history.groups, history.dataset, multi_frame, history.positions,
                      eps_thres, dominant_sets)
    file.write(' '.join(['\ttest_f1s:', ' '.join([str(result[0]) for result in results])]) + '\n')
    file.write(' '.join(['\tprecisions:', ' '.join([str(result[1]) for result in results])]) + '\n')
    file.write(' '.join(['\trecalls:', ' '.join([str(result[2]) for result in results])]) + '\n')


def write_history(file_name, history, test, multi_frame=False, eps_thres=1e-15, dominant_sets=True):
    """
    Writes evaluation metrics in file.
    :param file_name: name of the file to be written
    :param history: ValLoss to retrieve model and other parameters
    :param test: test dataset to be evaluated on
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :param eps_thres: threshold to be used in vector climb of dominant sets
    :param dominant_sets: True if dominant sets algorithm will be used, otherwise False
    :return: nothing
    """
    file = open(file_name, 'w+')

    file.write("best_val: {}\n".format(str(history.best_val_mse)))
    file.write("\tepoch: {}\n".format(str(history.best_epoch)))

    file.write("best_avg: {}\n".format(str(history.best_f1_avg)))
    file.write("\tepoch: {}\n".format(str(history.best_f1_avg_epoch)))
    results = predict(test, history.best_f1_avg_model, history.groups, history.dataset, multi_frame, history.positions,
                      eps_thres, dominant_sets)
    file.write(' '.join(['\ttest_f1s:', ' '.join([str(result[0]) for result in results])]) + '\n')
    file.write(' '.join(['\tprecisions:', ' '.join([str(result[1]) for result in results])]) + '\n')
    file.write(' '.join(['\trecalls:', ' '.join([str(result[2]) for result in results])]) + '\n')

    file.write("best_val_f1_1: {}\n".format(str(history.val_f1_one_obj['best_f1'])))
    write_object_history(
        file, history.val_f1_one_obj, history, test, multi_frame, eps_thres, dominant_sets)

    file.write("best_val_f1_2/3: {}\n".format(str(history.val_f1_two_thirds_obj['best_f1'])))
    write_object_history(
        file, history.val_f1_two_thirds_obj, history, test, multi_frame, eps_thres, dominant_sets)

    file.write("best_val_f1_gmitre: {}\n".format(str(history.val_f1_gmitre_obj['best_f1'])))
    write_object_history(file, history.val_f1_gmitre_obj, history, test, multi_frame, eps_thres, dominant_sets)

    file.write('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<13s} {:<10s}\n'.format(
        'train loss', 'val loss', 'train mse', 'val mse', 'val 1 f1', 'val 2/3 f1', 'val gmitre f1', 'f1 avg'))
    for i in range(len(history.train_losses)):
        avg = (history.val_f1_one_obj['f1s'][i] + history.val_f1_two_thirds_obj['f1s'][i] +
               history.val_f1_gmitre_obj['f1s'][i]) / 3
        file.write('{:<10.7f} {:<10.7f} {:<10.7f} {:<10.7f} {:<10.7f} {:<10.7f} {:<13.7f} {:<10.7f}\n'.format(
            history.train_losses[i], history.val_losses[i], history.train_mses[i], history.val_mses[i],
            history.val_f1_one_obj['f1s'][i], history.val_f1_two_thirds_obj['f1s'][i],
            history.val_f1_gmitre_obj['f1s'][i], avg))
    file.close()


def save_model_data(dir_name, reg, dropout, history, test, multi_frame=False, eps_thres=1e-15,
                    dominant_sets=True, layers={}, no_context=False):
    """
    Save model and metrics to files.
    :param dir_name: name of folder to save data
    :param reg: regularization factor
    :param dropout: dropout rate
    :param history: ValLoss to retrieve model and other parameters
    :param test: test dataset to be evaluated on
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :param eps_thres: threshold to be used in vector climb of dominant sets
    :param dominant_sets: True if dominant sets algorithm will be used, otherwise False
    :param layers: dict with info about layers
    :param no_context: True if no context data will be used, otherwise False
    :return: nothing
    """
    path = get_path(dir_name)

    write_architecture(path, reg, dropout, layers, eps_thres, dominant_sets, no_context)

    write_history(path + '/results.txt', history, test, multi_frame, eps_thres, dominant_sets)

    history.best_f1_avg_model.save(path + '/best_val_model.h5')
    print("saved best avg model as " + '/best_val_model.h5')


def train_and_save_model(global_filters, individual_filters, combined_filters,
                         train, test, val, epochs, dataset, dataset_path, reg=0.0000001, dropout=.35, batch_size=64,
                         patience=50, dir_name='', eps_thres=1e-15):
    """
    Train and save model based on given parameters.
    :param global_filters: filters for context branch
    :param individual_filters: filters for pair branch
    :param combined_filters: filters after concatenated
    :param train: train set
    :param test: test set
    :param val: val set
    :param epochs: epochs to train model
    :param dataset: name of dataset
    :param dataset_path: path to raw dataset
    :param reg: regularization factor
    :param dropout: dropout rate
    :param batch_size: batch size used in training of model
    :param patience: number of epochs to be used in EarlyStopping callback
    :param dir_name: location to save results
    :param eps_thres: threshold to be used in vector climb of dominant sets
    :return: nothing
    """
    _, _, max_people, d = train[0][0].shape

    # build model
    model = build_model(reg, dropout, max_people, d, global_filters, individual_filters, combined_filters)

    # train model
    tensorboard = TensorBoard(log_dir='./logs')
    early_stop = EarlyStopping(monitor='val_loss', patience=patience)
    history = ValLoss(val, dataset, dataset_path, eps_thres=eps_thres)

    model.fit(train[0], train[1], epochs=epochs, batch_size=batch_size,
              validation_data=(val[0], val[1]), callbacks=[tensorboard, history, early_stop])

    save_model_data(dir_name, reg, dropout, history, test, eps_thres=eps_thres)
