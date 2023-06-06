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

from datasets.preparer import read_obsmat
from models.DANTE.F1_calc import F1_calc, F1_calc_clone
from models.DANTE.reformat_data import add_time, import_data


def load_matrix(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


# must have run build_dataset.py first
def load_data(path):
    """
    Loads train, test and val sets
    :param path: string location of the files to be loaded
    :return:
    """
    train = load_matrix(path + '/train.p')
    test = load_matrix(path + '/test.p')
    val = load_matrix(path + '/val.p')
    return test, train, val


def predict(data, model, groups, dataset, multi_frame=False, positions=None, gmitre_calc=False):
    """
    Gives T=1 and T=2/3 F1 scores.
    :param data: data to be used during prediction
    :param model: model to be used for prediction
    :param groups: groups at each scene
    :param dataset: name of dataset
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :param positions: data in raw format
    :param gmitre_calc: True if group mitre should be calculated, otherwise False
    :return: T=1 and T=2/3 F1 scores
    """
    if "cocktail_party" in dataset:
        n_people = 6
        n_features = 4

        X, y, frames = data
        predictions = model.predict(X)

        return F1_calc([2 / 3, 1], predictions, frames, groups, positions, n_people, n_features)
    elif dataset in ["eth", "hotel", "zara01", "zara02", "students03"]:
        pass
    else:
        raise Exception("unknown dataset")

    X, y, frames, groups = data
    predictions = model.predict(X)

    return F1_calc_clone([2 / 3, 1], predictions, frames, groups, positions, multi_frame=multi_frame,
                         gmitre_calc=gmitre_calc)


class ValLoss(Callback):
    """
    Records train and val losses and mse.
    """

    def __init__(self, val_data, dataset, dataset_path, multi_frame=False, gmitre_calc=False):
        super(ValLoss, self).__init__()
        self.val_data = val_data
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.multi_frame = multi_frame
        self.gmitre_calc = gmitre_calc

        # each dataset has different params and possibly different F1 calc code
        if dataset in ["cocktail_party"]:
            self.positions, groups = import_data(dataset_path)
            self.groups = add_time(groups)
        elif dataset in ["eth", "hotel", "zara01", "zara02", "students03"]:
            self.positions = read_obsmat(dataset_path)
            self.groups = val_data[3]
        else:
            raise Exception("unrecognized dataset")

        self.best_model = None
        self.best_val_mse = float("inf")
        self.best_epoch = -1

        self.val_f1_one_obj = {"f1s": [], "best_f1": float('-inf')}
        self.val_f1_two_thirds_obj = {"f1s": [], "best_f1": float('-inf')}
        self.val_f1_gmitre_obj = {"f1s": [], "best_f1": float('-inf')}

        self.val_losses = []
        self.train_losses = []

        self.val_mses = []
        self.train_mses = []

    def on_epoch_end(self, epoch, logs={}):
        if logs['val_mse'] < self.best_val_mse:
            self.best_model = self.model
            self.best_val_mse = logs['val_mse']
            self.best_epoch = epoch

        results = predict(self.val_data, self.model, self.groups, self.dataset, self.multi_frame,
                          self.positions, self.gmitre_calc)

        objs = [self.val_f1_two_thirds_obj, self.val_f1_one_obj, self.val_f1_gmitre_obj]
        for result, obj in zip(results, objs):
            f1 = result[0]
            if f1 > obj['best_f1']:
                obj['best_f1'] = f1
                obj['epoch'] = epoch
                obj['model'] = self.model
            obj['f1s'].append(f1)
        self.val_losses.append(logs['val_loss'])
        self.train_losses.append(logs['loss'])
        self.val_mses.append(logs['val_mse'])
        self.train_mses.append(logs['mse'])


def conv(filters, reg, name=None):
    return Conv2D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
                  use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)


def build_model(reg_amt, drop_amt, max_people, d, global_filters,
                individual_filters, combined_filters, no_pointnet=False, symmetric=False):
    """
    Builds model based on given parameters.
    :param reg_amt: regularization factor
    :param drop_amt: dropout rate
    :param max_people: context size
    :param d: features
    :param global_filters: filters for context branch
    :param individual_filters: filters for pair branch
    :param combined_filters: filters after concatenation
    :param no_pointnet: TODO findout
    :param symmetric: TODO findout
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

    opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model


def write_object_history(file, history_object, history, test, multi_frame=False, gmitre_calc=False):
    file.write("\n\tepoch: " + str(history_object['epoch']))
    results = predict(test, history_object['model'], history.groups, history.dataset, multi_frame, history.positions,
                      gmitre_calc)
    file.write(' '.join(['\n\ttest_f1s:', ' '.join([str(result[0]) for result in results])]))
    file.write(' '.join(['\n\tprecisions:', ' '.join([str(result[1]) for result in results])]))
    file.write(' '.join(['\n\trecalls:', ' '.join([str(result[2]) for result in results])]))


def write_history(file_name, history, test, multi_frame=False, gmitre_calc=False):
    """
    Writes evaluation metrics in file.
    :param file_name:
    :param history: ValLoss to retrieve model and other parameters
    :param test: test dataset to be evaluated on
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :param gmitre_calc: True if group mitre should be calculated, otherwise False
    :return: nothing
    """
    file = open(file_name, 'w+')

    file.write("best_val: " + str(history.best_val_mse))
    file.write("\nepoch: " + str(history.best_epoch))

    file.write("\nbest_val_f1_1: " + str(history.val_f1_one_obj['best_f1']))
    write_object_history(file, history.val_f1_one_obj, history, test, multi_frame, gmitre_calc)

    file.write("\nbest_val_f1_2/3: " + str(history.val_f1_two_thirds_obj['best_f1']))
    write_object_history(file, history.val_f1_two_thirds_obj, history, test, multi_frame, gmitre_calc)

    if gmitre_calc:
        file.write("\nbest_val_f1_gmitre: " + str(history.val_f1_gmitre_obj['best_f1']))
        write_object_history(file, history.val_f1_gmitre_obj, history, test, multi_frame, gmitre_calc)

    file.write("\ntrain loss:")
    for loss in history.train_losses:
        file.write('\n\t' + str(loss))
    file.write("\nval loss:")
    for loss in history.val_losses:
        file.write('\n\t' + str(loss))
    file.write("\ntrain mse:")
    for loss in history.train_mses:
        file.write('\n\t' + str(loss))
    file.write("\nval mse:")
    for loss in history.val_mses:
        file.write('\n\t' + str(loss))
    file.write("\nval 1 f1:")
    for f1 in history.val_f1_one_obj['f1s']:
        file.write('\n\t' + str(f1))
    file.write("\nval 2/3 f1:")
    for f1 in history.val_f1_two_thirds_obj['f1s']:
        file.write('\n\t' + str(f1))
    if gmitre_calc:
        file.write("\nval gmitre f1:")
        for f1 in history.val_f1_gmitre_obj['f1s']:
            file.write('\n\t' + str(f1))
    file.close()


def get_path(dataset, no_pointnet=False):
    """
    # creates a new directory to save the model into.
    :param dataset: name of dataset
    :param no_pointnet: TODO findout
    :return: path
    """
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


def save_model_data(dataset, reg, dropout, history, test, multi_frame=False, no_pointnet=False,
                    gmitre_calc=False):
    """
    Save model and metrics to files.
    :param dataset: name of dataset
    :param reg: regularization factor
    :param dropout: dropout rate
    :param history: ValLoss to retrieve model and other parameters
    :param test: test dataset to be evaluated on
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :param no_pointnet: TODO findout
    :param gmitre_calc: True if group mitre should be calculated, otherwise False
    :return: nothing
    """
    best_val_mses = []
    best_val_f1s_one = []
    best_val_f1s_gmitre = []
    best_val_f1s_two_thirds = []
    best_val_mses.append(history.best_val_mse)
    best_val_f1s_one.append(history.val_f1_one_obj['best_f1'])
    best_val_f1s_gmitre.append(history.val_f1_gmitre_obj['best_f1'])
    best_val_f1s_two_thirds.append(history.val_f1_two_thirds_obj['best_f1'])
    path = get_path(dataset, no_pointnet)
    file = open(path + '/architecture.txt', 'w+')
    file.write("\nreg= " + str(reg) + "\ndropout= " + str(dropout))
    name = path
    if not os.path.isdir(name):
        os.makedirs(name)
    write_history(name + '/results.txt', history, test, multi_frame, gmitre_calc)
    # TODO check which model should be saved
    #  right now the one with the best F1 T=1 is saved
    history.val_f1_one_obj['model'].save(name + '/best_val_model.h5')
    print("saved best val model as " + '/best_val_model.h5')
    file.write("\n\nbest overall val loss: " + str(min(best_val_mses)))
    file.write("\nbest val losses per fold: " + str(best_val_mses))
    file.write("\n\nbest overall f1 1: " + str(max(best_val_f1s_one)))
    file.write("\nbest f1 1s per fold: " + str(best_val_f1s_one))
    file.write("\n\nbest overall f1 2/3: " + str(max(best_val_f1s_two_thirds)))
    file.write("\nbest f1 2/3s per fold: " + str(best_val_f1s_two_thirds))
    if gmitre_calc:
        file.write("\n\nbest overall f1 gmitre: " + str(max(best_val_f1s_gmitre)))
        file.write("\nbest f1 gmitres per fold: " + str(best_val_f1s_gmitre))
    file.close()


def train_and_save_model(global_filters, individual_filters, combined_filters,
                         train, test, val, epochs, dataset, dataset_path, reg=0.0000001, dropout=.35,
                         no_pointnet=False, symmetric=False, batch_size=64, patience=50, gmitre_calc=False):
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
    :param no_pointnet: TODO findout
    :param symmetric: TODO findout
    :param batch_size: batch size used in training of model
    :param patience: number of epochs to be used in EarlyStopping callback
    :param gmitre_calc: True if group mitre should be calculated, otherwise False
    :return: nothing
    """
    # ensures repeatability
    tf.random.set_seed(0)
    np.random.seed(0)

    _, _, max_people, d = train[0][0].shape

    # build model
    model = build_model(reg, dropout, max_people, d, global_filters, individual_filters, combined_filters,
                        no_pointnet=no_pointnet, symmetric=symmetric)

    # train model
    tensorboard = TensorBoard(log_dir='./logs')
    early_stop = EarlyStopping(monitor='val_loss', patience=patience)
    history = ValLoss(val, dataset, dataset_path)

    model.fit(train[0], train[1], epochs=epochs, batch_size=batch_size,
              validation_data=(val[0], val[1]), callbacks=[tensorboard, history, early_stop])

    save_model_data(dataset, reg, dropout, history, test, gmitre_calc=gmitre_calc)


def train_test_split_frames(frames, multi_frame=False):
    """
    Split train, test and val indices.
    :param frames: list of frames
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :return: train, test and val indices
    """
    frame_ids = [frame[0] for frame in frames]
    if multi_frame:
        frame_values = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids)]
        train, test = train_test_split(frame_values, test_size=0.3, random_state=0)
        idx_train = [i for i, frame in enumerate(frame_ids) if frame in train]
        frame_ids_train = [frame[0] for frame in frames[idx_train]]
        frame_values_train = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids_train)]
    else:
        frame_values = np.unique(frame_ids)
        train, test = train_test_split(frame_values, test_size=0.3, random_state=0)
        idx_train = [i for i, frame in enumerate(frame_ids) if frame in train]
        frame_ids_train = [frame[0] for frame in frames[idx_train]]
        frame_values_train = np.unique(frame_ids_train)
    train, val = train_test_split(frame_values_train, test_size=0.2, random_state=0)
    idx_train = [i for i, frame in enumerate(frame_ids) if frame in train]
    idx_test = [i for i, frame in enumerate(frame_ids) if frame in test]
    idx_val = [i for i, frame in enumerate(frame_ids) if frame in val]
    return idx_train, idx_test, idx_val


def train_test_split_groups(groups, frames_train, frames_test, frames_val, multi_frame=False):
    """
    Split groups in train, test and val groups.
    :param groups: list of groups per frame
    :param frames_train: list of train frames
    :param frames_test: list of test frames
    :param frames_val: list of val frames
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :return: groups split in train, test and val sets
    """
    if multi_frame:
        frame_ids_train = [frame[0] for frame in frames_train]
        frame_ids_train = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids_train)]
        frame_ids_test = [frame[0] for frame in frames_test]
        frame_ids_test = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids_test)]
        frame_ids_val = [frame[0] for frame in frames_val]
        frame_ids_val = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids_val)]
    else:

        frame_ids_train = np.unique([frame[0] for frame in frames_train])
        frame_ids_test = np.unique([frame[0] for frame in frames_test])
        frame_ids_val = np.unique([frame[0] for frame in frames_val])
    groups_train = [group for group in groups if group[0] in frame_ids_train]
    groups_test = [group for group in groups if group[0] in frame_ids_test]
    groups_val = [group for group in groups if group[0] in frame_ids_val]
    return groups_train, groups_test, groups_val


# todo read specific fold 
def load_dataset(path, agents, features=None, multi_frame=False):
    """
    Load dataset and reformat it to match model input.
    :param path: string of path to data
    :param agents: number of agents
    :param features: number of features, only needed when multi_frame is frame
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :return: train, test and val data
    """
    X = np.load(path + '_data.npy')
    if not multi_frame:
        X = X.reshape((len(X), 1, agents, features))

    y = np.load(path + '_labels.npy')
    frames = np.load(path + '_frames.npy', allow_pickle=True)
    groups = np.load(path + '_groups.npy', allow_pickle=True)

    idx_train, idx_test, idx_val = train_test_split_frames(frames, multi_frame)
    groups_train, groups_test, groups_val = \
        train_test_split_groups(groups, frames[idx_train], frames[idx_test], frames[idx_val], multi_frame)

    if multi_frame:
        train = ([X[idx_train, :, i] for i in range(agents)], y[idx_train], frames[idx_train], groups_train)
        test = ([X[idx_test, :, i] for i in range(agents)], y[idx_test], frames[idx_test], groups_test)
        val = ([X[idx_val, :, i] for i in range(agents)], y[idx_val], frames[idx_val], groups_val)
    else:
        train = ([X[idx_train, :, 2:], X[idx_train, :, :2]], y[idx_train], frames[idx_train], groups_train)
        test = ([X[idx_test, :, 2:], X[idx_test, :, :2]], y[idx_test], frames[idx_test], groups_test)
        val = ([X[idx_val, :, 2:], X[idx_val, :, :2]], y[idx_val], frames[idx_val], groups_val)
    return train, test, val
