import argparse
import os

import numpy as np
import tensorflow as tf

from models.utils import load_data, train_and_save_model, read_yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-d', '--dir_name', type=str, default="dir_name")
    parser.add_argument('-c', '--config', type=str, default="./config/dante.yml")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    config = read_yaml(args.config)

    # set model architecture
    global_filters = [64, 128, 512]
    individual_filters = [16, 64, 128]
    combined_filters = [256, 64]

    # get data
    if config['dataset'] == 'cocktail_party':
        train, test, val = load_data('../datasets/{}/fold_{}'.format(config['dataset'], args.fold))

        train_and_save_model(global_filters, individual_filters, combined_filters, train, test, val, args.epochs,
                             config['dataset'], config['dataset_path'], reg=config['reg'], dropout=config['dropout'],
                             patience=config['patience'], dir_name='{}/fold_{}'.format(config['dataset'], args.fold),
                             eps_thres=config['eps_thres'])
    else:
        train, test, val = load_data(
            '../datasets/reformatted/{}_1_{}/fold_{}'.format(config['dataset'], config['agents'], args.fold))

        train_and_save_model(global_filters, individual_filters, combined_filters, train, test, val, args.epochs,
                             config['dataset'], config['dataset_path'], reg=config['reg'], dropout=config['dropout'],
                             patience=config['patience'],
                             dir_name='{}_1_{}/fold_{}/{}'.format(
                                 config['dataset'], config['agents'], args.fold, args.dir_name),
                             eps_thres=config['eps_thres'])
