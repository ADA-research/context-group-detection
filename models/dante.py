import argparse
import os

from models.utils import load_data, train_and_save_model, read_yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default="./config/dante.yml")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = read_yaml(args.config)

    # set model architecture
    global_filters = [64, 128, 512]
    individual_filters = [16, 64, 128]
    combined_filters = [256, 64]

    # get data
    if config['dataset'] == 'cocktail_party':
        train, test, val = load_data('../datasets/{}/fold_{}'.format(config['dataset'], config['fold']))

        train_and_save_model(global_filters, individual_filters, combined_filters, train, test, val, config['epochs'],
                             config['dataset'], config['dataset_path'], reg=config['reg'], dropout=config['dropout'],
                             gmitre_calc=config['gmitre_calc'], patience=config['patience'],
                             dir_name='{}/fold_{}'.format(config['dataset'], config['fold']),
                             eps_thres=config['eps_thres'])
    else:
        train, test, val = load_data(
            '../datasets/reformatted/{}_1_{}/fold_{}'.format(config['dataset'], config['agents'], config['fold']))

        train_and_save_model(global_filters, individual_filters, combined_filters, train, test, val, config['epochs'],
                             config['dataset'], config['dataset_path'], reg=config['reg'], dropout=config['dropout'],
                             gmitre_calc=config['gmitre_calc'], patience=config['patience'],
                             dir_name='{}_1_{}/fold_{}/{}'.format(
                                 config['dataset'], config['agents'], config['fold'], config['dir_name']),
                             eps_thres=config['eps_thres'])
