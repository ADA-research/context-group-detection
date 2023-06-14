import argparse
import os

from models.utils import load_data, train_and_save_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-k', '--fold', type=str, default='0')
    parser.add_argument('--dir_name', type=str, default="dir_name")
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-a', '--agents', type=int, default=10)
    parser.add_argument('-f', '--features', type=int, default=4)
    parser.add_argument('-p', '--patience', type=int, default=50)
    parser.add_argument('-d', '--dropout', type=float, default=0.35)
    parser.add_argument('-r', '--reg', type=float, default=0.0000001)
    parser.add_argument('-et', '--eps_thres', type=float, default=1e-13)
    # parser.add_argument('--dataset', type=str, default="cocktail_party")
    # parser.add_argument('--dataset_path', type=str, default="../datasets/cocktail_party")
    parser.add_argument('--dataset', type=str, default="eth")
    parser.add_argument('--dataset_path', type=str, default="../datasets/ETH/seq_eth")
    parser.add_argument('-gm', '--gmitre_calc', action="store_true", default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # set model architecture
    global_filters = [64, 128, 512]
    individual_filters = [16, 64, 128]
    combined_filters = [256, 64]

    # get data
    if args.dataset == 'cocktail_party':
        train, test, val = load_data('../datasets/{}/fold_{}'.format(args.dataset, args.fold))

        train_and_save_model(global_filters, individual_filters, combined_filters,
                             train, test, val, args.epochs, args.dataset, args.dataset_path,
                             reg=args.reg, dropout=args.dropout, gmitre_calc=args.gmitre_calc, patience=args.patience,
                             dir_name='{}/fold_{}'.format(args.dataset, args.fold), eps_thres=args.eps_thres)
    else:
        train, test, val = load_data(
            '../datasets/reformatted/{}_1_{}/fold_{}'.format(args.dataset, args.agents, args.fold))

        train_and_save_model(global_filters, individual_filters, combined_filters,
                             train, test, val, args.epochs, args.dataset, args.dataset_path,
                             reg=args.reg, dropout=args.dropout, gmitre_calc=args.gmitre_calc, patience=args.patience,
                             dir_name='{}_1_{}/fold_{}/{}'.format(args.dataset, args.agents, args.fold, args.dir_name),
                             eps_thres=args.eps_thres)
