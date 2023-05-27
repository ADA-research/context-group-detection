import argparse

from models.utils import load_data, load_dataset, train_and_save_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-k', '--fold', type=str, default='0')
    parser.add_argument('-r', '--reg', type=float, default=0.0000001)
    parser.add_argument('-d', '--dropout', type=float, default=0.35)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-f', '--features', type=int, default=4)
    parser.add_argument('-a', '--agents', type=int, default=10)
    # parser.add_argument('--dataset', type=str, default="cocktail_party")
    # parser.add_argument('--dataset_path', type=str, default="../datasets/cocktail_party")
    parser.add_argument('--dataset', type=str, default="eth")
    parser.add_argument('--dataset_path', type=str, default="../datasets/ETH/seq_eth")
    parser.add_argument('-p', '--no_pointnet', action="store_true", default=False)
    parser.add_argument('-s', '--symmetric', action="store_true", default=False)
    parser.add_argument('-gm', '--gmitre_calc', action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # set model architecture
    global_filters = [64, 128, 512]
    individual_filters = [16, 64, 128]
    combined_filters = [256, 64]

    # get data
    if args.dataset == 'cocktail_party':
        test, train, val = load_data("../datasets/cocktail_party/fold_" + args.fold)

        train_and_save_model(global_filters, individual_filters, combined_filters,
                             train, test, val, args.epochs, args.dataset, args.dataset_path,
                             reg=args.reg, dropout=args.dropout, no_pointnet=args.no_pointnet,
                             symmetric=args.symmetric, gmitre_calc=args.gmitre_calc)
    else:
        train, test, val, samples = load_dataset(
            '../datasets/reformatted/{}_1_{}'.format(args.dataset, args.agents),
            args.agents, args.features)

        train_and_save_model(global_filters, individual_filters, combined_filters,
                             train, test, val, args.epochs, args.dataset, args.dataset_path, samples,
                             reg=args.reg, dropout=args.dropout, no_pointnet=args.no_pointnet,
                             symmetric=args.symmetric, new=True, gmitre_calc=args.gmitre_calc)
