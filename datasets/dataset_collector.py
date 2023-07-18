import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.utils import load_data


def collect_info(data):
    group_samples = sum(data[1])
    non_group_samples = len(data[1]) - group_samples

    frame_ids = [frame[0] for frame in data[2]]
    if args.frames_num != 1:
        unique_frame_ids = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids)]
    else:
        unique_frame_ids = np.unique(frame_ids)

    info = {
        'gs': group_samples,
        'ngs': non_group_samples,
        'frames': len(unique_frame_ids)
    }

    return info


def write_info(results, file_path):
    file_name = file_path + '/info.csv'

    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(file_name)


def group_sizes_info(data, key):
    groups = []
    for scene_frames, scene_groups in data[3]:
        for group in scene_groups:
            if group not in groups:
                groups.append(group)
    group_sizes = [len(group) for group in groups]
    # counter = Counter(group_sizes)

    groups_df = pd.DataFrame(group_sizes, columns=['size'])
    groups_df['dataset'] = key

    return groups_df


def groups_size_hist(groups_df, save_loc):
    """
    Produces a plot of counts of group lengths per dataset
    :param groups_dict: dictionary of group data for each dataset
    :param save_loc: path to location to save the histogram
    :return: nothing
    """
    # bar plot using seaborn
    sns.set_theme(style="whitegrid")
    sns.catplot(data=groups_df, kind='count', x='size', hue='dataset')
    plt.suptitle('Group sizes per dataset')
    plt.ylabel('Count')
    plt.xlabel('Group size')
    plt.savefig(save_loc)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('-f', '--frames_num', type=int, default=10)
    parser.add_argument('-a', '--agents_num', type=int, default=10)
    parser.add_argument('-ts', '--target_size', type=int, default=100000)
    parser.add_argument('-d', '--dataset', type=str, default='students03_shifted')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    dataset_path = './reformatted/{}_{}_{}/fold_{}'.format(args.dataset, args.frames_num, args.agents_num, args.fold)

    train, test, val = load_data(dataset_path)

    info = {
        'train': collect_info(train),
        'test': collect_info(test),
        'val': collect_info(val)
    }

    write_info(info, dataset_path)

    train_group_sizes_info = group_sizes_info(train, 'train')
    test_group_sizes_info = group_sizes_info(test, 'test')
    val_group_sizes_info = group_sizes_info(val, 'val')

    groups_df = pd.concat([train_group_sizes_info, test_group_sizes_info, val_group_sizes_info])

    groups_size_hist(groups_df, '{}/group_size_plot.png'.format(dataset_path))
