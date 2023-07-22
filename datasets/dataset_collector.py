import argparse
import os

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

    frame_pairs = [frame[1] for frame in data[2]]
    unique_agents = np.unique(frame_pairs)

    scene_agents = {}
    for frames, agents in data[2]:
        frames_tuple = tuple(frames)  # Convert the list of frames to a tuple for dictionary key

        if frames_tuple not in scene_agents:
            scene_agents[frames_tuple] = set()  # Initialize an empty set for unique agents

        scene_agents[frames_tuple].update(agents)

    counts = [len(agents) for frames, agents in scene_agents.items()]

    info = {
        'gs': group_samples,
        'ngs': non_group_samples,
        'frames': len(unique_frame_ids),
        'agents': len(unique_agents),
        'agents avg': round(sum(counts) / len(counts), 1)
    }

    return info, counts


def write_info(results, file_path):
    file_name = file_path + '/info.csv'

    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(file_name)


def agent_counts_plot(counts, sets, save_loc):
    num_agents = []
    scene_sets = []
    for set_counts, set_name in zip(counts, sets):
        for count in set_counts:
            num_agents.append(count)
            scene_sets.append(set_name)

    data = pd.DataFrame({'Number of Agents': num_agents, 'Set': scene_sets})

    sns.set(style='whitegrid')

    sns.boxenplot(data=data, x='Number of Agents', order=sets)

    plt.xlabel('# Agents')

    dataset = args.dataset.replace('_shifted', '')
    plt.suptitle('Number of agents in scenes of {} dataset\nusing {}-frame scenes and {} agents'.format(
        dataset, args.frames_num, args.agents_num))
    plt.savefig(save_loc)
    plt.show()


def group_sizes_info(data, key=None):
    groups = []
    for scene_frames, scene_groups in data[3]:
        for group in scene_groups:
            if group not in groups:
                groups.append(group)
    group_sizes = [len(group) for group in groups]

    groups_df = pd.DataFrame(group_sizes, columns=['size'])
    if key is not None:
        groups_df['dataset'] = key

    return groups_df


def groups_size_hist(groups_df, save_loc):
    """
    Produces a plot of counts of group lengths per dataset
    :param groups_df: dataframe of groups of dataset
    :param save_loc: path to location to save the histogram
    :return: nothing
    """
    sns.set(style='whitegrid')
    sns.catplot(data=groups_df, kind='count', x='size')
    dataset = args.dataset.replace('_shifted', '')
    plt.suptitle('Group sizes of {} dataset\nusing {}-frame scenes'.format(dataset, args.frames_num))
    plt.tight_layout()
    plt.ylabel('Count')
    plt.xlabel('Group size')
    plt.savefig(save_loc)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--frames_num', type=int, default=10)
    parser.add_argument('-a', '--agents_num', type=int, default=10)
    parser.add_argument('-d', '--dataset', type=str, default='zara02_shifted')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    dataset_path = './reformatted/{}_{}_{}'.format(args.dataset, args.frames_num, args.agents_num)
    dataset_name = args.dataset.replace('_shifted', '')
    dataset_name = '{}_{}_{}'.format(dataset_name, args.frames_num, args.agents_num)

    for fold in os.listdir(dataset_path):
        fold_path = dataset_path + '/' + fold
        if os.path.isdir(fold_path):
            fold_number = int(fold[-1])
            train, test, val = load_data(fold_path)

            train_info, train_counts = collect_info(train)
            test_info, test_counts = collect_info(test)
            val_info, val_counts = collect_info(val)

            info = {
                'train': train_info,
                'test': test_info,
                'val': val_info
            }

            write_info(info, fold_path)

            train_group_sizes_info = group_sizes_info(train)
            test_group_sizes_info = group_sizes_info(test)
            val_group_sizes_info = group_sizes_info(val)

            groups_df = pd.concat([train_group_sizes_info, test_group_sizes_info, val_group_sizes_info])

            groups_size_hist(groups_df, '{}/group_size_plot_{}.png'.format(dataset_path, dataset_name))

            counts = [train_counts, test_counts, val_counts]
            sets = ['train', 'test', 'val']
            agent_counts_plot(counts, sets, '{}/agent_counts_plot_{}.png'.format(dataset_path, dataset_name))
            break
