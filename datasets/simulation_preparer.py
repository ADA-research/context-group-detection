import argparse
import random
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from datasets.loader import read_sim, read_multi_groups
from datasets.preparer import dataset_reformat, save_folds, get_scene_data, get_nri_data, get_folds_info, \
    save_nri_folds, report


def groups_size_hist(groups_dict, save_loc):
    """
    Produces a plot of counts of group lengths per dataset
    :param groups_dict: dictionary of group data for each dataset
    :param save_loc: path to location to save the histogram
    :return: nothing
    """
    # create dataframe with group sizes from all datasets
    groups_df_list = []
    for key, group in groups_dict.items():
        groups_len = [len(group) for scene_groups in groups_dict[key] for group in scene_groups]
        temp_df = pd.DataFrame(groups_len, columns=['size'])
        temp_df['dataset'] = key
        groups_df_list.append(temp_df)
    groups_df = pd.concat(groups_df_list)

    # bar plot using seaborn
    sns.set_theme(style="whitegrid")
    sns.catplot(data=groups_df, kind='count', x='size', hue='dataset')
    plt.tight_layout()
    plt.suptitle('Group sizes per dataset')
    plt.ylabel('Count')
    plt.xlabel('Group size')
    plt.savefig(save_loc)
    plt.show()


def dataset_data(dataset_path, sample_frequency):
    """
    Get data for specified dataset.
    :param dataset_path: string of where to find dataset
    :return: dictionary with data
    """
    df = read_sim(dataset_path, sample_frequency)
    groups = read_multi_groups(dataset_path)

    agents_num = df.agent_id.unique().size
    frames = df.frame_id.unique()
    frames_num = frames.size
    frames_difference = frames[1] - frames[0]

    # count_dict = Counter([agent for scene_groups in groups for group in scene_groups for agent in group])
    # agents_in_groups = [agent for agent in count_dict.elements()]
    # single_groups = agents_num - len(agents_in_groups)

    return {
        'df': df,
        'groups': groups,
        'agents': agents_num,
        'frames': frames_num,
        # 'single agent groups': single_groups,
        'difference': frames_difference,
        'duration': df.loc[df.frame_id.idxmax()]['timestamp'] - df.loc[df.frame_id.idxmin()]['timestamp']
    }


def get_group_pairs(groups):
    """
    Get pairs of agents that are in same group.
    :param groups: list of groups
    :return: list of pairs of agents
    """
    scene_pairs = []
    for scene_groups in groups:
        pairs = []
        for group in scene_groups:
            pairs.extend(list(combinations(group, 2)))
        scene_pairs.append(pairs)
    return scene_pairs


def get_sample_rates(scenes, group_pairs, factor=1, target_size=100000):
    """
    Calculate sample rates for same/different group pairs.
    :param scenes: list of scene data
    :param group_pairs: list of group pairs
    :param factor: factor to multiply sample rate to reach desired target size.
    :param target_size: size of dataset to be created
    :return:
    """
    same = []
    different = []
    for scene in scenes:
        scene_id = scene['scene_id']
        pairs = list(combinations(scene['common_agents'], 2))
        for pair in pairs:
            if pair in group_pairs[scene_id]:
                same.append(pair)
            else:
                different.append(pair)

    same_pairs_num = len(same)
    different_pairs_num = len(different)

    desired_proportion = target_size / 2

    same_pairs_rate = int((desired_proportion / same_pairs_num) * factor)
    different_pairs_rate = int((desired_proportion / different_pairs_num) * factor)

    return {
        'same': same_pairs_rate if same_pairs_rate != 0 else 1,
        'different': different_pairs_rate if different_pairs_rate != 0 else 1
    }


def get_sample_params(frames_num, agents_num):
    steps = {
        'sim_1': 1
    }
    multi_frame = True
    if frames_num == 1:
        multi_frame = False
        if agents_num == 6:
            factor = {
                'sim_1': 1
            }
        elif agents_num == 10:
            factor = {
                'sim_1': 1
            }
    elif frames_num == 5:
        if agents_num == 6:
            factor = {
                'sim_1': 1
            }
        elif agents_num == 10:
            factor = {
                'sim_1': 1
            }
    elif frames_num == 10:
        if agents_num == 6:
            factor = {
                'sim_1': 1
            }
        elif agents_num == 10:
            factor = {
                'sim_1': 1
            }
    elif frames_num == 15:
        if agents_num == 6:
            factor = {
                'sim_1': 1
            }
        elif agents_num == 10:
            factor = {
                'sim_1': 1
            }
    return multi_frame, steps, factor


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('--samples_freq', type=int, default=50)
    parser.add_argument('-f', '--frames_num', type=int, default=15)
    parser.add_argument('-a', '--agents_num', type=int, default=6)
    parser.add_argument('-ts', '--target_size', type=int, default=100000)
    parser.add_argument('-d', '--dataset', type=str, default='eth')
    parser.add_argument('-sf', '--save_folder', type=str, default='./reformatted')
    parser.add_argument('-p', '--plot', action="store_true", default=False)
    parser.add_argument('-s', '--shift', action="store_true", default=True)
    parser.add_argument('-r', '--report', action="store_true", default=False)
    parser.add_argument('--nri', action="store_true", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    start = datetime.now()
    print('Started at {}'.format(start))

    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # create datasets report
    datasets_dict = {
        'sim_1': dataset_data('./simulation/sim_10_3_2', args.samples_freq)
    }
    if args.report:
        report('simulation_datasets.csv', datasets_dict)
        exit()

    # create datasets group size histogram
    groups_dict = {
        'sim_1': read_multi_groups('./simulation/sim_10_3_2')
    }
    if args.plot:
        groups_size_hist(groups_dict, './group_size_plot.png')
        exit()

    multi_frame = True
    if args.frames_num == 1:
        multi_frame = False

    multi_frame, steps, factor = get_sample_params(args.frames_num, args.agents_num)

    for dataset in datasets_dict.keys():
        dataset_start = datetime.now()
        print('Dataset: {}, started at: {}'.format(dataset, dataset_start))

        df = datasets_dict[dataset]['df']
        groups = datasets_dict[dataset]['groups']
        difference = datasets_dict[dataset]['difference']

        group_pairs = get_group_pairs(groups)

        scenes = get_scene_data(dataframe=df, consecutive_frames=args.frames_num, difference_between_frames=difference,
                                groups=groups, step=1, sim=True)

        sample_rates = get_sample_rates(scenes, group_pairs, factor=factor[dataset])

        if not args.nri:
            # format dataset to be used by proposed approach
            data, labels, frames, filtered_groups = dataset_reformat(dataframe=df, groups=groups,
                                                                     scene_data=scenes,
                                                                     group_pairs=group_pairs,
                                                                     agents_num=args.agents_num,
                                                                     sample_rates=sample_rates,
                                                                     shift=args.shift)
            dataset = '{}_shifted'.format(dataset) if args.shift else dataset
            # save dataset in folds
            save_folds(args.save_folder, dataset, args.frames_num, args.agents_num, data, labels, frames,
                       filtered_groups, multi_frame, sim=True)
            print('\tdata size: {}'.format(len(data)))
        else:
            nri_data, nri_labels, nri_frames = get_nri_data(dataframe=df, scene_data=scenes)
            dataset = '{}_shifted'.format(dataset) if args.shift else dataset
            folds_info = get_folds_info(args.save_folder, dataset, args.frames_num, args.agents_num)
            save_nri_folds(args.save_folder, dataset, args.frames_num, nri_data, nri_labels, nri_frames, folds_info)
            print('\tdata size: {}'.format(len(nri_data)))

        end = datetime.now()
        print('Dataset: {}, finished in: {}'.format(dataset, end - dataset_start))
        dataset_start = end

    end = datetime.now()
    print('Finished in: {}'.format(end - start))
