import argparse
import os
import pickle
import random
from collections import Counter
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import xlsxwriter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from datasets.loader import read_obsmat, read_groups

random.seed(14)
np.random.seed(14)


def report(name, data):
    """
    Generate excel report file with dataset data.
    :param name: string for Excel file name
    :param data: dictionary of data for every dataset
    :return: nothing
    """
    workbook = xlsxwriter.Workbook(name)
    worksheet = workbook.add_worksheet('Datasets')
    header_row = 0
    header_column = 0
    worksheet.write(header_row, header_column, 'Dataset')
    worksheet.write(header_row, header_column + 1, 'Agents #')
    worksheet.write(header_row, header_column + 2, 'Frames #')
    worksheet.write(header_row, header_column + 3, 'Groups #')
    worksheet.write(header_row, header_column + 4, 'Duration')
    # worksheet.write(header_row, header_column + 4, 'Agents # in multiple groups')
    # worksheet.write(header_row, header_column + 5, 'Single agent groups #')

    row = header_row + 1
    for key in data.keys():
        worksheet.write(row, 0, key)
        worksheet.write(row, 1, data[key]['agents'])
        worksheet.write(row, 2, data[key]['frames'])
        worksheet.write(row, 3, len(data[key]['groups']))
        worksheet.write(row, 4, data[key]['duration'])
        # worksheet.write(row, 4, data[key]['multigroup agents'])
        # worksheet.write(row, 5, data[key]['single agent groups'])
        row += 1

    workbook.close()


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
        groups_len = [len(group) for group in groups_dict[key]]
        temp_df = pd.DataFrame(groups_len, columns=['size'])
        temp_df['dataset'] = key
        groups_df_list.append(temp_df)
    groups_df = pd.concat(groups_df_list)

    # bar plot using seaborn
    sns.set_theme(style="whitegrid")
    sns.catplot(data=groups_df, kind='count', x='size', hue='dataset')
    plt.suptitle('Group sizes per dataset')
    plt.ylabel('Count')
    plt.xlabel('Group size')
    plt.savefig(save_loc)
    plt.show()


def dataset_data(dataset_path):
    """
    Get data for specified dataset.
    :param dataset_path: string of where to find dataset
    :return: dictionary with data
    """
    df = read_obsmat(dataset_path)
    groups = read_groups(dataset_path)

    agents_num = df.agent_id.unique().size
    frames = df.frame_id.unique()
    frames_num = frames.size
    frames_difference = frames[1] - frames[0]

    count_dict = Counter([agent for group in groups for agent in group])
    agents_in_groups = [agent for agent in count_dict.elements()]
    single_groups = agents_num - len(agents_in_groups)

    return {
        'df': df,
        'groups': groups,
        'agents': agents_num,
        'frames': frames_num,
        'single agent groups': single_groups,
        'difference': frames_difference,
        'duration': df.loc[df.frame_id.idxmax()]['timestamp'] - df.loc[df.frame_id.idxmin()]['timestamp']
    }


def get_group_pairs(groups):
    """
    Get pairs of agents that are in same group.
    :param groups: list of groups
    :return: list of pairs of agents
    """
    pairs = []
    for group in groups:
        pairs.extend(list(combinations(group, 2)))
    return pairs


def remove_agents_in_low_number_of_frames(dataframe, agent_ids_to_be_removed):
    """
    Filters dataframe to find agents with frames less than the given threshold.
    :param dataframe: dataframe to be filtered
    :param agent_ids_to_be_removed: agent ids to be removed
    :return: filtered dataframe
    """
    return dataframe[not dataframe.agent_id.isin(agent_ids_to_be_removed)]


def check_for_agents_in_low_number_of_frames(dataframe, frames_threshold):
    """
    Check if there are agents that need to be removed from the dataframe, given a frame threshold.
    :param dataframe: dataframe to be filtered
    :param frames_threshold: minimum number of frames for agent not to be removed
    :return: list of agent ids to be removed
    """
    agents_df = dataframe.groupby('agent_id')['frame_id'].apply(list).reset_index(name='frames')
    agents_df['frames_num'] = agents_df['frames'].apply(len)
    return list(agents_df[agents_df['frames_num'] < frames_threshold]['agent_id'].values)


def remove_frames_with_low_number_of_agents(dataframe, frame_ids_to_be_removed):
    """
    Filters dataframe to find frames with agents less than the given threshold.
    :param dataframe: dataframe to be filtered
    :param frame_ids_to_be_removed: frames to be removed
    :return: filtered dataframe
    """
    return dataframe[not dataframe.frame_id.isin(frame_ids_to_be_removed)]


def check_for_frames_with_low_number_of_agents(dataframe, agents_threshold):
    """
    Check if there are frames that need to be removed from the dataframe, given an agent threshold.
    :param dataframe: dataframe to be filtered
    :param agents_threshold: minimum number of agents for frame not to be removed
    :return: list of frame ids to be removed
    """
    frames_df = dataframe.groupby('frame_id')['agent_id'].apply(list).reset_index(name='agents')
    frames_df['agents_num'] = frames_df['agents'].apply(len)
    return list(frames_df[frames_df['agents_num'] < agents_threshold]['frame_id'].values)


def remove_agents_and_frames_with_insufficient_data(dataframe, agents_threshold, frames_threshold):
    """
    Remove agents and frames with insufficient data, based on given thresholds.
    :param dataframe: dataframe to be filtered
    :param agents_threshold: minimum number of agents for frame not to be removed
    :param frames_threshold: minimum number of frames for agent not to be removed
    :return: filtered dataframe
    """
    unwanted_frame_ids = check_for_frames_with_low_number_of_agents(dataframe, agents_threshold)
    unwanted_agent_ids = check_for_agents_in_low_number_of_frames(dataframe, frames_threshold)

    while len(unwanted_frame_ids) > 0 or len(unwanted_agent_ids) > 0:
        dataframe = dataframe[dataframe.agent_id.isin(unwanted_agent_ids) == False]
        dataframe = dataframe[dataframe.frame_id.isin(unwanted_frame_ids) == False]
        unwanted_frame_ids = check_for_frames_with_low_number_of_agents(dataframe, agents_threshold)
        unwanted_agent_ids = check_for_agents_in_low_number_of_frames(dataframe, frames_threshold)

    return dataframe


def filter_difference_between_frame_combinations(combinations, diff_between_frames):
    """
    Filter frame combinations based on given difference between frames to be considered continuous.
    :param combinations: list of frame combinations to be filtered
    :param diff_between_frames: difference between frames to be continuous
    :return: list of filtered frame combinations
    """
    filtered_combinations = []
    for frames in combinations:
        differences = [True for i, frame in enumerate(frames[:-1]) if frames[i + 1] - frame != diff_between_frames]
        if len(differences) == 0:
            filtered_combinations.append(frames)
    return filtered_combinations


def get_frame_combs_data(dataframe, agents_minimum, consecutive_frames, difference_between_frames, groups, step):
    """
    Get frame combinations based on given parameters.
    :param dataframe: dataframe to be filtered
    :param agents_minimum: minimum number of agents for frame not to be removed
    :param consecutive_frames: minimum number of frames for agent not to be removed
    :param difference_between_frames: difference between frames to be continuous
    :param groups: groups to check which groups exist in every combination
    :param step: difference between start of each time window
    :return: frame combinations after filtering
    """
    # get agents by frame
    agents_by_frame = dataframe.groupby('frame_id')['agent_id'].apply(list).reset_index(name='agents')

    # get frame combinations
    frame_ids = agents_by_frame.frame_id.values
    frame_id_combinations = [list(frame_ids[i:i + consecutive_frames]) for i in
                             range(0, len(frame_ids[:-consecutive_frames]), step)]
    frame_id_combinations = filter_difference_between_frame_combinations(frame_id_combinations,
                                                                         difference_between_frames)

    # check agents intersection in frame combinations
    combs = []
    for frames in frame_id_combinations:
        agent_list = [set(agents_by_frame[agents_by_frame['frame_id'] == frame]['agents'].iloc[0]) for frame in frames]
        common_agents = set.intersection(*agent_list)
        # ignore frame combinations with not enough common agents
        if len(common_agents) >= agents_minimum:
            comb_dict = {
                'frames': frames,
                'common_agents': common_agents,
                'total_agents': set.union(*agent_list),
                'groups': get_frame_comb_groups(common_agents, groups)
            }
            combs.append(comb_dict)

    return combs


def get_frame_comb_groups(agents, groups):
    """
    Filter groups with agents that exist in frame combination.
    :param agents: agents in frame combination
    :param groups: groups to be filtered
    :return: list of groups
    """
    comb_groups = []
    for agent in agents:
        for group in groups:
            if agent in group and group not in comb_groups:
                comb_groups.append(group)
    comb_groups_filtered = [[agent for agent in comb_group if agent in agents] for comb_group in comb_groups]
    return comb_groups_filtered


def get_agent_data_for_frames(dataframe, agents, frames):
    """
    Returns a list of tuples with location and velocity data for each frame and agent
    :param dataframe: dataframe to retrieve data
    :param agents: list of agents for who to retrieve data
    :param frames: list of frames for which to retrieve data
    :return: list of lists of data of each agent
    """
    data = dataframe[dataframe['frame_id'].isin(frames) & dataframe['agent_id'].isin(agents)]
    return list(data.groupby('agent_id')['measurement'].apply(list).values)


def shift_data(pair_data, context_data, frames):
    new_context_data = [[] for i in range(len(context_data))]
    for i in range(frames):
        x1 = pair_data[0][i][0]
        y1 = pair_data[0][i][1]
        x2 = pair_data[1][i][0]
        y2 = pair_data[1][i][1]
        a = .5 * (x1 + x2)
        b = .5 * (y1 + y2)
        dx = x1 - x2
        dy = y1 - y2
        b0 = dx / np.sqrt(dx ** 2 + dy ** 2)
        b1 = dy / np.sqrt(dx ** 2 + dy ** 2)

        for j, agent in enumerate(context_data):
            x = agent[i][0]
            y = agent[i][1]
            # shift, project each x and y
            x_shift = x - a
            y_shift = y - b
            x_proj = b0 * x_shift + b1 * y_shift
            y_proj = b1 * x_shift - b0 * y_shift
            new_context_data[j].append((x_proj, y_proj, agent[i][2], agent[i][3]))

    return new_context_data


def get_pair_label(groups, agents):
    """
    Checks if agents are in the same group.
    :param groups: list of groups to search
    :param agents: tuple of agents to check
    :return: True if agents are in the same, otherwise False
    """
    return any(all(agent in group for agent in agents) for group in groups)


def scene_sample(dataframe, groups, pair_agents, context_agents, frames, data, labels, shift=False):
    """
    Sampling scene by getting agents and label data.
    :param dataframe: dataframe to retrieve data
    :param groups: list of groups to search
    :param pair_agents: list of pair agents in the scene
    :param context_agents: list of context agents in the scene
    :param frames: list of frames for which to get data
    :param data: list to store agent data
    :param labels: list to store group relationship
    :return: nothing
    """
    pair_data = get_agent_data_for_frames(dataframe, pair_agents, frames)
    context_data = get_agent_data_for_frames(dataframe, context_agents, frames)
    if shift:
        context_data = shift_data(pair_data, context_data, len(frames))
    pair_data.extend(context_data)
    data.append(pair_data)
    label = get_pair_label(groups, pair_agents)
    labels.append(label)


def get_pairs_sample_rates(pairs, group_pairs, min_pair_samples, max_pair_samples):
    """
    Set sample rate for pairs in same and different groups in order to have balanced samples.
    :param pairs: list of pairs in scene
    :param group_pairs: list of pairs in same group
    :param min_pair_samples: minimum number of samples to get from a pair in a scene
    :param max_pair_samples: maximum number of samples to get from a pair in a scene
    :return: list of pairs sample rates
    """
    same = []
    different = []
    for pair in pairs:
        if pair in group_pairs:
            same.append(pair)
        else:
            different.append(pair)

    same_pairs_num = len(same)
    different_pairs_num = len(different)

    if same_pairs_num > different_pairs_num:
        same_pairs_sampling_rate = min_pair_samples
        different_pairs_sampling_rate = min(
            int(min_pair_samples * same_pairs_num / different_pairs_num), max_pair_samples)
    else:
        different_pairs_sampling_rate = min_pair_samples
        same_pairs_sampling_rate = min(
            int(min_pair_samples * different_pairs_num / same_pairs_num) if same_pairs_num > 0 else 0, max_pair_samples)

    pairs_sample_rates = []
    for pair in pairs:
        if pair in same:
            pairs_sample_rates.append(same_pairs_sampling_rate)
        else:
            pairs_sample_rates.append(different_pairs_sampling_rate)

    return pairs_sample_rates


def dataset_size_calculator(group_pairs, frame_comb_data, min_pair_samples, max_pair_samples):
    """
    Gather data from all possible scenes based on given parameters.
    :param group_pairs: pairs of agents in the same group
    :param frame_comb_data: valid continuous frame combinations
    :param min_pair_samples: minimum samples to get from a scene for each pair
    :param max_pair_samples: maximum samples to get from a scene for each pair
    :return: dataset
    """
    samples = 0
    for frame_comb in frame_comb_data:
        comb_agents = frame_comb['common_agents']

        pairs = list(combinations(comb_agents, 2))
        pairs_samples = get_pairs_sample_rates(pairs, group_pairs, min_pair_samples, max_pair_samples)
        samples += sum(pairs_samples)
    return samples


def dataset_reformat(dataframe, groups, group_pairs, frame_comb_data, agents_minimum, min_pair_samples,
                     max_pair_samples, shift=False):
    """
    Gather data from all possible scenes based on given parameters.
    :param dataframe: dataframe to retrieve data
    :param groups: list of groups
    :param group_pairs: pairs of agents in the same group
    :param frame_comb_data: valid continuous frame combinations
    :param agents_minimum: minimum agents (pair + context) in a scene
    :param min_pair_samples: minimum samples to get from a scene for each pair
    :param max_pair_samples: maximum samples to get from a scene for each pair
    :return: dataset
    """
    data = []
    labels = []
    frames = []
    combs_groups = []
    for frame_comb in frame_comb_data[:10]:
        comb_frames = frame_comb['frames']
        comb_agents = frame_comb['common_agents']
        comb_groups = frame_comb['groups']

        pairs = list(combinations(comb_agents, 2))
        pairs_samples = get_pairs_sample_rates(pairs, group_pairs, min_pair_samples, max_pair_samples)
        for pair_agents, pair_samples in zip(pairs, pairs_samples):
            scene_agents = comb_agents - set(pair_agents)
            for i in range(pair_samples):
                context_agents = random.sample(scene_agents, agents_minimum - 2)
                scene_sample(dataframe, groups, pair_agents, context_agents, comb_frames, data, labels, shift)
                frames.append((comb_frames, pair_agents))
        combs_groups.append((comb_frames, comb_groups))
    return np.asarray(data), np.asarray(labels), np.asarray(frames, dtype=object), np.asarray(combs_groups,
                                                                                              dtype=object)


def folds_split(frames, folds_num, multi_frame=False):
    """
    Split frames based on frame id and split in folds.
    :param frames: list of frames
    :param folds_num: number of folds to split frames
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :return:
    """
    frame_ids = [frame[0] for frame in frames]
    if multi_frame:
        frame_values = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids)]
    else:
        frame_values = np.unique(frame_ids)

    fold_size = len(frame_values) // folds_num

    folds_idx = []
    for i in range(folds_num):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size

        if i == folds_num - 1:
            test_fold_frame_values = frame_values[start_idx:]
        else:
            test_fold_frame_values = frame_values[start_idx:end_idx]
        test_fold_idx = [i for i, frame in enumerate(frame_ids) if frame in test_fold_frame_values]
        train_fold_idx = [i for i, frame in enumerate(frame_ids) if frame not in test_fold_frame_values]
        folds_idx.append((train_fold_idx, test_fold_idx))

    return folds_idx


def train_val_split_frames(frames, idx, multi_frame=False):
    """
    Split train, test and val indices.
    :param frames: list of frames
    :param idx: list of indices corresponding to the train and val set
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :return: train and val indices
    """
    frame_ids = [frame[0] for frame in frames]
    train_val_frame_ids = [frame[0] for frame in frames[idx]]
    if multi_frame:
        frame_values = [list(x) for x in set(tuple(frame_id) for frame_id in train_val_frame_ids)]
    else:
        frame_values = np.unique(train_val_frame_ids)
    train, val = train_test_split(frame_values, test_size=0.3, random_state=0)
    idx_train = [i for i, frame in enumerate(frame_ids) if frame in train]
    idx_val = [i for i, frame in enumerate(frame_ids) if frame in val]
    return idx_train, idx_val


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


def dump(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def save_folds(save_folder, dataset, frames_num, agents_num, data, labels, frames, groups, multi_frame, folds_num=5,
               features=4):
    if not multi_frame:
        data = data.reshape((len(data), 1, agents_num, features))

    for i, (idx_train_val, idx_test) in enumerate(folds_split(frames, folds_num, multi_frame)):
        idx_train, idx_val = train_val_split_frames(frames, idx_train_val, multi_frame)
        groups_train, groups_test, groups_val = \
            train_test_split_groups(groups, frames[idx_train], frames[idx_test], frames[idx_val], multi_frame)

        if multi_frame:
            train = (
                [data[idx_train, :, j] for j in range(agents_num)], labels[idx_train], frames[idx_train], groups_train)
            test = ([data[idx_test, :, j] for j in range(agents_num)], labels[idx_test], frames[idx_test], groups_test)
            val = ([data[idx_val, :, j] for j in range(agents_num)], labels[idx_val], frames[idx_val], groups_val)
        else:
            train = (
                [data[idx_train, :, 2:], data[idx_train, :, :2]], labels[idx_train], frames[idx_train], groups_train)
            test = ([data[idx_test, :, 2:], data[idx_test, :, :2]], labels[idx_test], frames[idx_test], groups_test)
            val = ([data[idx_val, :, 2:], data[idx_val, :, :2]], labels[idx_val], frames[idx_val], groups_val)

        path = '{}/{}_{}_{}/fold_{}'.format(save_folder, dataset, frames_num, agents_num, i)
        os.makedirs(path, exist_ok=True)
        dump('{}/train.p'.format(path), train)
        dump('{}/test.p'.format(path), test)
        dump('{}/val.p'.format(path), val)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--report', action="store_true", default=False)
    parser.add_argument('-s', '--shift', action="store_true", default=True)
    parser.add_argument('-p', '--plot', action="store_true", default=False)
    parser.add_argument('-f', '--frames_num', type=int, default=10)
    parser.add_argument('-a', '--agents_num', type=int, default=10)
    parser.add_argument('-ts', '--target_size', type=int, default=100000)
    parser.add_argument('-d', '--dataset', type=str, default='eth')
    parser.add_argument('-sf', '--save_folder', type=str, default='./reformatted')

    return parser.parse_args()


if __name__ == '__main__':
    start = datetime.now()
    print('Started at {}'.format(start))

    args = get_args()

    # create datasets report
    datasets_dict = {
        'eth': dataset_data('./ETH/seq_eth'),
        'hotel': dataset_data('./ETH/seq_hotel'),
        'zara01': dataset_data('./UCY/zara01'),
        'zara02': dataset_data('./UCY/zara02'),
        'students03': dataset_data('./UCY/students03')
    }
    if args.report:
        report('datasets.xlsx', datasets_dict)

    # create datasets group size histogram
    groups_dict = {
        'eth': read_groups('./ETH/seq_eth'),
        'hotel': read_groups('./ETH/seq_hotel'),
        'zara01': read_groups('./UCY/zara01'),
        'zara02': read_groups('./UCY/zara02'),
        'students03': read_groups('./UCY/students03')
    }
    if args.plot:
        groups_size_hist(groups_dict, './group_size_plot.png')

    if args.frames_num == 1:
        multi_frame = False
        steps = {
            'eth': 2,
            'hotel': 1,
            'zara01': 1,
            'zara02': 3,
            'students03': 5
        }
        min_samples = {
            'eth': 8,
            'hotel': 10,
            'zara01': 10,
            'zara02': 5,
            'students03': 2
        }
        max_samples = {
            'eth': 100,
            'hotel': 100,
            'zara01': 100,
            'zara02': 100,
            'students03': 5
        }
    else:
        multi_frame = True
        steps = {
            'eth': 1,
            'hotel': 1,
            'zara01': 1,
            'zara02': 3,
            'students03': 5
        }
        min_samples = {
            'eth': 10,
            'hotel': 40,
            'zara01': 20,
            'zara02': 10,
            'students03': 2
        }
        max_samples = {
            'eth': 1000,
            'hotel': 1000,
            'zara01': 1000,
            'zara02': 1000,
            'students03': 10
        }
    for dataset in datasets_dict.keys():
        dataset_start = datetime.now()
        print('Dataset: {}, started at: {}'.format(dataset, dataset_start))

        df = datasets_dict[dataset]['df']
        groups = datasets_dict[dataset]['groups']
        group_pairs = get_group_pairs(groups)
        difference = datasets_dict[dataset]['difference']

        # remove agents with low number of frames or agents
        df = remove_agents_and_frames_with_insufficient_data(dataframe=df, frames_threshold=args.frames_num,
                                                             agents_threshold=args.agents_num)

        # get frame combinations data
        combs = get_frame_combs_data(dataframe=df, agents_minimum=args.agents_num,
                                     consecutive_frames=args.frames_num, difference_between_frames=difference,
                                     groups=groups, step=steps[dataset])

        # format dataset to be used by proposed approach
        data, labels, frames, filtered_groups = dataset_reformat(dataframe=df, groups=groups, group_pairs=group_pairs,
                                                                 frame_comb_data=combs, agents_minimum=args.agents_num,
                                                                 min_pair_samples=min_samples[dataset],
                                                                 max_pair_samples=max_samples[dataset],
                                                                 shift=args.shift)

        dataset = '{}_shifted'.format(dataset) if args.shift else dataset
        # save dataset in folds
        save_folds(args.save_folder, dataset, args.frames_num, args.agents_num, data, labels, frames, filtered_groups,
                   multi_frame)

        end = datetime.now()
        print('Dataset: {}, finished in: {}'.format(dataset, end - dataset_start))
        print('\tdata size: {}'.format(len(data)))
        dataset_start = end

    end = datetime.now()
    print('Finished in: {}'.format(end - start))
