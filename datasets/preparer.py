import argparse
import math
import os
import pickle
import random
from collections import Counter
from datetime import datetime
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xlsxwriter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from datasets.loader import read_obsmat, read_groups


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


def get_no_context_group_pairs(groups):
    """
    Get pairs of agents that are in same group.
    :param groups: list of groups
    :return: list of pairs of agents
    """
    pairs = []
    for group in groups:
        pairs.extend(list(permutations(group, 2)))
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


def get_scene_groups(agents, groups):
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


def get_scene_data(dataframe, consecutive_frames, difference_between_frames, groups, step, sim=False):
    """
    Get scenes based on given parameters.
    :param dataframe: dataframe to be filtered
    :param consecutive_frames: minimum number of frames for agent not to be removed
    :param difference_between_frames: difference between frames to be continuous
    :param groups: groups to check which groups exist in every scene
    :param step: difference between start of each time window
    :return: scenes after filtering
    """
    # get agents by frame
    agents_by_frame = dataframe.groupby('frame_id')['agent_id'].apply(list).reset_index(name='agents')

    # get frame combinations
    frame_ids = agents_by_frame.frame_id.values
    frame_id_combinations = [list(frame_ids[i:i + consecutive_frames]) for i in
                             range(0, len(frame_ids[:-consecutive_frames]), step)]
    frame_id_combinations = filter_difference_between_frame_combinations(frame_id_combinations,
                                                                         difference_between_frames)

    # check agents intersection in frame id combinations
    scenes = []
    for frames in frame_id_combinations:
        agent_list = [set(agents_by_frame[agents_by_frame['frame_id'] == frame]['agents'].iloc[0]) for frame in frames]
        common_agents = set.intersection(*agent_list)
        if sim:
            scene_id = dataframe[dataframe['frame_id'] == frames[0]]['sim'].iloc[0]
        # ignore scenes with not enough common agents
        if len(common_agents) >= 2:
            scene_dict = {
                'frames': frames,
                'common_agents': common_agents,
                'total_agents': set.union(*agent_list),
                'groups': groups[scene_id] if sim else get_scene_groups(common_agents, groups)
            }
            scenes.append(scene_dict)

    return scenes


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
        if different_pairs_num == 0:
            different_pairs_sampling_rate = 0
        else:
            different_pairs_sampling_rate = min(
                int(min_pair_samples * same_pairs_num / different_pairs_num), max_pair_samples)
    else:
        different_pairs_sampling_rate = min_pair_samples
        if same_pairs_num == 0:
            same_pairs_sampling_rate = 0
        else:
            same_pairs_sampling_rate = min(
                int(min_pair_samples * different_pairs_num / same_pairs_num) if same_pairs_num > 0 else 0,
                max_pair_samples)

    pairs_sample_rates = []
    for pair in pairs:
        if pair in same:
            pairs_sample_rates.append(same_pairs_sampling_rate)
        else:
            pairs_sample_rates.append(different_pairs_sampling_rate)

    return pairs_sample_rates


def dataset_size_calculator(group_pairs, scene_data, agents_minimum, min_pair_samples, max_pair_samples):
    """
    Gather data from all possible scenes based on given parameters.
    :param group_pairs: pairs of agents in the same group
    :param scene_data: valid scenes
    :param agents_minimum: minimum agents (pair + context) in a scene
    :param min_pair_samples: minimum samples to get from a scene for each pair
    :param max_pair_samples: maximum samples to get from a scene for each pair
    :return: dataset
    """
    samples, same_pairs, different_pairs = 0, 0, 0
    for scene in scene_data:
        scene_agents = scene['common_agents']

        pairs = list(combinations(scene_agents, 2))
        pairs_samples = get_pairs_sample_rates(pairs, group_pairs, min_pair_samples, max_pair_samples)
        for pair_agents, pair_samples in zip(pairs, pairs_samples):
            non_pair_agents = scene_agents - set(pair_agents)
            if len(non_pair_agents) <= agents_minimum - 2:
                samples += 1
            else:
                samples += pair_samples
    return samples


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


def fill_data(pair_data, context_data, fake_context):
    for i in range(fake_context):
        context_data.append([tuple([0] * len(pair_data[0][0]))] * len(pair_data[0]))
    return context_data


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def context_sample(pair_data, non_pair_data, context_size):
    """
    Sample the closest agents to the pair as the context.
    :param pair_data: data of agent pair
    :param non_pair_data: data of rest of the agents in the scene
    :param context_size: size of context.
    :return: context data
    """
    frames = len(pair_data[0])
    x1 = np.mean([pair_data[0][i][0] for i in range(frames)])
    y1 = np.mean([pair_data[0][i][1] for i in range(frames)])
    x2 = np.mean([pair_data[1][i][0] for i in range(frames)])
    y2 = np.mean([pair_data[1][i][1] for i in range(frames)])
    pair_locations = ((x1 + x2) / 2, (y1 + y2) / 2)
    non_pair_distances = []
    for j, non_pair in enumerate(non_pair_data):
        x = np.mean([non_pair[i][0] for i in range(frames)])
        y = np.mean([non_pair[i][1] for i in range(frames)])
        distance = calculate_distance(pair_locations[0], pair_locations[1], x, y)
        non_pair_distances.append((j, distance))

    non_pair_distances_sorted = sorted(non_pair_distances, key=lambda x: x[1])
    context_data = [non_pair_data[i] for i in [distance[0] for distance in non_pair_distances_sorted[:context_size]]]

    return context_data


def get_pair_label(groups, agents):
    """
    Checks if agents are in the same group.
    :param groups: list of groups to search
    :param agents: tuple of agents to check
    :return: True if agents are in the same, otherwise False
    """
    return any(all(agent in group for agent in agents) for group in groups)


def gather_data(context_data, data, groups, labels, pair_agents, pair_data, scene_frame_ids, scenes_frames):
    data.append(pair_data + context_data)
    labels.append(get_pair_label(groups, pair_agents))
    scenes_frames.append((scene_frame_ids, pair_agents))


def dataset_reformat(dataframe, groups, group_pairs, scene_data, agents_num, min_pair_samples,
                     max_pair_samples, shift=False):
    """
    Gather data from all possible scenes based on given parameters.
    :param dataframe: dataframe to retrieve data
    :param groups: list of groups
    :param group_pairs: pairs of agents in the same group
    :param scene_data: valid continuous frame combinations
    :param agents_num: minimum agents (pair + context) in a scene
    :param min_pair_samples: minimum samples to get from a scene for each pair
    :param max_pair_samples: maximum samples to get from a scene for each pair
    :param shift: True if to transform context according to pair coordinates, otherwise False
    :return: dataset
    """
    data = []
    labels = []
    scenes_frames = []
    scenes_groups = []
    for scene in scene_data:
        scene_frame_ids = scene['frames']
        scene_groups = scene['groups']
        scene_agents = scene['common_agents']

        pairs = list(combinations(scene_agents, 2))
        pairs_samples = get_pairs_sample_rates(pairs, group_pairs, min_pair_samples, max_pair_samples)
        for pair_agents, pair_samples in zip(pairs, pairs_samples):
            non_pair_agents = scene_agents - set(pair_agents)
            pair_data = get_agent_data_for_frames(dataframe, pair_agents, scene_frame_ids)
            non_pair_data = get_agent_data_for_frames(dataframe, non_pair_agents, scene_frame_ids)
            if shift and len(non_pair_data) > 0:
                non_pair_data = shift_data(pair_data, non_pair_data, len(scene_frame_ids))
            if len(non_pair_data) <= agents_num - 2:
                context_data = non_pair_data[:]
                fake_context = agents_num - 2 - len(non_pair_data)
                context_data = fill_data(pair_data, context_data, fake_context)
                gather_data(context_data, data, groups, labels, pair_agents, pair_data, scene_frame_ids, scenes_frames)
            else:
                for i in range(pair_samples):
                    # random sampling
                    context_data = random.sample(non_pair_data, agents_num - 2)
                    # getting the closest agents
                    # context_data = context_sample(pair_data, non_pair_data, agents_minimum - 2)
                    gather_data(
                        context_data, data, groups, labels, pair_agents, pair_data, scene_frame_ids, scenes_frames)
        scenes_groups.append((scene_frame_ids, scene_groups))
    return np.asarray(data), np.asarray(labels), np.asarray(scenes_frames, dtype=object), np.asarray(scenes_groups,
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
                [data[idx_train, j, :] for j in range(agents_num)], labels[idx_train], frames[idx_train], groups_train)
            test = ([data[idx_test, j, :] for j in range(agents_num)], labels[idx_test], frames[idx_test], groups_test)
            val = ([data[idx_val, j, :] for j in range(agents_num)], labels[idx_val], frames[idx_val], groups_val)
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


def get_labels(agents, pairs):
    agent_permutations = list(permutations(agents, 2))
    labels = [1 if perm in pairs else 0 for perm in agent_permutations]
    return labels


def get_no_context_data(dataframe, scene_data):
    data = []
    labels = []
    scenes_frames = []
    for scene in scene_data:
        scene_frame_ids = scene['frames']
        scene_groups = scene['groups']
        group_pairs = get_no_context_group_pairs(scene_groups)
        scene_agents = scene['common_agents']

        agent_data = get_agent_data_for_frames(dataframe, scene_agents, scene_frame_ids)
        data.append(np.asarray(agent_data)[:, :, :2])
        labels.append(np.asarray(get_labels(scene_agents, group_pairs)))
        scenes_frames.append(scene_frame_ids)

    return np.asarray(data, dtype=object), np.asarray(labels, dtype=object), scenes_frames


def load_pickle_file(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_folds_info(save_folder, dataset, frames_num, agents_num):
    info = []
    path = '{}/{}_{}_{}'.format(save_folder, dataset, frames_num, agents_num)

    for fold in os.listdir(path):
        fold_path = '{}/{}'.format(path, fold)
        train = load_pickle_file('{}/train.p'.format(fold_path))
        test = load_pickle_file('{}/test.p'.format(fold_path))
        val = load_pickle_file('{}/val.p'.format(fold_path))

        info.append((list(train[2][:, 0]), list(test[2][:, 0]), list(val[2][:, 0])))

    return info


def save_no_context_folds(save_folder, dataset, frames_num, data, labels, frames, folds_info):
    for i, (frame_ids_train, frame_ids_test, frame_ids_val) in enumerate(folds_info):
        idx_train = [i for i, frame_id in enumerate(frames) if frame_id in frame_ids_train]
        idx_test = [i for i, frame_id in enumerate(frames) if frame_id in frame_ids_test]
        idx_val = [i for i, frame_id in enumerate(frames) if frame_id in frame_ids_val]
        train = ([torch.tensor(i) for i in data[idx_train]], [torch.tensor(i) for i in labels[idx_train]])
        test = ([torch.tensor(i) for i in data[idx_test]], [torch.tensor(i) for i in labels[idx_test]])
        val = ([torch.tensor(i) for i in data[idx_val]], [torch.tensor(i) for i in labels[idx_val]])

        path = '{}/{}_{}_nri/fold_{}'.format(save_folder, dataset, frames_num, i)
        os.makedirs(path, exist_ok=True)
        dump('{}/tensors_train.pkl'.format(path), train[0])
        dump('{}/labels_train.pkl'.format(path), train[1])
        dump('{}/tensors_test.pkl'.format(path), test[0])
        dump('{}/labels_test.pkl'.format(path), test[1])
        dump('{}/tensors_valid.pkl'.format(path), val[0])
        dump('{}/labels_valid.pkl'.format(path), val[1])


def get_sample_params(frames_num, agents_num):
    if frames_num == 1:
        multi_frame = False
        if agents_num == 6:
            steps = {
                'eth': 2,
                'hotel': 2,
                'zara01': 2,
                'zara02': 3,
                'students03': 6
            }
            min_samples = {
                'eth': 5,
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
                'students03': 10
            }
        elif agents_num == 10:
            steps = {
                'eth': 2,
                'hotel': 2,
                'zara01': 1,
                'zara02': 3,
                'students03': 5
            }
            min_samples = {
                'eth': 5,
                'hotel': 10,
                'zara01': 10,
                'zara02': 5,
                'students03': 2
            }
            max_samples = {
                'eth': 1000,
                'hotel': 1000,
                'zara01': 1000,
                'zara02': 1000,
                'students03': 5
            }
    elif frames_num == 5:
        multi_frame = True
        if agents_num == 6:
            steps = {
                'eth': 3,
                'hotel': 2,
                'zara01': 2,
                'zara02': 5,
                'students03': 6
            }
            min_samples = {
                'eth': 10,
                'hotel': 15,
                'zara01': 15,
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
        elif agents_num == 10:
            steps = {
                'eth': 2,
                'hotel': 1,
                'zara01': 1,
                'zara02': 3,
                'students03': 5
            }
            min_samples = {
                'eth': 10,
                'hotel': 15,
                'zara01': 15,
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
    elif frames_num == 10:
        multi_frame = True
        if agents_num == 6:
            steps = {
                'eth': 2,
                'hotel': 1,
                'zara01': 1,
                'zara02': 4,
                'students03': 5
            }
            min_samples = {
                'eth': 10,
                'hotel': 15,
                'zara01': 10,
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
        elif agents_num == 10:
            steps = {
                'eth': 1,
                'hotel': 1,
                'zara01': 1,
                'zara02': 3,
                'students03': 5
            }
            min_samples = {
                'eth': 10,
                'hotel': 50,
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
    return multi_frame, min_samples, max_samples, steps


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('-f', '--frames_num', type=int, default=10)
    parser.add_argument('-a', '--agents_num', type=int, default=10)
    parser.add_argument('-ts', '--target_size', type=int, default=100000)
    parser.add_argument('-d', '--dataset', type=str, default='eth')
    parser.add_argument('-sf', '--save_folder', type=str, default='./reformatted')
    parser.add_argument('-p', '--plot', action="store_true", default=False)
    parser.add_argument('-s', '--shift', action="store_true", default=True)
    parser.add_argument('-r', '--report', action="store_true", default=False)
    parser.add_argument('-c', '--content', action="store_true", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    start = datetime.now()
    print('Started at {}'.format(start))

    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

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

    multi_frame, min_samples, max_samples, steps = get_sample_params(args.frames_num, args.agents_num)
    for dataset in datasets_dict.keys():
        dataset_start = datetime.now()
        print('Dataset: {}, started at: {}'.format(dataset, dataset_start))

        df = datasets_dict[dataset]['df']
        groups = datasets_dict[dataset]['groups']
        group_pairs = get_group_pairs(groups)
        difference = datasets_dict[dataset]['difference']

        # remove frames with low number of frames or agents
        # df = remove_agents_and_frames_with_insufficient_data(dataframe=df, frames_threshold=args.frames_num,
        #                                                      agents_threshold=args.agents_num)

        # get scene data
        scenes = get_scene_data(dataframe=df, consecutive_frames=args.frames_num, difference_between_frames=difference,
                                groups=groups, step=steps[dataset])

        if args.content:
            # format dataset to be used by proposed approach
            data, labels, frames, filtered_groups = dataset_reformat(dataframe=df, groups=groups,
                                                                     group_pairs=group_pairs,
                                                                     scene_data=scenes, agents_num=args.agents_num,
                                                                     min_pair_samples=min_samples[dataset],
                                                                     max_pair_samples=max_samples[dataset],
                                                                     shift=args.shift)

            dataset = '{}_shifted'.format(dataset) if args.shift else dataset
            # save dataset in folds
            save_folds(args.save_folder, dataset, args.frames_num, args.agents_num, data, labels, frames,
                       filtered_groups, multi_frame)
        else:
            no_context_data, no_context_labels, no_context_frames = get_no_context_data(dataframe=df, scene_data=scenes)
            dataset = '{}_shifted'.format(dataset) if args.shift else dataset
            folds_info = get_folds_info(args.save_folder, dataset, args.frames_num, args.agents_num)
            save_no_context_folds(args.save_folder, dataset, args.frames_num, no_context_data, no_context_labels,
                                  no_context_frames, folds_info)
            # data_tensor = torch.from_numpy(no_context_data)
            # labels_tensor = torch.from_numpy(no_context_data)

        end = datetime.now()
        print('Dataset: {}, finished in: {}'.format(dataset, end - dataset_start))
        # print('\tdata size: {}'.format(len(data)))
        dataset_start = end

    end = datetime.now()
    print('Finished in: {}'.format(end - start))
