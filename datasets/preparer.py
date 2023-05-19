import argparse
import random
from collections import Counter
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import xlsxwriter
from matplotlib import pyplot as plt

from loader import read_obsmat, read_groups

random.seed(14)


def report(name, data):
    '''
    Generate excel report file with dataset data.
    :param name: string for Excel file name
    :param data: dictionary of data for every dataset
    :return: nothing
    '''
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
    '''
    Produces a plot of counts of group lengths per dataset
    :param groups_dict:
    :return: nothing
    '''
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
    '''
    Get data for specified dataset.
    :param dataset_path: string of where to find dataset
    :return: dictionary with data
    '''
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


def remove_agents_in_low_number_of_frames(dataframe, agent_ids_to_be_removed):
    '''
    Filters dataframe to find agents with frames less than the given threshold.
    :param dataframe: dataframe to be filtered
    :param agent_ids_to_be_removed: agent ids to be removed
    :return: filtered dataframe
    '''
    return dataframe[not dataframe.agent_id.isin(agent_ids_to_be_removed)]


def check_for_agents_in_low_number_of_frames(dataframe, frames_threshold):
    '''
    Check if there are agents that need to be removed from the dataframe, given a frame threshold.
    :param dataframe: dataframe to be filtered
    :param frames_threshold: minimum number of frames for agent not to be removed
    :return: list of agent ids to be removed
    '''
    agents_df = dataframe.groupby('agent_id')['frame_id'].apply(list).reset_index(name='frames')
    agents_df['frames_num'] = agents_df['frames'].apply(len)
    return list(agents_df[agents_df['frames_num'] < frames_threshold]['agent_id'].values)


def remove_frames_with_low_number_of_agents(dataframe, frame_ids_to_be_removed):
    '''
    Filters dataframe to find frames with agents less than the given threshold.
    :param dataframe: dataframe to be filtered
    :param frame_ids_to_be_removed: frames to be removed
    :return: filtered dataframe
    '''
    return dataframe[not dataframe.frame_id.isin(frame_ids_to_be_removed)]


def check_for_frames_with_low_number_of_agents(dataframe, agents_threshold):
    '''
    Check if there are frames that need to be removed from the dataframe, given an agent threshold.
    :param dataframe: dataframe to be filtered
    :param agents_threshold: minimum number of agents for frame not to be removed
    :return: list of frame ids to be removed
    '''
    frames_df = dataframe.groupby('frame_id')['agent_id'].apply(list).reset_index(name='agents')
    frames_df['agents_num'] = frames_df['agents'].apply(len)
    return list(frames_df[frames_df['agents_num'] < agents_threshold]['frame_id'].values)


def remove_agents_and_frames_with_insufficient_data(dataframe, agents_threshold, frames_threshold):
    '''
    Remove agents and frames with insufficient data, based on given thresholds.
    :param dataframe: dataframe to be filtered
    :param agents_threshold: minimum number of agents for frame not to be removed
    :param frames_threshold: minimum number of frames for agent not to be removed
    :return: filtered dataframe
    '''
    unwanted_frame_ids = check_for_frames_with_low_number_of_agents(dataframe, agents_threshold)
    unwanted_agent_ids = check_for_agents_in_low_number_of_frames(dataframe, frames_threshold)

    while len(unwanted_frame_ids) > 0 or len(unwanted_agent_ids) > 0:
        dataframe = dataframe[dataframe.agent_id.isin(unwanted_agent_ids) == False]
        dataframe = dataframe[dataframe.frame_id.isin(unwanted_frame_ids) == False]
        unwanted_frame_ids = check_for_frames_with_low_number_of_agents(dataframe, agents_threshold)
        unwanted_agent_ids = check_for_agents_in_low_number_of_frames(dataframe, frames_threshold)

    return dataframe


def filter_difference_between_frame_combinations(combinations, diff_between_frames):
    '''
    Filter frame combinations based on given difference between frames to be considered continuous.
    :param combinations: list of frame combinations to be filtered
    :param diff_between_frames: difference between frames to be continuous
    :return: list of filtered frame combinations
    '''
    filtered_combinations = []
    for frames in combinations:
        differences = [True for i, frame in enumerate(frames[:-1]) if frames[i + 1] - frame != diff_between_frames]
        if len(differences) == 0:
            filtered_combinations.append(frames)
    return filtered_combinations


def get_frame_combs_data(dataframe, agents_minimum, consecutive_frames, difference_between_frames):
    '''
    Get frame combinations based on given parameters.
    :param dataframe: dataframe to be filtered
    :param agents_minimum: minimum number of agents for frame not to be removed
    :param consecutive_frames: minimum number of frames for agent not to be removed
    :param difference_between_frames: difference between frames to be continuous
    :return: frame combinations after filtering
    '''
    # get agents by frame
    agents_by_frame = dataframe.groupby('frame_id')['agent_id'].apply(list).reset_index(name='agents')

    # get frame combinations
    frame_ids = agents_by_frame.frame_id.values
    frame_id_combinations = [list(frame_ids[i:i + consecutive_frames]) for i, frame_id in
                             enumerate(frame_ids[:-consecutive_frames])]
    frame_id_combinations = filter_difference_between_frame_combinations(frame_id_combinations,
                                                                         difference_between_frames)

    # check agents intersection in frame combinations
    combs = []
    for frames in frame_id_combinations:
        comb_dict = {}
        agent_list = [set(agents_by_frame[agents_by_frame['frame_id'] == frame]['agents'].iloc[0]) for frame in frames]
        comb_dict['frames'] = frames
        comb_dict['common_agents'] = set.intersection(*agent_list)
        comb_dict['total_agents'] = set.union(*agent_list)
        # ignore frame combinations with not enough common agents
        if len(comb_dict['common_agents']) >= agents_minimum:
            combs.append(comb_dict)

    return combs


def get_agent_data_for_frames(dataframe, agents, frames):
    '''
    Returns a list of tuples with location and velocity data for each frame and agent
    :param dataframe: dataframe to retrieve data
    :param agents: list of agents for who to retrieve data
    :param frames: list of frames for which to retrieve data
    :return: list of lists of data of each agent
    '''
    data = dataframe[dataframe['frame_id'].isin(frames) & dataframe['agent_id'].isin(agents)]
    return list(data.groupby('agent_id')['measurement'].apply(list).values)


def get_pair_label(groups, agents):
    '''
    Checks if agents are in the same group.
    :param groups: list of groups to search
    :param agents: tuple of agents to check
    :return: True if agents are in the same, otherwise False
    '''
    return any(all(agent in group for agent in agents) for group in groups)


def scene_sample(dataframe, groups, pair_agents, context_agents, frames, data, labels):
    '''
    Sampling scene by getting agents and label data.
    :param dataframe: dataframe to retrieve data
    :param groups: list of groups to search
    :param agents: list of agents in the scene
    :param frames: list of frames for which to get data
    :param data: list to store agent data
    :param labels: list to store group relationship
    :return: nothing
    '''
    pair_data = get_agent_data_for_frames(dataframe, pair_agents, frames)
    context_data = get_agent_data_for_frames(dataframe, context_agents, frames)
    pair_data.extend(context_data)
    data.append(pair_data)
    label = get_pair_label(groups, pair_agents)
    labels.append(label)


def filter_scene_pairs(pairs, group_pairs):
    '''
    Filter pairs in order to have balanced samples.
    :param pairs: list of pairs in scene
    :param group_pairs: list of pairs in same group
    :return: filtered list of pairs
    '''
    same = []
    different = []
    for pair in pairs:
        if pair in group_pairs:
            same.append(pair)
        else:
            different.append(pair)

    return same + random.sample(different, len(same))


def dataset_reformat(dataframe, groups, group_pairs, frame_comb_data, agents_minimum, scene_samples):
    '''
    Gather data from all possible scenes based on given parameters.
    :param dataframe: dataframe to retrieve data
    :param groups: list of groups
    :param frame_comb_data: valid continuous frame combinations
    :param agents_minimum: minimum agents (pair + context) in a scene
    :param scene_samples: maximum samples to get from a scene for each pair
    :return: dataset
    '''
    data = []
    labels = []
    frames = []
    for frame_comb in frame_comb_data:
        comb_frames = frame_comb['frames']
        comb_agents = frame_comb['common_agents']

        pairs = list(combinations(comb_agents, 2))
        pairs = filter_scene_pairs(pairs, group_pairs)
        for pair_agents in pairs:
            scene_agents = comb_agents - set(pair_agents)
            for i in range(scene_samples):
                context_agents = random.sample(scene_agents, agents_minimum - 2)
                scene_sample(dataframe, groups, pair_agents, context_agents, comb_frames, data, labels)
                frames.append(comb_frames)
    return np.asarray(data), np.asarray(labels), np.asarray(frames)


def get_group_pairs(groups):
    '''
    Get pairs of agents that are in same group.
    :param groups: list of groups
    :return: list of pairs of agents
    '''
    pairs = []
    for group in groups:
        pairs.extend(list(combinations(group, 2)))
    return pairs


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--report', action="store_true", default=False)
    parser.add_argument('-p', '--plot', action="store_true", default=False)
    parser.add_argument('-f', '--frames', type=int, default=1)
    parser.add_argument('-a', '--agents', type=int, default=10)
    parser.add_argument('-ss', '--scene_samples', type=int, default=5)
    parser.add_argument('-d', '--dataset', type=str, default='eth')
    parser.add_argument('-sf', '--save_folder', type=str, default='./reformatted')

    return parser.parse_args()


if __name__ == '__main__':
    start = datetime.now()

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

    for dataset in datasets_dict.keys():
        df = datasets_dict[dataset]['df']
        groups = datasets_dict[dataset]['groups']
        group_pairs = get_group_pairs(groups)
        difference = datasets_dict[dataset]['difference']

        # remove agents with low number of frames or agents
        df = remove_agents_and_frames_with_insufficient_data(dataframe=df, frames_threshold=args.frames,
                                                             agents_threshold=args.agents)

        # get frame combinations data
        combs = get_frame_combs_data(dataframe=df, agents_minimum=args.agents,
                                     consecutive_frames=args.frames, difference_between_frames=difference)

        # format dataset to be used by proposed approach
        data, labels, frames = dataset_reformat(dataframe=df, groups=groups, group_pairs=group_pairs,
                                                frame_comb_data=combs, agents_minimum=args.agents,
                                                scene_samples=args.scene_samples)

        filename = '{}/{}_{}_{}_data.npy'.format(args.save_folder, dataset, args.frames, args.agents)
        np.save(filename, data)
        filename = '{}/{}_{}_{}_labels.npy'.format(args.save_folder, dataset, args.frames, args.agents)
        np.save(filename, labels)
        filename = '{}/{}_{}_{}_frames.npy'.format(args.save_folder, dataset, args.frames, args.agents)
        np.save(filename, frames)

        end = datetime.now()
        print('Dataset: {}, Duration: {}'.format(dataset, end - start))
        start = end
