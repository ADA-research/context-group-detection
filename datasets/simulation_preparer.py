import argparse
import random
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import xlsxwriter
from matplotlib import pyplot as plt

from datasets.loader import read_sim, read_multi_groups
from datasets.preparer import dataset_reformat, save_folds, get_scene_data


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
    # TODO possibly use sim column to differentiate frames
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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('--samples_freq', type=int, default=50)
    parser.add_argument('-f', '--frames_num', type=int, default=10)
    parser.add_argument('-a', '--agents_num', type=int, default=10)
    parser.add_argument('-ts', '--target_size', type=int, default=100000)
    parser.add_argument('-d', '--dataset', type=str, default='eth')
    parser.add_argument('-sf', '--save_folder', type=str, default='./reformatted')
    parser.add_argument('-p', '--plot', action="store_true", default=False)
    parser.add_argument('-s', '--shift', action="store_true", default=True)
    parser.add_argument('-r', '--report', action="store_true", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    start = datetime.now()
    print('Started at {}'.format(start))

    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # create datasets report
    datasets_dict = {
        'sim_1': dataset_data('./simulation/sim_10_3_5', args.samples_freq)
    }
    if args.report:
        report('datasets.xlsx', datasets_dict)
        exit()

    # create datasets group size histogram
    groups_dict = {
        'sim_1': read_multi_groups('./simulation/sim_10_3_5')
    }
    if args.plot:
        groups_size_hist(groups_dict, './group_size_plot.png')
        exit()

    multi_frame = True
    if args.frames_num == 1:
        multi_frame = False

    for dataset in datasets_dict.keys():
        dataset_start = datetime.now()
        print('Dataset: {}, started at: {}'.format(dataset, dataset_start))

        df = datasets_dict[dataset]['df']
        groups = datasets_dict[dataset]['groups']
        difference = datasets_dict[dataset]['difference']

        group_pairs = get_group_pairs(groups)

        scenes = get_scene_data(dataframe=df, consecutive_frames=args.frames_num, difference_between_frames=difference,
                                groups=groups, step=1)

        # format dataset to be used by proposed approach
        data, labels, frames, filtered_groups = dataset_reformat(dataframe=df,
                                                                 scene_data=scenes,
                                                                 groups=groups,
                                                                 group_pairs=group_pairs,
                                                                 agents_num=args.agents_num,
                                                                 shift=args.shift,
                                                                 min_pair_samples=1,
                                                                 max_pair_samples=1)
        dataset = '{}_shifted'.format(dataset) if args.shift else dataset
        # save dataset in folds
        save_folds(args.save_folder, dataset, args.frames_num, args.agents_num, data, labels, frames,
                   filtered_groups, multi_frame)

        end = datetime.now()
        print('Dataset: {}, finished in: {}'.format(dataset, end - dataset_start))
        print('\tdata size: {}'.format(len(data)))
        dataset_start = end

    end = datetime.now()
    print('Finished in: {}'.format(end - start))
