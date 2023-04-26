from collections import Counter

import pandas as pd
import seaborn as sns
import xlsxwriter
from matplotlib import pyplot as plt

from loader import read_obsmat, read_groups


def report(name, stats):
    '''
    Generate excel report file with dataset stats.
    :param name: string for Excel file name
    :param stats: dictionary of stats for every dataset
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
    for key in stats.keys():
        worksheet.write(row, 0, key)
        worksheet.write(row, 1, stats[key]['agents'])
        worksheet.write(row, 2, stats[key]['frames'])
        worksheet.write(row, 3, len(stats[key]['groups']))
        worksheet.write(row, 4, stats[key]['duration'])
        # worksheet.write(row, 4, stats[key]['multigroup agents'])
        # worksheet.write(row, 5, stats[key]['single agent groups'])
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


def dataset_stats(dataset_path):
    '''
    Get stats for specified dataset.
    :param dataset_path: string of where to find dataset
    :return: dictionary with stats
    '''
    df = read_obsmat(dataset_path)
    groups = read_groups(dataset_path)

    agents_num = df.agent_id.unique().size
    frames_num = df.frame_id.unique().size

    count_dict = Counter([agent for group in groups for agent in group])
    agents_in_groups = [agent for agent in count_dict.elements()]
    single_groups = agents_num - len(agents_in_groups)

    return {
        'df': df,
        'groups': groups,
        'agents': agents_num,
        'frames': frames_num,
        'single agent groups': single_groups,
        'duration': df.loc[df.frame_id.idxmax()]['timestamp'] - df.loc[df.frame_id.idxmin()]['timestamp']
    }


def remove_agents_in_low_number_of_frames(dataframe, frames_threshold):
    '''
    Filters dataframe to find agents with frames less than the given threshold.
    :param dataframe: dataframe to be filtered
    :param frames_threshold: minimum number of frames for agent not to be removed
    :return: filtered dataframe
    '''
    agents_df = dataframe.groupby('agent_id')['frame_id'].apply(list).reset_index(name='frames')
    agents_df['frames_num'] = agents_df['frames'].apply(len)
    agent_ids_to_be_removed = list(agents_df[agents_df['frames_num'] < frames_threshold]['agent_id'].values)

    return dataframe[not dataframe.agent_id.isin(agent_ids_to_be_removed)]


def remove_frames_with_low_number_of_agents(dataframe, agents_threshold):
    '''
    Filters dataframe to find frames with agents less than the given threshold.
    :param dataframe: dataframe to be filtered
    :param agents_threshold: minimum number of agents for frame not to be removed
    :return: filtered dataframe
    '''
    frames_df = dataframe.groupby('frame_id')['agent_id'].apply(list).reset_index(name='agents')
    frames_df['agents_num'] = frames_df['agents'].apply(len)
    frame_ids_to_be_removed = list(frames_df[frames_df['agents_num'] < agents_threshold]['frame_id'].values)

    return dataframe[not dataframe.frame_id.isin(frame_ids_to_be_removed)]


# TODO calculate number of same agents in X consecutive frames,
#  find set of agents in each frame, check intersection between the sets
def get_frame_combs_stats(dataframe, consecutive_frames):
    agent_sets = dataframe.groupby('frame_id')['agent_id'].apply(list).reset_index(name='agents')

    frame_ids = agent_sets.frame_id.values
    frame_id_combs = [list(frame_ids[i:i + consecutive_frames]) for i, frame_id in
                      enumerate(frame_ids[:-consecutive_frames])]

    combs = []
    for frames in frame_id_combs:
        comb_dict = {}
        agent_list = [set(agent_sets[agent_sets['frame_id'] == frame]['agents'].iloc[0]) for frame in frames]
        comb_dict['frames'] = frames
        comb_dict['common_agents'] = set.intersection(*agent_list)
        comb_dict['total_agents'] = set.union(*agent_list)
        combs.append(comb_dict)

    return combs


# TODO for agents with lower than X frames, generate data to reach X frames
# TODO check number of data when time window is 1 frame
# TODO check number of data when time window is X frame

if __name__ == '__main__':
    # create datasets report
    datasets_dict = {
        'eth': dataset_stats('./ETH/seq_eth'),
        'hotel': dataset_stats('./ETH/seq_hotel'),
        'zara01': dataset_stats('./UCY/zara01'),
        'zara02': dataset_stats('./UCY/zara02'),
        'students03': dataset_stats('./UCY/students03')
    }
    # uncomment to produce report
    # report('datasets.xlsx', datasets_dict)

    # create datasets group size histogram
    groups_dict = {
        'eth': read_groups('./ETH/seq_eth'),
        'hotel': read_groups('./ETH/seq_hotel'),
        'zara01': read_groups('./UCY/zara01'),
        'zara02': read_groups('./UCY/zara02'),
        'students03': read_groups('./UCY/students03')
    }
    # uncomment to produce plot
    # groups_size_hist(groups_dict, './group_size_plot.png')

    # select one dataframe
    dataset = 'eth'
    df = datasets_dict[dataset]['df']
    groups = datasets_dict[dataset]['groups']

    # get frame stats
    # get_frame_combs_stats(dataframe=df, consecutive_frames=3)

    # remove agents with low number of frames
    # TODO figure out how to filter until both dont have anything else to remove
    df = remove_agents_in_low_number_of_frames(dataframe=df, frames_threshold=5)
    df = remove_frames_with_low_number_of_agents(dataframe=df, agents_threshold=5)
    pass
