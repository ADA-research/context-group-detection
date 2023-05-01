from collections import Counter

import pandas as pd
import seaborn as sns
import xlsxwriter
from matplotlib import pyplot as plt

from loader import read_obsmat, read_groups


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


# TODO handle frames with more than agents_minimum
#  get all combinations of agents_minimum length for the common agents
#  create data with frames - agents in frame comb - data of each agent (location, velocity, frame)
def dataset_creator(dataframe, frame_comb_data, agents_minimum):
    dataset = []
    for frame_comb in frame_comb_data:
        # for frames with minimum agents data
        # get trajectories of each agent in an array
        if len(frame_comb['common_agents']) >= agents_minimum:
            pass
        # for frames with more agents
        # get all possible combinations of common agents and handle them as different samples
        else:
            pass
    return dataset


if __name__ == '__main__':
    # create datasets report
    datasets_dict = {
        'eth': dataset_data('./ETH/seq_eth'),
        'hotel': dataset_data('./ETH/seq_hotel'),
        'zara01': dataset_data('./UCY/zara01'),
        'zara02': dataset_data('./UCY/zara02'),
        'students03': dataset_data('./UCY/students03')
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

    # remove agents with low number of frames
    df = remove_agents_and_frames_with_insufficient_data(dataframe=df, frames_threshold=5, agents_threshold=7)

    # get frame combinations data
    combinations = get_frame_combs_data(dataframe=df, agents_minimum=7, consecutive_frames=3,
                                        difference_between_frames=6)
    pass
