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
        worksheet.write(row, 3, stats[key]['groups'])
        worksheet.write(row, 4, stats[key]['duration'])
        # worksheet.write(row, 4, stats[key]['multigroup agents'])
        # worksheet.write(row, 5, stats[key]['single agent groups'])
        row += 1

    workbook.close()


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
        'agents': agents_num,
        'frames': frames_num,
        'groups': len(groups),
        'single agent groups': single_groups,
        'duration': df.loc[df.frame_id.idxmax()]['timestamp'] - df.loc[df.frame_id.idxmin()]['timestamp']
    }


def groups_size_hist(groups_dict, save_loc):
    '''
    Produces a plot of counts of group lengths per dataset
    :param groups_dict:
    :return:
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


if __name__ == '__main__':
    # create datasets report
    stats_dict = {
        'eth': dataset_stats('./ETH/seq_eth'),
        'hotel': dataset_stats('./ETH/seq_hotel'),
        'zara01': dataset_stats('./UCY/zara01'),
        'zara02': dataset_stats('./UCY/zara02'),
        'students03': dataset_stats('./UCY/students03')
    }
    report('datasets.xlsx', stats_dict)

    # create datasets group size histogram
    groups_dict = {
        'eth': read_groups('./ETH/seq_eth'),
        'hotel': read_groups('./ETH/seq_hotel'),
        'zara01': read_groups('./UCY/zara01'),
        'zara02': read_groups('./UCY/zara02'),
        'students03': read_groups('./UCY/students03')
    }
    groups_size_hist(groups_dict, './group_size_plot.png')
