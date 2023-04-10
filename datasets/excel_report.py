from collections import Counter

import xlsxwriter

from loader import read_obsmat, read_groups


def report(name, stats):
    workbook = xlsxwriter.Workbook(name)
    worksheet = workbook.add_worksheet('Datasets')
    header_row = 0
    header_column = 0
    worksheet.write(header_row, header_column, 'Dataset')
    worksheet.write(header_row, header_column + 1, 'Agents #')
    worksheet.write(header_row, header_column + 2, 'Frames #')
    worksheet.write(header_row, header_column + 3, 'Groups #')
    worksheet.write(header_row, header_column + 4, 'Agents # in multiple groups')
    worksheet.write(header_row, header_column + 5, 'Single agent groups #')

    row = header_row + 1
    for key in stats.keys():
        worksheet.write(row, 0, key)
        worksheet.write(row, 1, stats[key]['agents'])
        worksheet.write(row, 2, stats[key]['frames'])
        worksheet.write(row, 3, stats[key]['groups'])
        worksheet.write(row, 4, stats[key]['multigroup agents'])
        worksheet.write(row, 5, stats[key]['single agent groups'])
        row += 1

    workbook.close()


def dataset_stats(dataset_path):
    df = read_obsmat(dataset_path)
    groups = read_groups(dataset_path)

    agents_num = df.agent_id.unique().size
    frames_num = df.frame_id.unique().size

    count_dict = Counter([agent for group in groups for agent in group])
    agents_in_multiple_groups = [key for key, value in count_dict.items() if value > 1]
    agents_in_groups = [agent for agent in count_dict.elements()]
    single_groups = agents_num - len(agents_in_groups)

    return {
        'agents': agents_num,
        'frames': frames_num,
        'groups': len(groups),
        'multigroup agents': len(agents_in_multiple_groups),
        'single agent groups': single_groups
    }


if __name__ == '__main__':
    stats = {
        'eth': dataset_stats('./ETH/seq_eth'),
        'hotel': dataset_stats('./ETH/seq_hotel'),
        'zara01': dataset_stats('./UCY/zara01'),
        'zara02': dataset_stats('./UCY/zara02'),
        'students03': dataset_stats('./UCY/students03')
    }

    workbook_name = 'datasets.xlsx'
    report(workbook_name, stats)
