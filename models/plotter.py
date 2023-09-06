import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()

        line_counter = 1
        while line_counter < len(lines):
            mean_line = lines[line_counter]
            std_line = lines[line_counter + 1]

            new_line = mean_line.rstrip() + ',' + std_line[std_line.find(',') + 1:].rstrip()
            data.append(new_line.split(','))
            line_counter += 2

    columns = lines[0].rstrip().split(',')
    columns += [column + '_std' for column in columns[1:]]
    dataframe = pd.DataFrame(data, columns=columns)

    return dataframe


def modify_df(dataframe, no_context=False):
    details = dataframe['dataset'].str.split('_')
    details_size = len(details[0])
    dataframe['dataset'] = details.apply(lambda x: x[0])
    if details_size == 2:
        dataframe['name'] = 'WavenetNRI'
    else:
        if details[0][1] == '1':
            dataframe['name'] = 'DANTE c' + details.apply(lambda x: str(int(x[2]) - 2))
        else:
            if no_context:
                dataframe['name'] = 'T-DANTE nc'
            else:
                dataframe['name'] = 'T-DANTE c' + details.apply(lambda x: str(int(x[2]) - 2))

    float_columns = list(dataframe.columns.values[1:-1])
    dataframe[float_columns] = dataframe[float_columns].astype(float)

    return dataframe


def modify_sim_df(dataframe, no_context=False, fix_datasets=False):
    if fix_datasets:
        dataframe['dataset'] = dataframe['dataset'].replace('8_3_2_2_nri', 'sim_1')
        dataframe['dataset'] = dataframe['dataset'].replace('9_3_2_2_nri', 'sim_2')
        dataframe['dataset'] = dataframe['dataset'].replace('9_3_2_3_nri', 'sim_3')
        dataframe['dataset'] = dataframe['dataset'].replace('10_3_2_2_nri', 'sim_4')
        dataframe['dataset'] = dataframe['dataset'].replace('10_3_2_3_nri', 'sim_5')
        dataframe['dataset'] = dataframe['dataset'].replace('10_3_2_4_nri', 'sim_6')
        dataframe['name'] = 'WavenetNRI'
    else:
        details = dataframe['dataset'].str.split('_')
        dataframe['dataset'] = details.apply(lambda x: '_'.join(x[0:2]))
        if no_context:
            dataframe['name'] = 'T-DANTE nc'
        else:
            dataframe['name'] = 'T-DANTE c' + details.apply(lambda x: str(int(x[-1]) - 2))

    float_columns = list(dataframe.columns.values[1:-1])
    dataframe[float_columns] = dataframe[float_columns].astype(float)

    return dataframe


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nri', action="store_true", default=True)
    parser.add_argument('--no_context', action="store_true", default=False)
    parser.add_argument('--single_dataset', action="store_true", default=False)
    parser.add_argument('--sim', action="store_true", default=False)
    parser.add_argument('--dataset', type=str, default="sim_1")

    return parser.parse_args()


def plot_df(dataframe, metric, ylabel, title, savefile):
    # plt.figure(figsize=(12, 6))
    sns.barplot(data=dataframe, x='dataset', y=metric, hue='name', errorbar='sd', errcolor='red')
    plt.xlabel('Dataset')
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.ylim(0, 1)
    plt.legend(title='Model', loc='lower center')
    # plt.tight_layout()
    plt.savefig(savefile)
    plt.show()


if __name__ == '__main__':
    args = get_args()

    # read data
    pede_nri_results = read_csv("./WavenetNRI/logs/nripedsu/all_f_15_nri_results.csv")
    pede_nc_results = read_csv("./results/all_nc_f_15_a_6_results.csv")
    pede_dante_results = read_csv("./results/all_f_1_a_6_10_results.csv")
    pede_tdante_results = read_csv("./results/all_f_15_a_6_10_results.csv")
    sim_nri_results = read_csv("./WavenetNRI/logs/nrisu/all_sim_f_15_nri_results.csv")
    sim_nc_results = read_csv("./results/all_sim_nc_f_49_a_6_10_results.csv")
    # sim_dante_results = read_csv("")
    sim_tdante_results = read_csv("./results/all_sim_f_49_a_6_10_results.csv")

    pede_nri_results = modify_df(pede_nri_results)
    pede_nc_results = modify_df(pede_nc_results, no_context=True)
    pede_dante_results = modify_df(pede_dante_results)
    pede_tdante_results = modify_df(pede_tdante_results)

    sim_nri_results = modify_sim_df(sim_nri_results, fix_datasets=True)
    sim_nc_results = modify_sim_df(sim_nc_results, no_context=True)
    sim_tdante_results = modify_sim_df(sim_tdante_results)

    # final form data
    abl_pede = pd.concat([pede_nc_results, pede_tdante_results])
    abl_sim = pd.concat([sim_nc_results, sim_tdante_results])
    bas_pede = pd.concat([pede_dante_results, pede_nri_results, pede_tdante_results])
    bas_pede = bas_pede[~bas_pede['name'].isin(['DANTE c4', 'T-DANTE c4'])]
    bas_sim = pd.concat([sim_nri_results, sim_tdante_results])
    bas_sim = bas_sim[~bas_sim['name'].isin(['DANTE c4', 'T-DANTE c4'])]

    sns.set(style='whitegrid')
    plot_df(abl_pede, metric='f1_1', ylabel='F1',
            title='Group Correctness P=1 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='abl_pede_f1_1')
    plot_df(bas_pede, metric='f1_1', ylabel='F1',
            title='Group Correctness P=1 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='bas_pede_f1_1')
    plot_df(abl_pede, metric='f1_2/3', ylabel='F1',
            title='Group Correctness P=2/3 F1 values\nfor each Model and Pedestrian Dataset', savefile='abl_pede_f1_23')
    plot_df(bas_pede, metric='f1_2/3', ylabel='F1',
            title='Group Correctness P=2/3 F1 values\nfor each Model and Pedestrian Dataset', savefile='bas_pede_f1_23')
    plot_df(abl_pede, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Pedestrian Dataset',
            savefile='abl_pede_f1_gmitre')
    plot_df(bas_pede, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Pedestrian Dataset',
            savefile='bas_pede_f1_gmitre')

    plot_df(abl_sim, metric='f1_1', ylabel='F1',
            title='Group Correctness P=1 F1 values\nfor each Model and Simulation Dataset',
            savefile='abl_sim_f1_1')
    plot_df(bas_sim, metric='f1_1', ylabel='F1',
            title='Group Correctness P=1 F1 values\nfor each Model and Simulation Dataset',
            savefile='bas_sim_f1_1')
    plot_df(abl_sim, metric='f1_2/3', ylabel='F1',
            title='Group Correctness P=2/3 F1 values\nfor each Model and Simulation Dataset',
            savefile='abl_sim_f1_23')
    plot_df(bas_sim, metric='f1_2/3', ylabel='F1',
            title='Group Correctness P=2/3 F1 values\nfor each Model and Simulation Dataset',
            savefile='bas_sim_f1_23')
    plot_df(abl_sim, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Simulation Dataset',
            savefile='abl_sim_f1_gmitre')
    plot_df(bas_sim, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Simulation Dataset',
            savefile='bas_sim_f1_gmitre')
