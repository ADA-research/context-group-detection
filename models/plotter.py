import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.collector import collect_nri_results, collect_results, collect_sim_results, collect_nri_sim_results


def create_nri_df(results, suffix):
    dfs = []
    for dataset, dataset_results in results:
        f1_one = [res['f1_one']['f1'] for res in dataset_results]
        f1_two_thirds = [res['f1_two_thirds']['f1'] for res in dataset_results]
        f1_gmitre = [res['f1_gmitre']['f1'] for res in dataset_results]

        data_dict = {
            'dataset': '{}_{}'.format(dataset, suffix),
            'f1_1': f1_one,
            'f1_2/3': f1_two_thirds,
            'f1_gmitre': f1_gmitre
        }

        df = pd.DataFrame(data_dict)
        dfs.append(df)
    final_df = pd.concat(dfs)

    return final_df


def create_df(results):
    dfs = []
    for info, dataset_results in results:
        info_str = '{}_{}_{}'.format(info[0], info[1], info[2])
        dataset_info = [info_str for res in dataset_results]
        f1_one = [res['best_val_f1_1']['value'] for res in dataset_results]
        f1_two_thirds = [res['best_val_f1_2/3']['value'] for res in dataset_results]
        f1_gmitre = [res['best_val_f1_gmitre']['value'] for res in dataset_results]

        data_dict = {
            'dataset': dataset_info,
            'f1_1': f1_one,
            'f1_2/3': f1_two_thirds,
            'f1_gmitre': f1_gmitre
        }
        df = pd.DataFrame(data_dict)
        dfs.append(df)
    final_df = pd.concat(dfs)

    return final_df


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


def modify_df(dataframe, name=None, no_context=False, model=None):
    details = dataframe['dataset'].str.split('_')
    details_size = len(details.iloc[0])
    dataframe['dataset'] = details.apply(lambda x: x[0])
    if details_size == 2:
        dataframe['name'] = name
    else:
        if details.iloc[0][1] == '1':
            dataframe['name'] = 'DANTE c' + details.apply(lambda x: str(int(x[2]) - 2))
        else:
            if no_context:
                dataframe['name'] = 'T-DANTE GD nc' if model == 'gd' else 'T-DANTE nc'
            else:
                dataframe['name'] = ('T-DANTE GD c' if model == 'gd' else 'T-DANTE c') + details.apply(
                    lambda x: str(int(x[2]) - 2))

    float_columns = list(dataframe.columns.values[1:-1])
    dataframe[float_columns] = dataframe[float_columns].astype(float)

    return dataframe


def modify_sim_df(dataframe, name=None, no_context=False, fix_datasets=False, model=None, suffix=None,
                  single_frame=False):
    if fix_datasets:
        dataframe['dataset'] = dataframe['dataset'].str.replace('_{}'.format(suffix), '')
        dataframe['dataset'] = dataframe['dataset'].replace('8_3_2_2', 'sim_1')
        dataframe['dataset'] = dataframe['dataset'].replace('9_3_2_2', 'sim_2')
        dataframe['dataset'] = dataframe['dataset'].replace('9_3_2_3', 'sim_3')
        dataframe['dataset'] = dataframe['dataset'].replace('10_3_2_2', 'sim_4')
        dataframe['dataset'] = dataframe['dataset'].replace('10_3_2_3', 'sim_5')
        dataframe['dataset'] = dataframe['dataset'].replace('10_3_2_4', 'sim_6')
        dataframe['name'] = name
    else:
        details = dataframe['dataset'].str.split('_')
        dataframe['dataset'] = details.apply(lambda x: '_'.join(x[0:2]))
        if single_frame:
            dataframe['name'] = 'DANTE c8'
        else:
            if no_context:
                dataframe['name'] = 'T-DANTE GD nc' if model == 'gd' else 'T-DANTE nc'
            else:
                dataframe['name'] = ('T-DANTE GD c' if model == 'gd' else 'T-DANTE c') + details.apply(
                    lambda x: str(int(x[-1]) - 2))

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


def save_latex_data(dataframe, metric, savefile, title, label):
    with open(savefile, 'w') as file:
        models = dataframe['name'].unique()
        datasets = dataframe['dataset'].unique()
        datasets_str = f'& {" & ".join(datasets)}'
        datasets_str = datasets_str.replace('_', '\_')
        if 'sim' in datasets_str:
            file.write(
                f'\\begin{{table}}[]\n\def\\arraystretch{{1.35}}\n\centering\n\\begin{{tabular}}{{c|c|c|c|c|c|c|}}\n\cline{{2-7}}\n{datasets_str} \\\\ \hline\n')
        else:
            file.write(
                f'\\begin{{table}}[]\n\def\\arraystretch{{1.35}}\n\centering\n\\begin{{tabular}}{{c|c|c|c|c|c|}}\n\cline{{2-6}}\n{datasets_str} \\\\ \hline\n')
        metric_data_all = []
        std_data_all = []
        for model in models:
            metric_data = []
            std_data = []
            for dataset in datasets:
                model_data = dataframe[(dataframe['name'] == model) & (dataframe['dataset'] == dataset)]
                metric_data.append(np.round(np.mean(model_data[metric]), 4))
                std_data.append(np.round(np.std(model_data[metric]), 4))
            metric_data_all.append(metric_data)
            std_data_all.append(std_data)
        # Find the index of the maximum value in each column
        max_indices = [metric_data.index(max(metric_data)) for metric_data in zip(*metric_data_all)]
        # Create a 2D boolean array to mark maximum values
        is_max = [[i == max_indices[j] for i in range(len(models))] for j in range(len(datasets))]
        transposed_is_max = list(map(list, zip(*is_max)))
        for model, metric_data, std_data, max_flags in zip(models, metric_data_all, std_data_all, transposed_is_max):
            # Write model name
            file.write(
                f'\multicolumn{{1}}{{|c|}}{{\multirow{{2}}{{*}}{{\\begin{{tabular}}[c]{{@{{}}c@{{}}}}\n{model}\n\end{{tabular}}}}}}\n')
            # Write metric_data, making the maximum values in each column bold
            for i, (value, is_max_value) in enumerate(zip(metric_data, max_flags)):
                if is_max_value:
                    file.write(f' & \\textbf{{{value}}}')
                else:
                    file.write(f' & {value}')
            # Write standard deviation
            std_data_str = ' & $\pm$'.join(map(str, std_data))
            file.write(f' \\\\\n\multicolumn{{1}}{{|c|}}{{}} & $\pm${std_data_str} \\\\ \hline\n')
        file.write(f'\end{{tabular}}\n\caption{{{title}}}\n\label{{{label}}}\n\end{{table}}')


def plot_df(dataframe, metric, ylabel, title, savefile):
    # plt.figure(figsize=(12, 6))
    sns.barplot(data=dataframe, x='dataset', y=metric, hue='name', errorbar='sd')
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

    nri_datasets = ['8_3_2_2', '9_3_2_2', '9_3_2_3', '10_3_2_2', '10_3_2_3', '10_3_2_4']
    sim_datasets = ['sim_1', 'sim_2', 'sim_3', 'sim_4', 'sim_5', 'sim_6']
    pede_datasets = ['eth', 'hotel', 'zara01', 'zara02', 'students03']

    pede_wavenet_results = []
    pede_gdgan_results = []
    pede_nri_results = []
    pede_nc_results = []
    pede_nc_gd_results = []
    pede_dante_results = []
    pede_tdante_results = []
    pede_tdante_gd_results = []
    sim_wavenet_results = []
    sim_gdgan_results = []
    sim_nri_results = []
    sim_nc_results = []
    sim_nc_gd_results = []
    # sim_dante_results = []
    sim_tdante_results = []
    sim_tdante_gd_results = []

    for dataset in pede_datasets:
        dataset_results = []
        results_path = './WavenetNRI/logs/nripedsu/wavenetsym_{}_shifted_{}'.format(dataset, 15)
        pede_wavenet_results.append((dataset, collect_nri_results(results_path, average=False)))
        results_path = './WavenetNRI/logs/nripedsu/cnn_{}_shifted_{}'.format(dataset, 15)
        pede_nri_results.append((dataset, collect_nri_results(results_path, average=False)))
        results_path = './GDGAN/logs/nripedsu/{}_shifted_{}'.format(dataset, 15)
        pede_gdgan_results.append((dataset, collect_nri_results(results_path, average=False)))
        for agents_num in [6, 10]:
            pede_dante_results.append((
                (dataset, 1, agents_num),
                collect_results('./results/{}_shifted_1_{}'.format(dataset, agents_num), 'e150', average=False)))
            frames_num = 15
            results_path = './results/{}_shifted_{}_{}_'.format(dataset, frames_num, agents_num)
            results_path_nc = './results/{}_shifted_{}_{}'.format(dataset, frames_num, agents_num)
            results_path_gd = './results/{}_shifted_{}_{}_gd'.format(dataset, frames_num, agents_num)
            pede_tdante_results.append(
                ((dataset, frames_num, agents_num), collect_results(results_path, 'e150', average=False)))
            pede_tdante_gd_results.append(
                ((dataset, frames_num, agents_num), collect_results(results_path_gd, 'e150', average=False)))
            if agents_num == 6:
                pede_nc_results.append(
                    ((dataset, frames_num, agents_num), collect_results(results_path_nc, 'e150_nc', average=False)))
                pede_nc_gd_results.append(
                    ((dataset, frames_num, agents_num), collect_results(results_path_gd, 'e150_nc', average=False)))

    for dataset in sim_datasets:
        # sim_dante_results.append((
        #     (dataset, 1, 10),
        #     collect_sim_results('./results/{}_shifted_1_{}'.format(dataset, 10), 'e50', average=False)))
        for agents_num in [6, 10]:
            frames_num = 49
            if dataset in ['sim_4', 'sim_5', 'sim_6']:
                results_path = './results/{}_shifted_{}_{}'.format(dataset, frames_num, agents_num)
            else:
                results_path = './results/{}_shifted_{}_{}_'.format(dataset, frames_num, agents_num)
            results_path_nc = './results/{}_shifted_{}_{}'.format(dataset, frames_num, agents_num)
            results_path_gd = './results/{}_shifted_{}_{}_gd'.format(dataset, frames_num, agents_num)
            sim_tdante_results.append(
                ((dataset, frames_num, agents_num), collect_sim_results(results_path, 'e50', average=False)))
            sim_tdante_gd_results.append(
                ((dataset, frames_num, agents_num), collect_sim_results(results_path_gd, 'e50', average=False)))
            if agents_num == 6:
                sim_nc_results.append(
                    ((dataset, frames_num, agents_num), collect_sim_results(results_path_nc, 'e50_nc', average=False)))
                sim_nc_gd_results.append(
                    ((dataset, frames_num, agents_num), collect_sim_results(results_path_gd, 'e50_nc', average=False)))

    for dataset in nri_datasets:
        results_path = './WavenetNRI/logs/nrisu/su_wavenetsym_{}'.format(dataset)
        sim_wavenet_results.append((dataset, collect_nri_sim_results(results_path, average=False)))
        results_path = './WavenetNRI/logs/nrisu/su_cnn_{}'.format(dataset)
        sim_nri_results.append((dataset, collect_nri_sim_results(results_path, average=False)))
        results_path = './GDGAN/logs/nrisu/{}'.format(dataset)
        sim_gdgan_results.append((dataset, collect_nri_sim_results(results_path, average=False)))

    # read data
    pede_wavenet_results = create_nri_df(pede_wavenet_results, 'wavenet')
    pede_nri_results = create_nri_df(pede_nri_results, 'nri')
    pede_gdgan_results = create_nri_df(pede_gdgan_results, 'gdgan')
    pede_nc_results = create_df(pede_nc_results)
    pede_nc_gd_results = create_df(pede_nc_gd_results)
    pede_dante_results = create_df(pede_dante_results)
    pede_tdante_results = create_df(pede_tdante_results)
    pede_tdante_gd_results = create_df(pede_tdante_gd_results)
    sim_wavenet_results = create_nri_df(sim_wavenet_results, 'wavenet')
    sim_nri_results = create_nri_df(sim_nri_results, 'nri')
    sim_gdgan_results = create_nri_df(sim_gdgan_results, 'gdgan')
    sim_nc_results = create_df(sim_nc_results)
    sim_nc_gd_results = create_df(sim_nc_gd_results)
    # sim_dante_results = create_df(sim_dante_results)
    sim_tdante_results = create_df(sim_tdante_results)
    sim_tdante_gd_results = create_df(sim_tdante_gd_results)

    pede_wavenet_results = modify_df(pede_wavenet_results, name='WavenetNRI')
    pede_nri_results = modify_df(pede_nri_results, name='NRI')
    pede_gdgan_results = modify_df(pede_gdgan_results, name='GDGAN')
    pede_nc_results = modify_df(pede_nc_results, no_context=True)
    pede_nc_gd_results = modify_df(pede_nc_gd_results, no_context=True, model='gd')
    pede_dante_results = modify_df(pede_dante_results)
    pede_tdante_results = modify_df(pede_tdante_results)
    pede_tdante_gd_results = modify_df(pede_tdante_gd_results, model='gd')

    sim_wavenet_results = modify_sim_df(sim_wavenet_results, name='WavenetNRI', fix_datasets=True, suffix='wavenet')
    sim_nri_results = modify_sim_df(sim_nri_results, name='NRI', fix_datasets=True, suffix='nri')
    sim_gdgan_results = modify_sim_df(sim_gdgan_results, name='GDGAN', fix_datasets=True, suffix='gdgan')
    sim_nc_results = modify_sim_df(sim_nc_results, no_context=True)
    sim_nc_gd_results = modify_sim_df(sim_nc_gd_results, no_context=True, model='gd')
    # sim_dante_results = modify_sim_df(sim_dante_results, name='DANTE', single_frame=True)
    sim_tdante_results = modify_sim_df(sim_tdante_results)
    sim_tdante_gd_results = modify_sim_df(sim_tdante_gd_results, model='gd')

    # final form data
    abl_pede = pd.concat([pede_nc_results, pede_tdante_results, pede_nc_gd_results, pede_tdante_gd_results])
    abl_sim = pd.concat([sim_nc_results, sim_tdante_results, sim_nc_gd_results, sim_tdante_gd_results])
    bas_pede = pd.concat([pede_dante_results,
                          pede_nri_results,
                          pede_gdgan_results,
                          pede_wavenet_results, pede_tdante_results])
    bas_pede = bas_pede[~bas_pede['name'].isin(['DANTE c4', 'T-DANTE c4'])]
    bas_sim = pd.concat([sim_wavenet_results,
                         sim_nri_results,
                         sim_gdgan_results,
                         # sim_dante_results,
                         sim_tdante_results])
    bas_sim = bas_sim[~bas_sim['name'].isin(['T-DANTE c8'])]

    sns.set(style='whitegrid')
    plot_df(abl_pede, metric='f1_1', ylabel='F1',
            title='Group Correctness with P=1 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/abl_pede_f1_1')
    plot_df(bas_pede, metric='f1_1', ylabel='F1',
            title='Group Correctness with P=1 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/bas_pede_f1_1')
    plot_df(abl_pede, metric='f1_2/3', ylabel='F1',
            title='Group Correctness with P=2/3 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/abl_pede_f1_23')
    plot_df(bas_pede, metric='f1_2/3', ylabel='F1',
            title='Group Correctness with P=2/3 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/bas_pede_f1_23')
    plot_df(abl_pede, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/abl_pede_f1_gmitre')
    plot_df(bas_pede, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/bas_pede_f1_gmitre')

    plot_df(abl_sim, metric='f1_1', ylabel='F1',
            title='Group Correctness with P=1 F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/abl_sim_f1_1')
    plot_df(bas_sim, metric='f1_1', ylabel='F1',
            title='Group Correctness with P=1 F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/bas_sim_f1_1')
    plot_df(abl_sim, metric='f1_2/3', ylabel='F1',
            title='Group Correctness with P=2/3 F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/abl_sim_f1_23')
    plot_df(bas_sim, metric='f1_2/3', ylabel='F1',
            title='Group Correctness with P=2/3 F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/bas_sim_f1_23')
    plot_df(abl_sim, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/abl_sim_f1_gmitre')
    plot_df(bas_sim, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/bas_sim_f1_gmitre')

    save_latex_data(abl_pede, metric='f1_1', savefile='latex/abl_pede_f1_1.tex',
                    title='Group Correctness metric with $P=1$ for T-DANTE variations in all pedestrian datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 15 consecutive timeframes. \\textbf{*} shows that this result is significantly different than all the other values in the same dataset.',
                    label='tab:abl pede f1_1')
    save_latex_data(bas_pede, metric='f1_1', savefile='latex/bas_pede_f1_1.tex',
                    title='Group Correctness metric with $P=1$ for T-DANTE vs Baselines in all pedestrian datasets. \\textbf{*} shows that this result is significantly different than all the other values in the same column.',
                    label='tab:bas pede f1_1')
    save_latex_data(abl_pede, metric='f1_2/3', savefile='latex/abl_pede_f1_23.tex',
                    title='Group Correctness metric with $P=2/3$ for T-DANTE variations in all pedestrian datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 15 consecutive timeframes.',
                    label='tab:abl pede f1_2/3')
    save_latex_data(bas_pede, metric='f1_2/3', savefile='latex/bas_pede_f1_23.tex',
                    title='Group Correctness metric with $P=2/3$ for T-DANTE vs Baselines in all pedestrian datasets.',
                    label='tab:bas pede f1_2/3')
    save_latex_data(abl_pede, metric='f1_gmitre', savefile='latex/abl_pede_f1_gmitre.tex',
                    title='Group Mitre metric for T-DANTE variations in all pedestrian datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 15 consecutive timeframes.',
                    label='tab:abl pede f1_gmitre')
    save_latex_data(bas_pede, metric='f1_gmitre', savefile='latex/bas_pede_f1_gmitre.tex',
                    title='Group Mitre metric for T-DANTE vs Baselines in all pedestrian datasets.',
                    label='tab:bas pede f1_gmitre')

    save_latex_data(abl_sim, metric='f1_1', savefile='latex/abl_sim_f1_1.tex',
                    title='Group Correctness metric with $P=1$ for T-DANTE variations in all spring simulation datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 50 consecutive timeframes.',
                    label='tab:abl sim f1_1')
    save_latex_data(bas_sim, metric='f1_1', savefile='latex/bas_sim_f1_1.tex',
                    title='Group Correctness metric with $P=1$ for T-DANTE vs Baselines in all spring simulation datasets.',
                    label='tab:bas sim f1_1')
    save_latex_data(abl_sim, metric='f1_2/3', savefile='latex/abl_sim_f1_23.tex',
                    title='Group Correctness metric with $P=2/3$ for T-DANTE variations in all spring simulation datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 50 consecutive timeframes.',
                    label='tab:abl sim f1_2/3')
    save_latex_data(bas_sim, metric='f1_2/3', savefile='latex/bas_sim_f1_23.tex',
                    title='Group Correctness metric with $P=2/3$ for T-DANTE vs Baselines in all spring simulation datasets.',
                    label='tab:bas sim f1_2/3')
    save_latex_data(abl_sim, metric='f1_gmitre', savefile='latex/abl_sim_f1_gmitre.tex',
                    title='Group Mitre metric for T-DANTE variations in all spring simulation datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 50 consecutive timeframes.',
                    label='tab:abl sim f1_gmitre')
    save_latex_data(bas_sim, metric='f1_gmitre', savefile='latex/bas_sim_f1_gmitre.tex',
                    title='Group Mitre metric for T-DANTE vs Baselines in all spring simulation datasets.',
                    label='tab:bas sim f1_gmitre')
