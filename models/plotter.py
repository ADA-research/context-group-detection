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


def modify_df(dataframe, name=None, no_context=False, model=None):
    details = dataframe['dataset'].str.split('_')
    details_size = len(details[0])
    dataframe['dataset'] = details.apply(lambda x: x[0])
    if details_size == 2:
        dataframe['name'] = name
    else:
        if details[0][1] == '1':
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


def modify_sim_df(dataframe, name=None, no_context=False, fix_datasets=False, model=None):
    if fix_datasets:
        dataframe['dataset'] = dataframe['dataset'].replace('8_3_2_2_nri', 'sim_1')
        dataframe['dataset'] = dataframe['dataset'].replace('9_3_2_2_nri', 'sim_2')
        dataframe['dataset'] = dataframe['dataset'].replace('9_3_2_3_nri', 'sim_3')
        dataframe['dataset'] = dataframe['dataset'].replace('10_3_2_2_nri', 'sim_4')
        dataframe['dataset'] = dataframe['dataset'].replace('10_3_2_3_nri', 'sim_5')
        dataframe['dataset'] = dataframe['dataset'].replace('10_3_2_4_nri', 'sim_6')
        dataframe['name'] = name
    else:
        details = dataframe['dataset'].str.split('_')
        dataframe['dataset'] = details.apply(lambda x: '_'.join(x[0:2]))
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
        for model in models:
            metric_data = []
            std_data = []
            for dataset in datasets:
                model_data = dataframe[(dataframe['name'] == model) & (dataframe['dataset'] == dataset)]
                metric_data.append(model_data[metric].values[0])
                std_data.append(model_data['{}_std'.format(metric)].values[0])
            # write model name
            file.write(
                f'\multicolumn{{1}}{{|c|}}{{\multirow{{2}}{{*}}{{\\begin{{tabular}}[c]{{@{{}}c@{{}}}}\n{model}\n\end{{tabular}}}}}}\n')
            metric_data = ' & '.join(map(str, metric_data))
            std_data = ' & $\pm$'.join(map(str, std_data))
            file.write(f' & {metric_data} \\\\\n')
            file.write(f'\multicolumn{{1}}{{|c|}}{{}} & $\pm${std_data} \\\\ \hline\n')
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

    # read data
    pede_wavenet_results = read_csv("./WavenetNRI/logs/nripedsu/pede_wavenet_results.csv")
    pede_nri_results = read_csv("./WavenetNRI/logs/nripedsu/pede_nri_results.csv")
    pede_gdgan_results = read_csv("./GDGAN/logs/nripedsu/pede_gdgan_results.csv")
    pede_nc_results = read_csv("./results/pede_nc_results.csv")
    pede_nc_gd_results = read_csv("./results/pede_nc_gd_results.csv")
    pede_dante_results = read_csv("./results/pede_dante_results.csv")
    pede_tdante_results = read_csv("./results/pede_tdante_results.csv")
    pede_tdante_gd_results = read_csv("./results/pede_tdante_gd_results.csv")
    sim_wavenet_results = read_csv("./WavenetNRI/logs/nrisu/sim_wavenet_results.csv")
    sim_nri_results = read_csv("./WavenetNRI/logs/nrisu/sim_nri_results.csv")
    sim_gdgan_results = read_csv("./GDGAN/logs/nrisu/sim_gdgan_results.csv")
    sim_nc_results = read_csv("./results/sim_nc_results.csv")
    sim_nc_gd_results = read_csv("./results/sim_nc_gd_results.csv")
    sim_tdante_results = read_csv("./results/sim_tdante_results.csv")
    sim_tdante_gd_results = read_csv("./results/sim_tdante_gd_results.csv")

    pede_wavenet_results = modify_df(pede_wavenet_results, name='WavenetNRI')
    pede_nri_results = modify_df(pede_nri_results, name='NRI')
    pede_gdgan_results = modify_df(pede_gdgan_results, name='GDGAN')
    pede_nc_results = modify_df(pede_nc_results, no_context=True)
    pede_nc_gd_results = modify_df(pede_nc_gd_results, no_context=True, model='gd')
    pede_dante_results = modify_df(pede_dante_results)
    pede_tdante_results = modify_df(pede_tdante_results)
    pede_tdante_gd_results = modify_df(pede_tdante_gd_results, model='gd')

    sim_wavenet_results = modify_sim_df(sim_wavenet_results, name='WavenetNRI', fix_datasets=True)
    sim_nri_results = modify_sim_df(sim_nri_results, name='NRI', fix_datasets=True)
    sim_gdgan_results = modify_sim_df(sim_gdgan_results, name='GDGAN', fix_datasets=True)
    sim_nc_results = modify_sim_df(sim_nc_results, no_context=True)
    sim_nc_gd_results = modify_sim_df(sim_nc_gd_results, no_context=True, model='gd')
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
                         sim_tdante_results])
    bas_sim = bas_sim[~bas_sim['name'].isin(['DANTE c4', 'T-DANTE c4'])]

    sns.set(style='whitegrid')
    plot_df(abl_pede, metric='f1_1', ylabel='F1',
            title='Group Correctness P=1 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/abl_pede_f1_1')
    save_latex_data(abl_pede, metric='f1_1', savefile='latex/abl_pede_f1_1.tex',
                    title='Group Correctness values for threshold value $1$ for T-DANTE variations in all pedestrian datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 15 consecutive timeframes.',
                    label='tab:abl pede f1_1')
    plot_df(bas_pede, metric='f1_1', ylabel='F1',
            title='Group Correctness P=1 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/bas_pede_f1_1')
    save_latex_data(bas_pede, metric='f1_1', savefile='latex/bas_pede_f1_1.tex',
                    title='Group Correctness values for threshold value $1$ for T-DANTE vs Baselines in all pedestrian datasets.',
                    label='tab:bas pede f1_1')
    plot_df(abl_pede, metric='f1_2/3', ylabel='F1',
            title='Group Correctness P=2/3 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/abl_pede_f1_23')
    save_latex_data(abl_pede, metric='f1_2/3', savefile='latex/abl_pede_f1_23.tex',
                    title='Group Correctness values for threshold value $2/3$ for T-DANTE variations in all pedestrian datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 15 consecutive timeframes.',
                    label='tab:abl pede f1_2/3')
    plot_df(bas_pede, metric='f1_2/3', ylabel='F1',
            title='Group Correctness P=2/3 F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/bas_pede_f1_23')
    save_latex_data(bas_pede, metric='f1_2/3', savefile='latex/bas_pede_f1_23.tex',
                    title='Group Correctness values for threshold value $2/3$ for T-DANTE vs Baselines in all pedestrian datasets.',
                    label='tab:bas pede f1_2/3')
    plot_df(abl_pede, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/abl_pede_f1_gmitre')
    save_latex_data(abl_pede, metric='f1_gmitre', savefile='latex/abl_pede_f1_gmitre.tex',
                    title='Group Mitre values for T-DANTE variations in all pedestrian datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 15 consecutive timeframes.',
                    label='tab:abl pede f1_gmitre')
    plot_df(bas_pede, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Pedestrian Dataset',
            savefile='pngs/bas_pede_f1_gmitre')
    save_latex_data(bas_pede, metric='f1_gmitre', savefile='latex/bas_pede_f1_gmitre.tex',
                    title='Group Mitre values  for T-DANTE vs Baselines in all pedestrian datasets.',
                    label='tab:bas pede f1_gmitre')

    plot_df(abl_sim, metric='f1_1', ylabel='F1',
            title='Group Correctness P=1 F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/abl_sim_f1_1')
    save_latex_data(abl_sim, metric='f1_1', savefile='latex/abl_sim_f1_1.tex',
                    title='Group Correctness values for threshold value $1$ for T-DANTE variations in all spring simulation datasets. Context sizes of $4$ and $8$ agents and scene size of 50 consecutive timeframes.',
                    label='tab:abl sim f1_1')
    plot_df(bas_sim, metric='f1_1', ylabel='F1',
            title='Group Correctness P=1 F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/bas_sim_f1_1')
    save_latex_data(bas_sim, metric='f1_1', savefile='latex/bas_sim_f1_1.tex',
                    title='Group Correctness values for threshold value $1$ for T-DANTE vs Baselines in all spring simulation datasets.',
                    label='tab:bas sim f1_1')
    plot_df(abl_sim, metric='f1_2/3', ylabel='F1',
            title='Group Correctness P=2/3 F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/abl_sim_f1_23')
    save_latex_data(abl_sim, metric='f1_2/3', savefile='latex/abl_sim_f1_23.tex',
                    title='Group Correctness values for threshold value $2/3$ for T-DANTE variations in all spring simulation datasets. Context sizes of $4$ and $8$ agents and scene size of 50 consecutive timeframes.',
                    label='tab:abl sim f1_2/3')
    plot_df(bas_sim, metric='f1_2/3', ylabel='F1',
            title='Group Correctness P=2/3 F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/bas_sim_f1_23')
    save_latex_data(bas_sim, metric='f1_2/3', savefile='latex/bas_sim_f1_23.tex',
                    title='Group Correctness values for threshold value $2/3$ for T-DANTE vs Baselines in all spring simulation datasets.',
                    label='tab:bas sim f1_2/3')
    plot_df(abl_sim, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/abl_sim_f1_gmitre')
    save_latex_data(abl_sim, metric='f1_gmitre', savefile='latex/abl_sim_f1_gmitre.tex',
                    title='Group Mitre values for T-DANTE variations in all spring simulation datasets. Context sizes of $0$, $4$ and $8$ agents and scene size of 50 consecutive timeframes.',
                    label='tab:abl sim f1_gmitre')
    plot_df(bas_sim, metric='f1_gmitre', ylabel='F1',
            title='Group Mitre F1 values\nfor each Model and Simulation Dataset',
            savefile='pngs/bas_sim_f1_gmitre')
    save_latex_data(bas_sim, metric='f1_gmitre', savefile='latex/bas_sim_f1_gmitre.tex',
                    title='Group Mitre values for T-DANTE vs Baselines in all spring simulation datasets.',
                    label='tab:bas sim f1_gmitre')
