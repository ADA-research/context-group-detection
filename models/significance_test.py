import argparse

import numpy as np
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import friedmanchisquare

from models.collector import collect_nri_results, collect_results, collect_sim_results, collect_nri_sim_results


# def anderson_darling_test(data):
#     # Sample data (replace this with your actual data)
#     # data = np.array([1.2, 1.8, 2.1, 2.5, 3.0, 3.2, 3.5, 4.0, 4.2, 4.8])
#
#     # Perform Anderson-Darling test
#     result = anderson(data, dist='norm')
#
#     print("Anderson-Darling Test Statistic:", result.statistic)
#     print("Critical Values:", result.critical_values)
#     print("Significance Levels:", result.significance_level)
#
#     # Interpret the result
#     for i in range(len(result.critical_values)):
#         sl, cv = result.significance_level[i], result.critical_values[i]
#         if result.statistic < cv:
#             print(f"At {sl * 100:.1f}% significance level, the data looks Gaussian (fail to reject H0).")
#         else:
#             print(f"At {sl * 100:.1f}% significance level, the data does not look Gaussian (reject H0).")


def friedman_test(data, models, dataset, metric, preffix):
    # Perform Friedman test
    statistic, p_value = friedmanchisquare(*data.T)

    # Interpret the result
    alpha = 0.05
    if p_value < alpha:
        print("\t\tSignificant difference")
        # Perform Nemenyi post-hoc test
        nemenyi_result = posthoc_nemenyi_friedman(data.T)
        nemenyi_result.columns = models
        nemenyi_result['model'] = models
        nemenyi_result.set_index('model', inplace=True)
        # print("Critical Difference (Nemenyi):", nemenyi_result)
        nemenyi_result.to_csv('./csvs/{}_{}_{}.csv'.format(preffix, dataset, metric))
    else:
        print("\t\tNo significant difference")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='sim_1_shifted')
    parser.add_argument('--metric', type=str, default='t2/3')
    parser.add_argument('--sim', action="store_true", default=True)

    return parser.parse_args()


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

    counter = 1
    names = ['T-DANTE nc', 'T-DANTE context', 'T-DANTE GD nc', 'T-DANTE GD context']
    results = [pede_nc_results, pede_tdante_results, pede_nc_gd_results, pede_tdante_gd_results]
    models = []
    for dataset in pede_datasets:
        for metric in ['best_val_f1_1', 'best_val_f1_2/3', 'best_val_f1_gmitre']:
            data = []
            names = ['T-DANTE nc', 'T-DANTE context', 'T-DANTE GD nc', 'T-DANTE GD context']
            for models_results, model_name in zip(results, names):
                for model_results in models_results:
                    if model_results[0][0] == dataset:
                        data.append([sample[metric]['value'] for sample in model_results[1]])
                        if counter == 1:
                            if 'nc' in model_name:
                                models.append(model_name)
                            else:
                                models.append('{} {}'.format(model_name, int(model_results[0][2]) - 2))
            if counter == 1:
                print('model names: {}'.format(models))
            print('\tDataset: {}, metric: {}'.format(dataset, metric))
            friedman_test(np.asarray(data), models, dataset, metric.replace('/', ''), preffix='abl_pede')
            counter += 1

    results = [sim_nc_results, sim_tdante_results, sim_nc_gd_results, sim_tdante_gd_results]
    for dataset in sim_datasets:
        for metric in ['best_val_f1_1', 'best_val_f1_2/3', 'best_val_f1_gmitre']:
            data = []
            for models_results, model_name in zip(results, names):
                for model_results in models_results:
                    if model_results[0][0] == dataset:
                        data.append([sample[metric]['value'] for sample in model_results[1]])
            print('\tDataset: {}, metric: {}'.format(dataset, metric))
            friedman_test(np.asarray(data), models, dataset, metric.replace('/', ''), preffix='abl_sim')

    names = ['DANTE', 'NRI', 'GDGAN', 'WavenetNRI', 'T-DANTE']
    results = [pede_dante_results, pede_nri_results, pede_gdgan_results, pede_wavenet_results, pede_tdante_results]
    print('Model names: {}'.format(names))
    for dataset in pede_datasets:
        for metric in ['best_val_f1_1', 'best_val_f1_2/3', 'best_val_f1_gmitre']:
            data = []
            for models_results, model_name in zip(results, names):
                for model_results in models_results:
                    if 'DANTE' in model_name:
                        if model_results[0][0] == dataset:
                            if model_results[0][2] == 10:
                                data.append([sample[metric]['value'] for sample in model_results[1]])
                    else:
                        if model_results[0] == dataset:
                            if metric == 'best_val_f1_1':
                                local_metric = 'f1_one'
                            elif metric == 'best_val_f1_2/3':
                                local_metric = 'f1_gmitre'
                            elif metric == 'best_val_f1_gmitre':
                                local_metric = 'f1_gmitre'
                            data.append(
                                [sample[local_metric]['f1'] for sample in model_results[1]])
            print('\tDataset: {}, metric: {}'.format(dataset, metric))
            friedman_test(np.asarray(data), names, dataset, metric.replace('/', ''), preffix='bas_pede')

    results = [sim_nc_results, sim_tdante_results, sim_nc_gd_results, sim_tdante_gd_results]
    for dataset in sim_datasets:
        for metric in ['best_val_f1_1', 'best_val_f1_2/3', 'best_val_f1_gmitre']:
            data = []
            for models_results, model_name in zip(results, names):
                for model_results in models_results:
                    if model_results[0][0] == dataset:
                        data.append([sample[metric]['value'] for sample in model_results[1]])
            print('\tDataset: {}, metric: {}'.format(dataset, metric))
            friedman_test(np.asarray(data), names, dataset, metric.replace('/', ''), preffix='bas_sim')
