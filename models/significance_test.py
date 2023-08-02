import argparse
import os

import numpy as np
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import anderson, friedmanchisquare

from models.collector import collect_nri_results, collect_results, collect_sim_results, collect_nri_sim_results


def anderson_darling_test(data):
    # Sample data (replace this with your actual data)
    # data = np.array([1.2, 1.8, 2.1, 2.5, 3.0, 3.2, 3.5, 4.0, 4.2, 4.8])

    # Perform Anderson-Darling test
    result = anderson(data, dist='norm')

    print("Anderson-Darling Test Statistic:", result.statistic)
    print("Critical Values:", result.critical_values)
    print("Significance Levels:", result.significance_level)

    # Interpret the result
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print(f"At {sl * 100:.1f}% significance level, the data looks Gaussian (fail to reject H0).")
        else:
            print(f"At {sl * 100:.1f}% significance level, the data does not look Gaussian (reject H0).")


def friedman_test(data, models, dataset, metric):
    # Perform Friedman test
    statistic, p_value = friedmanchisquare(*data.T)

    print("Friedman Test Statistic:", statistic)
    print("P-value:", p_value)

    # Interpret the result
    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference among the treatments (reject H0).")
        # Perform Nemenyi post-hoc test
        nemenyi_result = posthoc_nemenyi_friedman(data.T)
        nemenyi_result.columns = models
        nemenyi_result['model'] = models
        nemenyi_result.set_index('model', inplace=True)
        print("Critical Difference (Nemenyi):", nemenyi_result)
        nemenyi_result.to_csv('nemenyi_result_{}_{}.csv'.format(dataset, metric))
    else:
        print("There is no significant difference among the treatments (fail to reject H0).")


def collect_data(results_path, nri_results_path, dataset, dir_name, metric, sim):
    # collect dante + model results
    models = []
    results = []
    for item in os.listdir(results_path):
        item_path = '{}/{}'.format(results_path, item)
        if os.path.isdir(item_path) and item.startswith(dataset):
            for name in [dir_name, '{}_nc'.format(dir_name)]:
                if sim:
                    result = collect_sim_results(item_path, name, average=False)
                else:
                    result = collect_results(item_path, name, average=False)
                if result == {}:
                    continue
                if metric == 'gmitre':
                    result = [sample['best_val_f1_gmitre']['value'] for sample in result]
                elif metric == 't1':
                    result = [sample['best_val_f1_1']['value'] for sample in result]
                elif metric == 't2/3':
                    result = [sample['best_val_f1_2/3']['value'] for sample in result]
                context = '_nc' if '_nc' in name else ''
                models.append('{}{}'.format(item.replace(dataset + '_', ''), context))
                results.append(result)
    if args.sim:
        if dataset == 'sim_1_shifted':
            dataset = '10_3_2_3'
        elif dataset == 'sim_2_shifted':
            dataset = '10_3_2_4'
        elif dataset == 'sim_3_shifted':
            dataset = '10_3_2_2'
        elif dataset == 'sim_4_shifted':
            dataset = '9_3_2_2'
        elif dataset == 'sim_5_shifted':
            dataset = '9_3_2_3'
        elif dataset == 'sim_6_shifted':
            dataset = '8_3_2_2'
    # collect nri results
    for item in os.listdir(nri_results_path):
        item_path = '{}/{}'.format(nri_results_path, item)
        if os.path.isdir(item_path) and item.startswith('wavenetsym_{}'.format(dataset)):
            if sim:
                result = collect_nri_sim_results(item_path, average=False)
            else:
                result = collect_nri_results(item_path, average=False)

            if metric == 'gmitre':
                result = [sample['f1_gmitre']['f1'] for sample in result]
            elif metric == 't1':
                result = [sample['f1_one']['f1'] for sample in result]
            elif metric == 't2/3':
                result = [sample['f1_two_thirds']['f1'] for sample in result]
            models.append('nri')
            results.append(result)
            break
    return models, results


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='sim_1_shifted')
    parser.add_argument('--metric', type=str, default='t2/3')
    parser.add_argument('--sim', action="store_true", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    results_path = './results'
    if args.sim:
        nri_results_path = './WavenetNRI/logs/nrisu'
        dir_name = 'e50'
    else:
        nri_results_path = './WavenetNRI/logs/nripedsu'
        dir_name = 'e150'
    models, data = collect_data(results_path, nri_results_path, args.dataset, dir_name, args.metric, args.sim)
    friedman_test(np.asarray(data), models, args.dataset, args.metric)
