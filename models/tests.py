import argparse
import os
import numpy as np
from scipy.stats import anderson, friedmanchisquare

from models.collector import collect_nri_results, collect_results


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


def friedman_test(data):
    # Perform Friedman test
    statistic, p_value = friedmanchisquare(*data.T)

    print("Friedman Test Statistic:", statistic)
    print("P-value:", p_value)

    # Interpret the result
    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference among the treatments (reject H0).")
    else:
        print("There is no significant difference among the treatments (fail to reject H0).")


def collect_data(results_path, nri_results_path, dataset, dir_name, metric):
    # collect dante + model results
    models = []
    results = []
    for item in os.listdir(results_path):
        item_path = '{}/{}'.format(results_path, item)
        if os.path.isdir(item_path) and item.startswith(dataset):
            for name in [dir_name, '{}_nc'.format(dir_name)]:
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
                models.append('{}{}'.format(item.replace(dataset, ''), context))
                results.append(result)
    # collect nri results
    for item in os.listdir(nri_results_path):
        item_path = '{}/{}'.format(nri_results_path, item)
        if os.path.isdir(item_path) and item.startswith('wavenetsym_{}'.format(dataset)):
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

    parser.add_argument('--dataset', type=str, default='eth_shifted')
    parser.add_argument('--dir_name', type=str, default='e150')
    parser.add_argument('--metric', type=str, default='gmitre')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    results_path = './results'
    nri_results_path = './WavenetNRI/logs/nripedsu'
    models, data = collect_data(results_path, nri_results_path, args.dataset, args.dir_name, args.metric)
    friedman_test(np.asarray(data))
