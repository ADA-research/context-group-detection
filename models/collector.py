import argparse
import os

import numpy as np
import pandas as pd


def read_results(folder_path):
    file_path = folder_path + '/results.txt'
    with open(file_path, "r") as file:
        lines = [line.lstrip().rstrip().split() for line in file.readlines()]

    results = {
        'best_val': {
            'value': float(lines[0][1]),
            'epoch': int(lines[1][1])
        },
        'best_f1_avg': {
            'value': float(lines[2][1]),
            'epoch': int(lines[3][1]),
            'test_f1s': [float(i) for i in lines[4][1:]],
            'precision': [float(i) for i in lines[5][1:]],
            'recall': [float(i) for i in lines[6][1:]]
        },
        'best_val_f1_1': {
            'value': float(lines[7][1]),
            'epoch': int(lines[8][1]),
            'test_f1s': [float(i) for i in lines[9][1:]],
            'precision': [float(i) for i in lines[10][1:]],
            'recall': [float(i) for i in lines[11][1:]]
        },
        'best_val_f1_2/3': {
            'value': float(lines[12][1]),
            'epoch': int(lines[13][1]),
            'test_f1s': [float(i) for i in lines[14][1:]],
            'precision': [float(i) for i in lines[15][1:]],
            'recall': [float(i) for i in lines[16][1:]]
        },
        'best_val_f1_gmitre': {
            'value': float(lines[17][1]),
            'epoch': int(lines[18][1]),
            'test_f1s': [float(i) for i in lines[19][1:]],
            'precision': [float(i) for i in lines[20][1:]],
            'recall': [float(i) for i in lines[21][1:]]
        },
        "train loss": [float(line[0]) for line in lines[23:]],
        "val loss": [float(line[1]) for line in lines[23:]],
        "train mse": [float(line[2]) for line in lines[23:]],
        "val mse": [float(line[3]) for line in lines[23:]],
        "val 1 f1": [float(line[4]) for line in lines[23:]],
        "val 2/3 f1": [float(line[5]) for line in lines[23:]],
        "val gmitre f1": [float(line[6]) for line in lines[23:]],
        "f1 avg": [float(line[7]) for line in lines[23:]],
    }

    return results


def read_nri_results(folder_path):
    file_path = folder_path + '/log.txt'
    with open(file_path, "r") as file:
        lines = [line.lstrip().rstrip().split() for line in file.readlines()]

    results = {
        'f1_one': {
            'accuracy': float(lines[-2][7]),
            'recall': float(lines[-2][3]),
            'f1': float(lines[-2][11]),
        },
        'f1_two_thirds': {
            'accuracy': float(lines[-1][7]),
            'recall': float(lines[-1][3]),
            'f1': float(lines[-1][11]),
        },
        'f1_gmitre': {
            'accuracy': float(lines[-3][5]),
            'recall': float(lines[-3][2]),
            'f1': float(lines[-3][8]),
        }
    }

    return results


def get_averages(results):
    mse_vals = [res['best_val']['value'] for res in results]
    f1_1_vals = [res['best_val_f1_1']['value'] for res in results]
    f1_2_vals = [res['best_val_f1_2/3']['value'] for res in results]
    f1_gmitre_vals = [res['best_val_f1_gmitre']['value'] for res in results]

    return {
        'mse': (np.mean(mse_vals), np.std(mse_vals)),
        'f1_1': (np.mean(f1_1_vals), np.std(f1_1_vals)),
        'f1_2/3': (np.mean(f1_2_vals), np.std(f1_2_vals)),
        'f1_gmitre': (np.mean(f1_gmitre_vals), np.std(f1_gmitre_vals))
    }


def get_nri_averages(results):
    f1_one = [res['f1_one']['f1'] for res in results]
    f1_two_thirds = [res['f1_two_thirds']['f1'] for res in results]
    f1_gmitre = [res['f1_gmitre']['f1'] for res in results]

    return {
        'f1_1': (np.mean(f1_one), np.std(f1_one)),
        'f1_2/3': (np.mean(f1_two_thirds), np.std(f1_two_thirds)),
        'f1_gmitre': (np.mean(f1_gmitre), np.std(f1_gmitre))
    }


def collect_results(results_path, dir_name):
    results = []
    for fold in os.listdir(results_path):
        fold_path = results_path + '/' + fold
        if os.path.isdir(fold_path):
            for folder in os.listdir(fold_path):
                start = folder.startswith(dir_name)
                rest_digit = folder.replace(dir_name, '')[1:].isdigit()
                if start and rest_digit:
                    folder_path = fold_path + '/' + folder
                    results.append(read_results(folder_path))

    return get_averages(results)


def collect_nri_results(results_path):
    results = []
    for fold in os.listdir(results_path):
        fold_path = results_path + '/' + fold
        if os.path.isdir(fold_path):
            for folder in os.listdir(fold_path):
                folder_path = fold_path + '/' + folder
                results.append(read_nri_results(folder_path))

    return get_nri_averages(results)


def write_results(results, file_path, dir_name):
    file_name = file_path + '/{}_results.csv'.format(dir_name)

    df = pd.DataFrame.from_dict(results, orient='index').transpose()
    df['metric'] = ['mean', 'std']
    df = df.set_index('metric')
    df.to_csv(file_name)

    file_path = file_path + '/{}_results.txt'.format(dir_name)
    with open(file_path, "w") as file:
        file.write('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s}\n'.format('', 'mse', '1 f1', '2/3 f1', 'gmitre f1'))
        file.write('{:<10s} {:<10.7f} {:<10.7f} {:<10.7f} {:<10.7f}\n'.format(
            'mean', results['mse'][0], results['f1_1'][0], results['f1_2/3'][0], results['f1_gmitre'][0]))
        file.write('{:<10s} {:<10.7f} {:<10.7f} {:<10.7f} {:<10.7f}\n'.format(
            'std', results['mse'][1], results['f1_1'][1], results['f1_2/3'][1], results['f1_gmitre'][1]))


def write_nri_results(results, file_path):
    file_name = file_path + '/results.csv'

    df = pd.DataFrame.from_dict(results, orient='index').transpose()
    df['metric'] = ['mean', 'std']
    df = df.set_index('metric')
    df.to_csv(file_name)

    file_path = file_path + '/results.txt'
    with open(file_path, "w") as file:
        file.write('{:<10s} {:<10s} {:<10s} {:<10s}\n'.format('', 'accuracy', 'recall', 'f1'))
        file.write('{:<10s} {:<10.7f} {:<10.7f} {:<10.7f}\n'.format(
            'mean', results['f1_1'][0], results['f1_2/3'][0], results['f1_gmitre'][0]))
        file.write('{:<10s} {:<10.7f} {:<10.7f} {:<10.7f}\n'.format(
            'std', results['f1_1'][1], results['f1_2/3'][1], results['f1_gmitre'][1]))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="eth_shifted")
    parser.add_argument('-a', '--agents', type=int, default=10)
    parser.add_argument('-cf', '--frames', type=int, default=15)
    parser.add_argument('--dir_name', type=str, default="e150")
    parser.add_argument('--nri', action="store_true", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.nri:
        results_path = './WavenetNRI/logs/nripedsu/wavenetsym_{}_{}'.format(args.dataset, args.frames)
        results = collect_nri_results(results_path)
        write_nri_results(results, results_path)
    else:
        results_path = './results/{}_{}_{}'.format(args.dataset, args.frames, args.agents)
        results = collect_results(results_path, args.dir_name)
        write_results(results, results_path, args.dir_name)
