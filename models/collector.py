import argparse
import os

import numpy as np
import pandas as pd


def read_results(folder_path):
    file_path = folder_path + '/results.txt'
    with open(file_path, "r") as file:
        lines = [line.lstrip().rstrip().split() for line in file.readlines()]

    # print(folder_path)

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


def collect_results(results_path, dir_name, average=True):
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
    if not results:
        return {}
    if average:
        return get_averages(results)
    else:
        return results


def collect_sim_results(results_path, dir_name, average=True):
    results = []
    for seed in os.listdir(results_path):
        seed_path = results_path + '/' + seed
        if os.path.isdir(seed_path):
            start = seed.startswith(dir_name)
            rest_digit = seed.replace(dir_name, '')[1:].isdigit()
            if start and rest_digit:
                results.append(read_results(seed_path))
    if not results:
        return {}
    if average:
        return get_averages(results)
    else:
        return results


def collect_nri_results(results_path, average=True):
    results = []
    for fold in os.listdir(results_path):
        fold_path = results_path + '/' + fold
        if os.path.isdir(fold_path):
            for folder in os.listdir(fold_path):
                folder_path = fold_path + '/' + folder
                results.append(read_nri_results(folder_path))

    if average:
        return get_nri_averages(results)
    else:
        return results


def collect_nri_sim_results(results_path, average=True):
    results = []
    for seed in os.listdir(results_path):
        seed_path = results_path + '/' + seed
        if os.path.isdir(seed_path):
            results.append(read_nri_results(seed_path))

    if average:
        return get_nri_averages(results)
    else:
        return results


def write_results(results, file_path, dir_name):
    file_name = file_path + '/{}.csv'.format(dir_name)

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


def write_final_results(results, file_path, name):
    file_name = file_path + '/{}.csv'.format(name)

    dfs = []
    for info, result in results:
        dataset, frames, agents = info
        df = pd.DataFrame.from_dict(result, orient='index').transpose()
        df['dataset'] = '{}_{}_{}'.format(dataset, frames, agents)
        for col in 'f1_1', 'f1_2/3', 'f1_gmitre':
            df[col] = df[col].round(4)
        df = df[['dataset', 'f1_1', 'f1_2/3', 'f1_gmitre']]
        dfs.append(df)
    final_df = pd.concat(dfs)
    final_df.to_csv(file_name, index=False)


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


def write_final_nri_results(results, file_path, name):
    file_name = file_path + '/{}.csv'.format(name)

    dfs = []
    for dataset, result in results:
        df = pd.DataFrame.from_dict(result, orient='index').transpose()
        df['dataset'] = '{}_nri'.format(dataset)
        for col in 'f1_1', 'f1_2/3', 'f1_gmitre':
            df[col] = df[col].round(4)
        df = df[['dataset', 'f1_1', 'f1_2/3', 'f1_gmitre']]
        dfs.append(df)
    final_df = pd.concat(dfs)
    final_df.to_csv(file_name, index=False)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sim', action="store_true", default=False)
    parser.add_argument('--no_context', action="store_true", default=False)
    parser.add_argument('--model', type=str, default="nri")
    parser.add_argument('--dataset', type=str, default="sim_1")
    parser.add_argument('--single_dataset', action="store_true", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    nri_datasets = ['8_3_2_2', '9_3_2_2', '9_3_2_3', '10_3_2_2', '10_3_2_3', '10_3_2_4']
    sim_datasets = ['sim_1', 'sim_2', 'sim_3', 'sim_4', 'sim_5', 'sim_6']
    pede_datasets = ['eth', 'hotel', 'zara01', 'zara02', 'students03']
    if args.single_dataset:
        if args.model == 'wavenet':
            if args.sim:
                results_path = './WavenetNRI/logs/nrisu/wavenetsym_{}'.format(args.dataset, 15)
                results = (args.dataset, collect_nri_sim_results(results_path))
                write_final_nri_results(results, './WavenetNRI/logs/nrisu', 'sim_wavenet_results')
            else:
                results_path = './WavenetNRI/logs/nripedsu/wavenetsym_{}_shifted_{}'.format(args.dataset, 15)
                results = (args.dataset, collect_nri_results(results_path))
                write_final_nri_results(results, './WavenetNRI/logs/nripedsu', 'pede_wavenet_results')
        elif args.model == 'nri':
            if args.sim:
                results_path = './WavenetNRI/logs/nrisu/cnn_{}'.format(args.dataset, 15)
                results = (args.dataset, collect_nri_sim_results(results_path))
                write_final_nri_results(results, './WavenetNRI/logs/nrisu', 'sim_nri_results')
            else:
                results_path = './WavenetNRI/logs/nripedsu/cnn_{}_shifted_{}'.format(args.dataset, 15)
                results = (args.dataset, collect_nri_results(results_path))
                write_final_nri_results(results, './WavenetNRI/logs/nripedsu', 'pede_nri_results')
        elif args.model == 'gd':
            results_path = './results/{}_shifted_{}_{}_gd'.format(args.dataset, args.frames, args.agents)
            if args.sim:
                results = collect_sim_results(results_path, args.dir_name)
            else:
                results = collect_results(results_path, args.dir_name)
        else:
            results_path = './results/{}_shifted_{}_{}'.format(args.dataset, args.frames, args.agents)
            if args.sim:
                results = collect_sim_results(results_path, args.dir_name)
            else:
                results = collect_results(results_path, args.dir_name)
        exit()

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
    sim_nc_results = []
    sim_nc_gd_results = []
    sim_tdante_results = []
    sim_tdante_gd_results = []

    for dataset in pede_datasets:
        dataset_results = []
        results_path = './WavenetNRI/logs/nripedsu/wavenetsym_{}_shifted_{}'.format(dataset, 15)
        pede_wavenet_results.append((dataset, collect_nri_results(results_path)))
        results_path = './WavenetNRI/logs/nripedsu/cnn_{}_shifted_{}'.format(dataset, 15)
        pede_nri_results.append((dataset, collect_nri_results(results_path)))
        results_path = './GDGAN/logs/nripedsu/{}_shifted_{}'.format(dataset, 15)
        pede_gdgan_results.append((dataset, collect_nri_results(results_path)))
        for agents_num in [6, 10]:
            pede_dante_results.append((
                (dataset, 1, agents_num),
                collect_results('./results/{}_shifted_1_{}'.format(dataset, agents_num), 'e150')))
            for frames_num in [15]:
                results_path = './results/{}_shifted_{}_{}'.format(dataset, frames_num, agents_num)
                results_path_gd = './results/{}_shifted_{}_{}_gd'.format(dataset, frames_num, agents_num)
                pede_tdante_results.append(
                    ((dataset, frames_num, agents_num), collect_results(results_path, 'e150')))
                pede_tdante_gd_results.append(
                    ((dataset, frames_num, agents_num), collect_results(results_path_gd, 'e150')))
                if agents_num == 6:
                    pede_nc_results.append(
                        ((dataset, frames_num, agents_num), collect_results(results_path, 'e150_nc')))
                    pede_nc_gd_results.append(
                        ((dataset, frames_num, agents_num), collect_results(results_path_gd, 'e150_nc')))
    write_final_results(pede_dante_results, './results', 'pede_dante_results')
    write_final_results(pede_tdante_results, './results', 'pede_tdante_results')
    write_final_results(pede_tdante_gd_results, './results', 'pede_tdante_gd_results')
    write_final_results(pede_nc_results, './results', 'pede_nc_results')
    write_final_results(pede_nc_gd_results, './results', 'pede_nc_gd_results')
    write_final_nri_results(pede_wavenet_results, './WavenetNRI/logs/nripedsu', 'pede_wavenet_results')
    write_final_nri_results(pede_nri_results, './WavenetNRI/logs/nripedsu', 'pede_nri_results')
    write_final_nri_results(pede_gdgan_results, './GDGAN/logs/nripedsu', 'pede_gdgan_results')

    for dataset in sim_datasets:
        for agents_num in [6, 10]:
            for frames_num in [49]:
                results_path = './results/{}_shifted_{}_{}'.format(dataset, frames_num, agents_num)
                results_path_gd = './results/{}_shifted_{}_{}_gd'.format(dataset, frames_num, agents_num)
                sim_tdante_results.append(
                    ((dataset, frames_num, agents_num), collect_sim_results(results_path, 'e50')))
                sim_tdante_gd_results.append(
                    ((dataset, frames_num, agents_num), collect_sim_results(results_path_gd, 'e50')))
                if agents_num == 6:
                    sim_nc_results.append(
                        ((dataset, frames_num, agents_num), collect_sim_results(results_path, 'e50_nc')))
                    sim_nc_gd_results.append(
                        ((dataset, frames_num, agents_num), collect_sim_results(results_path_gd, 'e50_nc')))

    write_final_results(sim_nc_results, './results', 'sim_nc_results')
    write_final_results(sim_nc_gd_results, './results', 'sim_nc_gd_results')
    write_final_results(sim_tdante_results, './results', 'sim_tdante_results')
    write_final_results(sim_tdante_gd_results, './results', 'sim_tdante_gd_results')

    for dataset in nri_datasets:
        results_path = './WavenetNRI/logs/nrisu/su_wavenetsym_{}'.format(dataset)
        sim_wavenet_results.append((dataset, collect_nri_sim_results(results_path)))
        results_path = './WavenetNRI/logs/nrisu/su_cnn_{}'.format(dataset)
        sim_nri_results.append((dataset, collect_nri_sim_results(results_path)))
        results_path = './GDGAN/logs/nrisu/{}'.format(dataset)
        sim_gdgan_results.append((dataset, collect_nri_sim_results(results_path)))
    write_final_nri_results(sim_wavenet_results, './WavenetNRI/logs/nrisu', 'sim_wavenet_results')
    write_final_nri_results(sim_nri_results, './WavenetNRI/logs/nrisu', 'sim_nri_results')
    write_final_nri_results(sim_gdgan_results, './GDGAN/logs/nrisu', 'sim_gdgan_results')
