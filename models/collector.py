import argparse
import os


def read_results(folder_path):
    file_path = folder_path + '/results.txt'
    with open(file_path, "r") as file:
        lines = [line.lstrip().rstrip().split() for line in file.readlines()]

    results = {
        'best_val': {
            'value': float(lines[0][1]),
            'epoch': int(lines[1][1])
        },
        'best_val_f1_1': {
            'value': float(lines[2][1]),
            'epoch': int(lines[3][1]),
            'test_f1s': [float(i) for i in lines[4][1:]],
            'precision': [float(i) for i in lines[5][1:]],
            'recall': [float(i) for i in lines[6][1:]]
        },
        'best_val_f1_2/3': {
            'value': float(lines[7][1]),
            'epoch': int(lines[8][1]),
            'test_f1s': [float(i) for i in lines[9][1:]],
            'precision': [float(i) for i in lines[10][1:]],
            'recall': [float(i) for i in lines[11][1:]]
        },
        'best_val_f1_gmitre': {
            'value': float(lines[12][1]),
            'epoch': int(lines[13][1]),
            'test_f1s': [float(i) for i in lines[14][1:]],
            'precision': [float(i) for i in lines[15][1:]],
            'recall': [float(i) for i in lines[16][1:]]
        }
    }

    return results


def get_averages(results):
    mse_vals = [res['best_val']['value'] for res in results]
    f1_1_vals = [res['best_val_f1_1']['value'] for res in results]
    f1_2_vals = [res['best_val_f1_2/3']['value'] for res in results]
    f1_gmitre_vals = [res['best_val_f1_gmitre']['value'] for res in results]

    return {
        'mse': sum(mse_vals) / len(mse_vals),
        'f1_1': sum(f1_1_vals) / len(f1_1_vals),
        'f1_2/3': sum(f1_2_vals) / len(f1_2_vals),
        'f1_gmitre': sum(f1_gmitre_vals) / len(f1_gmitre_vals)
    }


def collect_results(results_path, dir_name):
    results = []
    for fold in os.listdir(results_path):
        fold_path = results_path + '/' + fold
        if os.path.isdir(fold_path):
            for folder in os.listdir(fold_path):
                if folder == dir_name:
                    folder_path = fold_path + '/' + folder
                    results.append(read_results(folder_path))

    return get_averages(results)


def write_results(results, file_path, dir_name):
    file_path = file_path + '/{}_results.txt'.format(dir_name)
    with open(file_path, "w") as file:
        file.write('{:<10s} {:<10s} {:<10s} {:<10s}\n'.format('mse', '1 f1', '2/3 f1', 'gmitre f1'))
        file.write('{:<10.7f} {:<10.7f} {:<10.7f} {:<10.7f}\n'.format(
                results['mse'], results['f1_1'], results['f1_2/3'], results['f1_gmitre']))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="eth")
    parser.add_argument('-a', '--agents', type=int, default=10)
    parser.add_argument('-cf', '--frames', type=int, default=10)
    parser.add_argument('--dir_name', type=str, default="e150")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    results_path = './results/{}_{}_{}'.format(args.dataset, args.frames, args.agents)

    results = collect_results(results_path, args.dir_name)
    write_results(results, results_path, args.dir_name)
