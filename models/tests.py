import argparse

from scipy.stats import anderson, friedmanchisquare


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
    # Sample data (replace this with your actual data)
    # Each row represents a participant/subject, and each column is a different treatment/condition
    # data = np.array([
    #     [3, 4, 5],
    #     [2, 2, 3],
    #     [5, 6, 4],
    #     [4, 5, 6],
    # ])

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


# TODO implement collect_data function
def collect_data(friedman):
    if friedman:
        pass
    else:
        pass
    return []


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--friedman', action="store_true", default=True)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    data = collect_data(args.friedman)
    if args.friedman:
        friedman_test(data)
    else:
        anderson_darling_test(data)
