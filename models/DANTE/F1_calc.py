from dominant_sets import *


# group_thres = T in most papers. The threshold for a correctly detected group
# affinities = rows of affinity predictions
# times = corresponding times
# Groups_at_time = dictionary from time to groups
# Positions = matrix of features at times. Form of row: time, ID001, x, y, ....
# ds_precision_thres = threshold for ending iterative dominant sets matrix algorithm. standard value is 1e-5
# n_features = number of features per person in Positions matrix, including name. eg (ID001, x, y, theta) is 4 features
# non_resuable = flag to set. if True, can't get multiple true groups with the same guess if the accuracy meets the threshold 
# 					for multiple GT groups with the same guess
# dominant_sets = flag to set. if True, uses dominant sets algorithm. if False, uses a naive grouping algorithm
def F1_calc(group_thres, affinities, times, Groups_at_time, Positions, n_people, n_features, non_reusable=False,
            dominant_sets=True):
    T = group_thres
    avg_results = np.array([0.0, 0.0])

    # this assumes affinities and times are the same length
    done = False
    prev_time_arr = [-1, -1]
    start_idx = 0
    num_times = 0
    while not done:
        num_times += 1
        looking = True
        end_idx = start_idx
        prev_time_arr[0] = times[start_idx].split(':')[0]
        prev_time_arr[1] = times[start_idx].split(':')[3]
        while looking:
            if end_idx == len(times):
                done = True
                break
            time = times[end_idx]
            if time.split(':')[0] == prev_time_arr[0] and time.split(':')[3] == prev_time_arr[1]:
                end_idx += 1
                continue
            else:
                break

        predictions = affinities[start_idx:end_idx].flatten()

        time = times[start_idx].split(':')[0]
        frame_idx = list(Positions[:, 0]).index(time)
        frame = Positions[frame_idx]

        if dominant_sets:
            bool_groups = iterate_climb_learned(predictions, frame, n_people, n_features=n_features)
        else:
            bool_groups = naive_group(predictions, frame, n_people, n_features=n_features)

        TP_n, FN_n, FP_n, precision, recall = \
            group_correctness(group_names(bool_groups, n_people), Groups_at_time[time], T, non_reusable=non_reusable)

        avg_results += np.array([precision, recall])
        start_idx = end_idx

    avg_results /= num_times

    if avg_results[0] * avg_results[1] == 0:
        f1_avg = 0
    else:
        f1_avg = float(2) * avg_results[0] * avg_results[1] / (avg_results[0] + avg_results[1])

    return f1_avg, avg_results[0], avg_results[1]


def F1_calc_clone(group_thres, affinities, frames, groups_at_time, positions, n_people, n_features, non_reusable=False,
                  dominant_sets=True):
    T = group_thres
    avg_results = np.array([0.0, 0.0])

    # this assumes affinities and times are the same length
    # TODO adapt to new approach
    frames = frames.flatten()
    unique_frames = np.unique(frames)
    num_times = 1
    for unique_frame in unique_frames:
        idx = [i for i, frame in enumerate(frames) if frame == unique_frame]
        predictions = affinities[idx].flatten()

        # TODO get positions (features of agents for each frame)
        frame_idx = list(positions[:, 0]).index(unique_frame)
        frame_positions = positions[frame_idx]

        if dominant_sets:
            bool_groups = iterate_climb_learned_clone(predictions, frame_positions, n_people, n_features=n_features)
        else:
            bool_groups = naive_group_clone(predictions, frame_positions, n_people, n_features=n_features)

        groups_at_time = groups_at_time[idx]
        TP_n, FN_n, FP_n, precision, recall = group_correctness_clone(group_names_clone(bool_groups, n_people),
                                                                      groups_at_time, T,
                                                                      non_reusable=non_reusable)

        avg_results += np.array([precision, recall])
        num_times += 1

    avg_results /= num_times

    if avg_results[0] * avg_results[1] == 0:
        f1_avg = 0
    else:
        f1_avg = float(2) * avg_results[0] * avg_results[1] / (avg_results[0] + avg_results[1])

    return f1_avg, avg_results[0], avg_results[1]


# calculates true positives, false negatives, and false positives
# given the guesses, the true groups, and the threshold T
def group_correctness(guesses, truth, T, non_reusable=False):
    n_true_groups = len(truth)
    n_guess_groups = len(guesses)

    if n_true_groups == 0 and n_guess_groups == 0:
        return 0, 0, 0, 1, 1

    elif n_true_groups == 0:
        return 0, n_guess_groups, 0, 0, 1

    elif n_guess_groups == 0:
        return 0, 0, n_true_groups, 1, 0
    else:
        TP = 0
        for true_group in truth:
            if len(true_group) <= 1:
                n_true_groups -= 1

        for guess in guesses:
            if len(guess) <= 1:
                n_guess_groups -= 1
                continue

        for true_group in truth:
            if len(true_group) <= 1:
                continue

            for guess in guesses:
                if len(guess) <= 1:
                    continue

                n_found = 0
                for person in guess:
                    if person in true_group:
                        n_found += 1

                if float(n_found) / max(len(true_group), len(guess)) >= T:
                    if non_reusable:
                        guesses.remove(guess)
                    TP += 1

        FP = n_guess_groups - TP
        FN = n_true_groups - TP
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        return TP, FN, FP, precision, recall


def group_correctness_clone(guesses, truth, T, non_reusable=False):
    n_true_groups = len(truth)
    n_guess_groups = len(guesses)

    if n_true_groups == 0 and n_guess_groups == 0:
        return 0, 0, 0, 1, 1

    elif n_true_groups == 0:
        return 0, n_guess_groups, 0, 0, 1

    elif n_guess_groups == 0:
        return 0, 0, n_true_groups, 1, 0
    else:
        TP = 0
        for true_group in truth:
            if len(true_group) <= 1:
                n_true_groups -= 1

        for guess in guesses:
            if len(guess) <= 1:
                n_guess_groups -= 1
                continue

        for true_group in truth:
            if len(true_group) <= 1:
                continue

            for guess in guesses:
                if len(guess) <= 1:
                    continue

                n_found = 0
                for person in guess:
                    if person in true_group:
                        n_found += 1

                if float(n_found) / max(len(true_group), len(guess)) >= T:
                    if non_reusable:
                        guesses.remove(guess)
                    TP += 1

        FP = n_guess_groups - TP
        FN = n_true_groups - TP
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        return TP, FN, FP, precision, recall


# for a set of vectors of the form [0,1,0,...,1], return a set of vectors of group names
# for more efficiency later, we should represent groups the first way, but for now we do this
def group_names(bool_groups, n_people):
    groups = []
    for bool_group in bool_groups:
        group = []
        for i in range(n_people):
            if bool_group[i]:
                group.append("ID_00" + str(i + 1))
        groups.append(group)
    return groups


def group_names_clone(bool_groups, n_people):
    groups = []
    for bool_group in bool_groups:
        group = []
        for i in range(n_people):
            if bool_group[i]:
                group.append("ID_00" + str(i + 1))
        groups.append(group)
    return groups
