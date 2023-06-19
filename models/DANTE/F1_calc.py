from sklearn.cluster import DBSCAN

from models.DANTE.dominant_sets import *
from models.gmitre import compute_groupMitre


def calculate_f1(avg_results, num_times):
    """
    Calculates f1 for each metric.
    :param avg_results: list of tuples (precision, recall)
    :param num_times: number of frames to be averaged
    :return: list of tuples (F1, precision, recall)
    """
    f1s = []
    for i in range(len(avg_results)):
        avg_results[i] = avg_results[i] / num_times
        if avg_results[i][0] * avg_results[i][1] == 0:
            f1 = 0
        else:
            f1 = float(2) * avg_results[i][0] * avg_results[i][1] / (avg_results[i][0] + avg_results[i][1])
        f1s.append(f1)
    return [(f1s[i], avg_result[0], avg_result[1]) for i, avg_result in enumerate(avg_results)]


def F1_calc(group_thresholds, affinities, times, groups, positions, n_people, n_features, non_reusable=False,
            dominant_sets=True, eps_thres=1e-15):
    """
    Calculates average F1 for given threshold T.
    :param group_thresholds: threshold for group to be considered correctly detected
    :param affinities: predicted affinities
    :param times: list of timestamps
    :param groups: list of groups per timestamp
    :param positions: data in raw format
    :param n_people: number of agents
    :param n_features: number of features
    :param non_reusable: if predicted groups can be reused
    :param dominant_sets: True if dominant sets algorithm will be used, otherwise False
    :param eps_thres: threshold to be used in vector climb of dominant sets
    :return: F1, precision, recall
    """
    avg_results = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]

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
        frame_idx = list(positions[:, 0]).index(time)
        frame = positions[frame_idx]

        if dominant_sets:
            bool_groups = iterate_climb_learned(predictions, n_people, frame, n_features=n_features,
                                                eps_thres=eps_thres)
        else:
            bool_groups = naive_group(predictions, n_people, frame, n_features=n_features)

        for i, T in enumerate(group_thresholds):
            _, _, _, precision, recall = group_correctness(
                group_names(bool_groups, n_people), groups[time], T, non_reusable=non_reusable)
            avg_results[i] += np.array([precision, recall])

        start_idx = end_idx

    return calculate_f1(avg_results, num_times)


def include_single_agent_groups(groups_at_time, agents):
    """
    Look for agent in groups and if agent is not found add him as a single agent group.
    :param groups_at_time: groups of a scene
    :param agents: agents that appear in the scene
    :return:
    """
    for agent in agents:
        found = False
        for group in groups_at_time:
            if agent in group:
                found = True
                break
        if not found:
            groups_at_time.append([agent])


def labels_to_groups(labels):
    group_labels = np.unique(labels)
    groups = []

    for group_label in group_labels:
        if group_label != -1:
            groups.append([True if label == group_label else False for label in labels])

    return groups


def dbscan_algo(predictions, n_people, frames):
    A, agents_map = learned_affinity_clone(predictions, n_people, frames)

    dbscan = DBSCAN(eps=1, min_samples=2)
    labels = dbscan.fit_predict(A)

    groups = labels_to_groups(labels)

    return groups, agents_map


def F1_calc_clone(group_thresholds, affinities, frames, groups, positions, multi_frame=False,
                  non_reusable=False, dominant_sets=True, eps_thres=1e-15):
    """
    Calculates average F1 for thresholds 2/3, 1 and group mitre.
    :param group_thresholds: threshold for group to be considered correctly detected
    :param affinities: predicted affinities
    :param frames: list of frames
    :param groups: list of groups per scene
    :param positions: data in raw format
    :param multi_frame: True if scenes include multiple frames, otherwise False
    :param non_reusable: if predicted groups can be reused
    :param dominant_sets: True if dominant sets algorithm will be used, otherwise False
    :param eps_thres: threshold to be used in vector climb of dominant sets
    :param dominant_sets: True if dominant sets algorithm will be used, otherwise False
    :return: list of F1, precision, recall for T=2/3, T=1 and group mitre
    """
    avg_results = [np.array([0.0, 0.0]) for _ in range(len(group_thresholds))]

    num_times = 1
    frame_ids = [frame[0] for frame in frames]

    if multi_frame:
        frame_values = [list(x) for x in set(tuple(frame_id) for frame_id in frame_ids)]
    else:
        frame_values = np.unique(frame_ids)
    for unique_frame in frame_values:
        idx = [i for i, frame in enumerate(frame_ids) if frame == unique_frame]
        predictions = affinities[idx].flatten()

        if multi_frame:
            agents_by_frame = positions.groupby('frame_id')['agent_id'].apply(list).reset_index(name='agents')
            agent_list = \
                [set(agents_by_frame[agents_by_frame['frame_id'] == frame]['agents'].iloc[0]) for frame in unique_frame]
            n_people = len(set.intersection(*agent_list))
        else:
            n_people = len(positions[positions.frame_id == unique_frame])

        if dominant_sets:
            bool_groups, agents_map = iterate_climb_learned(predictions, n_people, frames[idx], new=True,
                                                            eps_thres=eps_thres)
        else:
            bool_groups, agents_map = dbscan_algo(predictions, n_people, frames[idx])
            # bool_groups, agents_map = naive_group(predictions, n_people, frames[idx], new=True)

        groups_at_time = [group[1] for group in groups if group[0] == unique_frame][0]
        if len(agents_map.values()) < 10:
            print('Wrong number of agents in map')
        include_single_agent_groups(groups_at_time, agents_map.values())
        predicted_groups = group_names_clone(bool_groups, agents_map, n_people)
        include_single_agent_groups(predicted_groups, agents_map.values())
        for i, T in enumerate(group_thresholds):
            if T is None:
                precision, recall, _ = compute_groupMitre(groups_at_time, predicted_groups)
            else:
                _, _, _, precision, recall = group_correctness(
                    predicted_groups, groups_at_time, T, non_reusable=non_reusable)
            avg_results[i] += np.array([precision, recall])
        num_times += 1

    return calculate_f1(avg_results, num_times)


def group_correctness(guesses, truth, T, non_reusable=False):
    """
    Calculates true positives, false negatives, and false positives.
    Given the guesses, the true groups, and the threshold T.
    :param guesses: predicted groups
    :param truth: ground truth groups
    :param T: threshold for group to be considered correctly detected
    :param non_reusable: if predicted groups can be reused
    :return: true positives, false negatives, false positives, precision, recall
    """
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
        precision = float(TP) / (TP + FP) if TP + FP != 0 else 0
        recall = float(TP) / (TP + FN) if TP + FN != 0 else 0
        return TP, FN, FP, precision, recall


def group_names(bool_groups, n_people):
    """
    For a set of vectors of the form [0,1,0,...,1], return a set of vectors of group names.
    :param bool_groups: list of lists, each list shows which agents are included in a group.
    :param n_people: number of agents in the scene
    :return: groups with agent ids
    """
    groups = []
    for bool_group in bool_groups:
        group = []
        for i in range(n_people):
            if bool_group[i]:
                group.append("ID_00" + str(i + 1))
        groups.append(group)
    return groups


def group_names_clone(bool_groups, agents_map, n_people):
    """
    For a set of vectors of the form [0,1,0,...,1], return a set of vectors of group names.
    :param bool_groups: list of lists, each list shows which agents are included in a group.
    :param agents_map: mapping of indices to agent ids
    :param n_people: number of agents in the scene
    :return: groups with agent ids
    """
    groups = []
    for bool_group in bool_groups:
        group = []
        for i in range(n_people):
            if bool_group[i]:
                group.append(agents_map[i])
        groups.append(group)
    return groups
