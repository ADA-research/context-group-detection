from collections import Counter

import numpy as np
import pandas as pd

from trajdataset import TrajDataset


def load_eth(path, **kwargs):
    traj_dataset = TrajDataset()

    csv_columns = ["frame_id", "agent_id", "pos_x", "pos_z", "pos_y", "vel_x", "vel_z", "vel_y"]
    # read from csv => fill traj table
    raw_dataset = pd.read_csv(path, sep=r"\s+", header=None, names=csv_columns)

    traj_dataset.title = kwargs.get('title', "no_title")

    # copy columns
    traj_dataset.data[["frame_id", "agent_id",
                       "pos_x", "pos_y",
                       "vel_x", "vel_y"
                       ]] = \
        raw_dataset[["frame_id", "agent_id",
                     "pos_x", "pos_y",
                     "vel_x", "vel_y"
                     ]]

    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', -1)
    if fps < 0:
        d_frame = np.diff(pd.unique(raw_dataset["frame_id"]))
        fps = d_frame[0] * 2.5  # 2.5 is the common annotation fps for all (ETH+UCY) datasets

    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset


def read_obsmat(directory):
    '''
    Reads an obsmat.txt file from the given directory and converts it to a dataframe
    :param directory: name of the directory
    :return: dataframe
    '''
    columns = ['frame_id', 'agent_id', 'pos_x', 'pos_z', 'pos_y', 'v_x', 'v_z', 'v_y']
    df = pd.read_csv(directory + '/obsmat.txt', sep='\s+', names=columns, header=None)
    df.drop(columns=['pos_z', 'v_z'], inplace=True)
    # modify data types
    df["frame_id"] = df["frame_id"].astype(int)
    if str(df["agent_id"].iloc[0]).replace('.', '', 1).isdigit():
        df["agent_id"] = df["agent_id"].astype(int)
    df["pos_x"] = df["pos_x"].astype(float)
    df["pos_y"] = df["pos_y"].astype(float)

    d_frame = np.diff(pd.unique(df["frame_id"]))
    fps = d_frame[0] * 2.5  # 2.5 is the common annotation fps for all (ETH+UCY) datasets
    df["timestamp"] = df["frame_id"] / fps

    return df


def merge_groups_with_common_agents(agents_in_multiple_groups, groups):
    '''
    Merge groups with common agents.
    :param groups: list of lists representing the agent groups
    :return: list of lists without agents being in multiple groups
    '''
    for agent in agents_in_multiple_groups:
        group_indices = []
        for i, group in enumerate(groups):
            if agent in group:
                group_indices.append(i)
        groups_to_be_merged = [groups[i] for i in group_indices]
        merged_group = list(set(agent for group in groups_to_be_merged for agent in group))
        for c, i in enumerate(group_indices):
            groups.pop(i - c)
        groups.append(merged_group)
    return groups


def read_groups(directory):
    '''
    Reads a groups.txt file from the given directory and
    converts it to pairs of pedestrians in the same group
    :param directory: name of the directory
    :return: pairs
    '''

    with open(directory + '/groups.txt') as f:
        groups = [line.split() for line in f if not line.isspace()]

    # merge groups with common agents
    count_dict = Counter([agent for group in groups for agent in group])
    agents_in_multiple_groups = [key for key, value in count_dict.items() if value > 1]
    if len(agents_in_multiple_groups) > 0:
        groups_with_duplicate_agents_indices = \
            set(i for i, group in enumerate(groups) for agent in agents_in_multiple_groups if agent in group)
        groups_without_duplicate_agents = \
            [group for i, group in enumerate(groups) if i not in groups_with_duplicate_agents_indices]
        groups_with_duplicate_agents = \
            [group for i, group in enumerate(groups) if i in groups_with_duplicate_agents_indices]
        groups = \
            groups_without_duplicate_agents + merge_groups_with_common_agents(agents_in_multiple_groups,
                                                                              groups_with_duplicate_agents)

    return groups


if __name__ == '__main__':
    dataset_path = './UCY/students03'
    eth_df = read_obsmat(dataset_path)
    eth_groups = read_groups(dataset_path)

    eth_traj_dataset = load_eth('./ETH/seq_eth/obsmat.txt')
    eth_trajs = eth_traj_dataset.get_trajectories()
    traj_1 = eth_traj_dataset.data.iloc[eth_trajs.groups[0, 1]]  # get trajectory of scene_id 0 and pedestrian_id 1
    traj_1_v2 = eth_traj_dataset.data[eth_traj_dataset.data['agent_id'] == 1]
    pass
