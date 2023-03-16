import csv
import os

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
    columns = ['frame_number', 'pedestrian_ID', 'pos_x', 'pos_z', 'pos_y', 'v_x', 'v_z', 'v_y']
    df = pd.read_csv(directory + '/obsmat.txt', sep='\s+', names=columns, header=None)
    df.drop(columns=['pos_z', 'v_z'], inplace=True)

    return df


def read_groups(directory):
    '''
    Reads a groups.txt file from the given directory and
    converts it to pairs of pedestrians in the same group
    :param directory: name of the directory
    :return: pairs
    '''

    with open(directory + '/groups.txt') as f:
        groups = [line.rstrip().lstrip().split(' ') for line in f if line.rstrip().lstrip() != '']

    return groups


def read_geometry(directory):
    '''
    Reads multiple csv files from the given directory and stores their data to a dataframe
    :param directory: name of the directory
    :return: dataframe
    '''
    columns = ['Timestamp', 'Ground_Position_X', 'Ground_Position_Y', 'Useless_Field', 'Body_Pose',
               'Relative_Head2Body_Pose', 'Validity']
    dfs = []
    geometry = directory + '/geometryGT'
    for filename in os.listdir(geometry):
        df = pd.read_csv(geometry + '/' + filename, names=columns)
        df.drop(columns=['Useless_Field'], inplace=True)
        dfs.append(df)
    df = pd.concat(dfs)
    return df


# TODO group pairs per frame
#   check if it works
def read_fformation(directory):
    '''
    Reads a fformationGT.csv file from the given directory and
    converts it to pairs of pedestrians in the same group
    :param directory: name of the directory
    :return: pairs
    '''

    frame_groups = {}
    with open(directory + '/fformationGT.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        frames = [line for line in csv_reader]
        for frame in frames:
            frame_id = frame[0]
            group = frame[1:]
            if frame_id not in frame_groups.keys():
                frame_groups[frame_id] = [group]
            else:
                frame_groups[frame_id].append(group)

    return frame_groups


if __name__ == '__main__':
    eth_df = read_obsmat('./ETH/seq_eth')
    eth_groups = read_groups('./ETH/seq_eth')
    eth_traj_dataset = load_eth('./ETH/seq_eth/obsmat.txt')
    eth_trajs = eth_traj_dataset.get_trajectories()

    salsa_cp_df = read_geometry('./SALSA/CocktailParty')
    salsa_cp_groups_df = read_fformation('./SALSA/CocktailParty')
