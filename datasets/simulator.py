import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_waypoints(max_num_waypoints, x_range, y_range):
    """
    Generate waypoints for a group based on parameters.
    :param max_num_waypoints: maximum number of waypoints
    :param x_range: range of x-axis
    :param y_range: range of y-axis
    :return: coordinates of generated waypoints
    """
    num_waypoints = np.random.randint(0, max_num_waypoints + 1)
    waypoints = []
    for _ in range(num_waypoints):
        waypoint = np.array([np.random.uniform(*x_range), np.random.uniform(*y_range)])
        waypoints.append(waypoint)
    return waypoints


def generate_group_trajectories(agents_per_group, num_steps, initial_velocity, initial_position, final_position,
                                waypoints):
    """
    Generate trajectories of a group based on parameters.
    :param agents_per_group: number of agents in group
    :param num_steps: length of data
    :param initial_velocity: starting velocity of group
    :param initial_position: initial position of group members
    :param final_position: final position of group members
    :param waypoints: list of waypoints for group to go through
    :return: arrays of positions and velocities
    """
    # Generate initial positions and velocities for the group
    initial_positions = np.random.normal(loc=initial_position, scale=2, size=(agents_per_group, 2))
    final_positions = np.random.normal(loc=final_position, scale=2, size=(agents_per_group, 2))
    initial_velocities = np.random.normal(loc=initial_velocity, scale=0.2, size=(agents_per_group, 2))

    # Initialize arrays to store trajectory data
    # (agents_per_group, num_steps, x/y)
    positions = np.zeros((agents_per_group, num_steps, 2), dtype=np.float64)
    # (agents_per_group, num_steps, vx/vy)
    velocities = np.zeros((agents_per_group, num_steps, 2), dtype=np.float64)

    # Create a list of waypoints including the final position
    all_waypoints = waypoints + [final_position]

    steps = int((num_steps - 1) / len(all_waypoints))

    # Simulate the trajectory for each agent in the group
    for a in range(agents_per_group):
        position = initial_positions[a, :]
        positions[a, 0, :] = position
        velocity = initial_velocities[a, :]
        velocities[a, 0, :] = velocity

        current_waypoint = initial_position
        for i in range(len(all_waypoints)):
            next_waypoint = all_waypoints[i]
            # Calculate the direction vector towards the next waypoint
            direction = next_waypoint - current_waypoint
            distance = np.linalg.norm(direction)
            direction /= distance
            adjusted_velocity = distance / steps
            velocity = direction * adjusted_velocity

            # Update position based on velocity
            for t in range(i * steps, (i + 1) * steps):
                position += velocity + np.random.normal(scale=0.3, size=2)

                # Store the updated position and velocity
                positions[a, t + 1, :] = position
                velocities[a, t + 1, :] = velocity
                # Add small noise to the velocity vector
                velocity += np.random.normal(scale=0.01, size=2)
            current_waypoint = next_waypoint

    for step in reversed(range(num_steps)):
        if positions[0, step, 0] == 0 and positions[0, step, 1] == 0:
            positions[:, step, :] = final_positions + np.random.normal(scale=0.3, size=2)
            velocities[:, step, :] = velocity + np.random.normal(scale=0.01, size=2)

    return positions, velocities


def simulate_group_trajectories(agents_per_group, num_steps, initial_velocity, max_num_waypoints, x_range, y_range):
    """
    Simulate trajectories of single group based on parameters.
    :param agents_per_group: number of agents in group
    :param num_steps: length of data
    :param initial_velocity: starting velocity of group
    :param max_num_waypoints: maximum number of waypoints
    :param x_range: range of x-axis
    :param y_range: range of y-axis
    :return: arrays of positions and velocities
    """
    # Simulation parameters
    # Generate random start and final positions
    start_x = np.random.choice([x_range[0], x_range[1]])
    start_y = np.random.uniform(*y_range)
    if np.random.choice([True, False]):
        start_x, start_y = start_y, start_x
    start_position = np.array([start_x, start_y])

    final_x = np.random.choice([x_range[0], x_range[1]])
    final_y = np.random.uniform(*y_range)
    if np.random.choice([True, False]):
        final_x, final_y = final_y, final_x
    final_position = np.array([final_x, final_y])
    # Generate random waypoints
    waypoints = generate_waypoints(max_num_waypoints=max_num_waypoints, x_range=x_range, y_range=y_range)
    # Simulate trajectory for a single group
    positions, velocities = generate_group_trajectories(agents_per_group=agents_per_group,
                                                        num_steps=num_steps,
                                                        initial_velocity=initial_velocity,
                                                        initial_position=start_position,
                                                        final_position=final_position,
                                                        waypoints=waypoints)
    return positions, velocities


def create_simulation(num_steps, initial_velocity, x_range, y_range, num_groups, min_agents_per_group,
                      max_agents_per_group, max_num_waypoints):
    """
    Create simulation data based on parameters.
    :param num_steps: number of data frames for each group
    :param initial_velocity: starting velocity of group
    :param x_range: range of x-axis
    :param y_range: range of y-axis
    :param num_groups: number of groups to be generated
    :param min_agents_per_group: minimum number of agents in a group
    :param max_agents_per_group: maximum number of agents in a group
    :param max_num_waypoints: maximum number of waypoints
    :return: arrays of positions, velocities and groups
    """
    positions = []
    velocities = []
    groups = []
    agents = 0
    for group in range(num_groups):
        agents_per_group = np.random.randint(min_agents_per_group, max_agents_per_group)
        group_positions, group_velocities = simulate_group_trajectories(agents_per_group=agents_per_group,
                                                                        num_steps=num_steps,
                                                                        initial_velocity=initial_velocity,
                                                                        max_num_waypoints=max_num_waypoints,
                                                                        x_range=x_range,
                                                                        y_range=y_range)
        positions.append(group_positions)
        velocities.append(group_velocities)
        groups.append(list(range(agents, agents + len(group_positions))))
        agents += len(group_positions)
    positions = np.concatenate(positions)
    velocities = np.concatenate(velocities)

    return positions, velocities, groups


def get_simulation_dataframe(data, frames_per_group, dataset_frames):
    """
    Get dataframe of simulation.
    :param data: simulation data
    :param frames_per_group: number of data frames for each group
    :param dataset_frames: total number of data frames for simulation
    :return: dataframe of simulation
    """
    agent_dfs = []

    last_start_frame = dataset_frames - frames_per_group
    start_frames = [0, last_start_frame] + list(range(0, last_start_frame + 1, int(frames_per_group / 2)))

    for agent in range(data.shape[0]):
        start_frame = np.random.choice(start_frames)
        agent_df = pd.DataFrame(data[agent].reshape(-1, 4), columns=['pos_x', 'pos_y', 'v_x', 'v_y'])
        agent_df['agent_id'] = agent
        agent_df['frame_id'] = [i + start_frame for i, _ in enumerate(agent_df['agent_id'])]
        agent_dfs.append(agent_df)

    df = pd.concat(agent_dfs, ignore_index=True)

    d_frame = np.diff(pd.unique(df["frame_id"]))
    fps = d_frame[0] * 1
    df["timestamp"] = df["frame_id"] / fps

    df.sort_values(by=['agent_id', 'frame_id'], inplace=True)
    df['measurement'] = df[['pos_x', 'pos_y', 'v_x', 'v_y']].apply(tuple, axis=1)

    return df


def get_simulation_data(frames_per_group, initial_velocity, num_groups, min_agents_per_group, max_agents_per_group,
                        max_num_waypoints, dataset_frames, seed):
    """
    Create simulation and get data based on parameters.
    :param frames_per_group: number of data frames for each group
    :param initial_velocity: starting velocity of group
    :param num_groups: number of groups to be generated
    :param min_agents_per_group: minimum number of agents in a group
    :param max_agents_per_group: maximum number of agents in a group
    :param max_num_waypoints: maximum number of waypoints
    :param dataset_frames: total number of data frames for simulation
    :param seed: random seed to be used
    :return: dataframe + groups
    """

    np.random.seed(seed)

    positions, velocities, groups = create_simulation(num_steps=frames_per_group,
                                                      initial_velocity=initial_velocity,
                                                      num_groups=num_groups,
                                                      min_agents_per_group=min_agents_per_group,
                                                      max_agents_per_group=max_agents_per_group,
                                                      max_num_waypoints=max_num_waypoints,
                                                      x_range=(0, 100),
                                                      y_range=(0, 100))
    data = np.concatenate((positions, velocities), axis=2)
    df = get_simulation_dataframe(data, frames_per_group, dataset_frames)

    return df, groups


def plot_trajectories(positions, groups, frames_range=(0, 100)):
    """
    Plot the trajectories for given frames.
    :param positions: position data
    :param groups: group information
    :param frames_range: frames to be considered on the plot
    :return:
    """

    positions = positions[:, frames_range[0]:frames_range[1]]

    markers = ['.', ',', 'o', 'v', '^', '<', '>', 's', '+', 'x', 'D', 'd', 'p', '*', 'h', 'H']
    for group_idx, group in enumerate(groups):
        marker = np.random.choice(markers)
        markers.remove(marker)
        if not markers:
            markers = ['.', ',', 'o', 'v', '^', '<', '>', 's', '+', 'x', 'D', 'd', 'p', '*', 'h', 'H']
        for i, agent in enumerate(group):
            # TODO do not plot zeros
            if i == 0:
                plt.plot(positions[agent, :, 0], positions[agent, :, 1], marker=marker,
                         label='group {}'.format(group_idx))
            else:
                plt.plot(positions[agent, :, 0], positions[agent, :, 1], marker=marker)

    plt.title('Trajectory Simulation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


def get_positions(df):
    """
    Get positions from dataframe.
    :param df: simulation dataframe
    :return: position data
    """
    grouped = df.groupby(['agent_id', 'frame_id'])
    measurements = grouped['measurement'].apply(lambda x: x.apply(lambda y: y[:2]).values[:2])

    agent_ids = df['agent_id'].unique()
    frame_ids = df['frame_id'].unique()
    agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    frame_id_to_idx = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}

    positions = np.zeros((len(agent_ids), len(frame_ids), 2))

    for (agent_id, frame_id), measurement in measurements.items():
        agent_idx = agent_id_to_idx[agent_id]
        frame_idx = frame_id_to_idx[frame_id]
        positions[agent_idx][frame_idx] = measurement[0]

    return positions


def save_data(df, groups):
    os.makedirs(args.save_folder + '/sim_{}'.format(args.seed), exist_ok=True)
    df.to_csv(args.save_folder + '/sim_{}/data.csv'.format(args.seed))

    group_filename = args.save_folder + '/sim_{}/groups.txt'.format(args.seed)

    with open(group_filename, 'w') as file:
        for group in groups:
            file.write(' '.join(str(agent_id) for agent_id in group) + '\n')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--groups', type=int, default=10)
    parser.add_argument('--velocity', type=float, default=0.1)
    parser.add_argument('--frames_per_group', type=int, default=100)
    parser.add_argument('--dataset_frames', type=int, default=1000)
    parser.add_argument('--max_num_waypoints', type=int, default=3)
    parser.add_argument('--min_agents_per_group', type=int, default=2)
    parser.add_argument('--max_agents_per_group', type=int, default=6)
    parser.add_argument('--save_folder', type=str, default='./simulation')
    parser.add_argument('--plot', action="store_true", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    df, groups = get_simulation_data(num_groups=args.groups,
                                     dataset_frames=args.dataset_frames,
                                     frames_per_group=args.frames_per_group,
                                     initial_velocity=args.velocity,
                                     min_agents_per_group=args.min_agents_per_group,
                                     max_agents_per_group=args.max_agents_per_group,
                                     max_num_waypoints=args.max_num_waypoints,
                                     seed=args.seed)

    save_data(df, groups)

    positions = get_positions(df)

    if args.plot:
        plot_trajectories(positions, groups)
