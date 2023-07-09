import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def groups_initialize(n, ga_values_factor=3):
    """
    Initialize groups
    :param n: number of atoms
    :param ga_values_factor:
    :return:
    """
    # TODO handle group sizes
    # an array denoting groups corresponding to each individual
    group_initial = np.random.choice(np.arange(n), size=(n,))
    # a dictionary denoting groups corresponding to each individual
    ga_dict = dict({i: g for i, g in enumerate(group_initial)})
    # group assignment matrix: ga_matrix[i,k]=1 denotes node i belongs to kth group
    ga_matrix = np.zeros((n, n))
    ga_matrix[list(ga_dict.keys()), list(ga_dict.values())] = 1
    ga_values = ga_values_factor * ga_matrix.copy()
    # group relation matrix: gr_matrix[i,j]=1 denotes i and j are in one group
    gr_matrix = np.zeros((n, n))
    for i in range(gr_matrix.shape[0]):
        for j in range(gr_matrix.shape[1]):
            if ga_dict[i] == ga_dict[j]:
                gr_matrix[i, j] = 1
    return group_initial, ga_dict, ga_matrix, ga_values, gr_matrix


def groups_to_interactions(groups, k=3, b=0.1):
    """
    Convert groups to interactions based on equation P(I[i,j]=1|G[i,j])=1-exp(-k(G[i,j]+b)).
    :param groups: G of the equation
    :param k: controls the overall magnitude of the probabilities
    :param b: has great impact on the non-group interaction probability
    :return: interactions
    """
    I = np.zeros_like(groups)
    for i in range(groups.shape[0]):
        for j in range(groups.shape[1]):
            I[i, j] = np.random.choice([1, 0], p=[1 - np.exp(-k * (groups[i, j] + b)), np.exp(-k * (groups[i, j] + b))])
    np.fill_diagonal(I, 0)
    # Symmetric
    I = np.tril(I) + np.tril(I).T
    return I


class SpringSim(object):
    """
    copied from https://github.com/ethanfetaya/NRI/blob/master/data/synthetic_sim.py
    """

    def __init__(self, n_balls=5, box_size=10., loc_std=0.5, vel_norm=0.5, interaction_strength=0.1, noise_var=0.,
                 ga_values_factor=3, K=3, b=0.001):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._delta_T = 0.004
        self._max_F = 0.1 / self._delta_T
        self.ga_values_factor = ga_values_factor
        self.K = K
        self.b = b

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):
            K = 0.5 * (vel ** 2).sum()  # kinetic energy
            U = 0  # potential energy
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        """
        :param loc: 2xN locations at one time step
        :param vel: 2xN velocity at one time step
        :return: location and velocity after hitting walls and returning after elastically colliding with walls
        """
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]

        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        i.e. dist[i,j] = |A[i,:]-B[j,:]|^2
        :param A: Nxd matrix
        :param B: Mxd matrix
        :return: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=100):
        """
        Interaction edges may change at each timestep
        if dynamic, the group assignment will be re-evaluated at each sampled
            timestep according to ga_values
            ga_values will change at each time according to current group assignment ga_matrix
            and group assignment ages ga_ages
            if the group assignment ga_matrix[i] changes at one sampled timestep,
            the corresponding ga_ages[i] will be reset to 0 and the ga_values[i] will be reset
            to ga_values_factor*ga_matrix[i]
        """
        n = self.n_balls
        K = self.K
        b = self.b
        ga_values_factor = self.ga_values_factor
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        counter = 0

        # Initialize groups
        ga, ga_dict, ga_matrix, ga_values, gr = groups_initialize(n, ga_values_factor)
        # Initialize Interaction matrix
        edges = groups_to_interactions(gr, K, b)

        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)
        inter = np.zeros((T_save, n, n))
        inter[0, :, :] = edges

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):
            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size, 0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F

            # run leapfrog
            for i in range(1, T):
                # Assumption: the next states(loc and vel) are determined by
                # current states and current interaction edges
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)
                # compute current interaction edges based on group relationship
                edges = groups_to_interactions(gr, K, b)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    inter[counter, :, :] = edges
                    counter += 1

                forces_size = -self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n)))).sum(
                    axis=-1)

                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F

            # Add noise to observations
            loc += np.random.randn(T_save, 2, n) * self.noise_var
            vel += np.random.randn(T_save, 2, n) * self.noise_var
            return loc, vel, inter, ga, gr


def generate_dataset(num_sims, length, sample_freq):
    """
    Generate dataset of simulations.
    :param num_sims: number of simulations to generate
    :param length: length of simulation
    :param sample_freq: sample frequency of simulation
    :return:
    """
    locations = list()  # shape: [num_sims, num_sampledTimesteps, num_features, num_atoms]
    velocities = list()  # shape: [num_sims, num_sampledTimesteps, num_features, num_atoms]
    interactions = list()  # shape: [num_sims, num_sampledTimesteps, num_atoms, num_atoms]
    # group assignment list
    group_assignments = list()  # shape: [num_sims, (num_sampledTimesteps), num_atoms]
    # group relationship list
    group_relationships = list()  # shape: [num_sims, (num_sampledTimesteps), num_atoms, num_atoms]

    for i in range(num_sims):
        t = time.time()
        # return vectors of one simulation
        loc, vel, inter, ga, gr = sim.sample_trajectory(T=length, sample_freq=sample_freq)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

        locations.append(loc)
        velocities.append(vel)
        interactions.append(inter)
        group_assignments.append(ga)
        group_relationships.append(gr)

    locations = np.stack(locations)
    velocities = np.stack(velocities)
    interactions = np.stack(interactions)
    group_assignments = np.stack(group_assignments)
    group_relationships = np.stack(group_relationships)

    return locations, velocities, interactions, group_assignments, group_relationships


def get_simulation_dataframe(locations, velocities):
    """
    Get dataframe of simulation.
    :param locations: location data
    :param velocities: velocity data
    :return: dataframe of simulation
    """
    data = np.concatenate((locations.transpose(0, 3, 1, 2), velocities.transpose(0, 3, 1, 2)), axis=3)

    agent_dfs = []

    for sim in range(data.shape[0]):
        sim_data = data[sim]
        for agent in range(sim_data.shape[0]):
            agent_df = pd.DataFrame(sim_data[agent].reshape(-1, 4), columns=['pos_x', 'pos_y', 'v_x', 'v_y'])
            agent_df['agent_id'] = agent
            agent_df['frame_id'] = [i for i, _ in enumerate(agent_df['agent_id'])]
            agent_df['sim'] = sim
            agent_dfs.append(agent_df)

    df = pd.concat(agent_dfs, ignore_index=True)

    return df


def get_group_list(ga):
    sims = ga.shape[0]
    group_list = []
    for sim in range(sims):
        sim_ga = ga[sim]
        group_dict = {}

        for agent, group in enumerate(sim_ga):
            if group in group_dict:
                group_dict[group].append(agent)
            else:
                group_dict[group] = [agent]

        group_list.append(list(group_dict.values()))
    return group_list


def save_data(save_folder, df, groups, number, data):
    save_folder_path = save_folder + '/sim_{}'.format(number)

    os.makedirs(save_folder_path, exist_ok=True)
    df.to_csv('{}/data.csv'.format(save_folder_path), index=False)

    group_filename = '{}/groups.txt'.format(save_folder_path)

    with open(group_filename, 'w') as file:
        sims = len(groups)
        for sim in range(sims):
            sim_groups = groups[sim]
            for group in sim_groups:
                file.write(' '.join(str(agent_id) for agent_id in group) + '\n')
            file.write('-\n')

    np.save('{}/loc_all_sim{}.npy'.format(save_folder_path, suffix), data['loc'])
    np.save('{}/vel_all_sim{}.npy'.format(save_folder_path, suffix), data['vel'])
    np.save('{}/inter_all_sim{}.npy'.format(save_folder_path, suffix), data['inter'])
    np.save('{}/ga_sim{}.npy'.format(save_folder_path, suffix), data['ga'])
    np.save('{}/gr_sim{}.npy'.format(save_folder_path, suffix), data['gr'])


def plot(sim, loc, vel, inter, ga):
    plt.set_cmap('Set3')

    # Get a color palette and assign colors to groups
    color_palette = plt.get_cmap('Set3')
    colors = {group: color_palette(i) for i, group in enumerate(np.unique(ga))}

    plt.figure()
    axes = plt.gca()
    # axes.set_xlim([-2., 3.])
    # axes.set_ylim([-3., 3.])
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_title('Trajectories')
    # plt.title('Trajectories')
    for i in range(loc.shape[-1]):
        plt.plot(loc[:, 0, i], loc[:, 1, i], color=colors[ga[i]])
        plt.plot(loc[0, 0, i], loc[0, 1, i], color=colors[ga[i]], marker='o')

    plt.figure()
    energies = [sim._energy(loc[i, :, :], vel[i, :, :], inter) for i in range(loc.shape[0])]
    plt.plot(energies)
    plt.title('Energies')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--num-sim", type=int, default=1, help="number of simulations to perform.")
    parser.add_argument("--length", type=int, default=5000, help="length of trajectory.")
    parser.add_argument("--sample-freq", type=int, default=100, help="how often to sample the trajectory.")
    parser.add_argument("--n-balls", type=int, default=10, help="number of balls in the simulation.")
    parser.add_argument("--ga-values-factor", type=int, default=5, help="group assignment value factor")
    parser.add_argument("--K", type=float, default=3.0, help="K")
    parser.add_argument("--b", type=float, default=0.05, help="b")

    parser.add_argument('--groups', type=int, default=20)
    parser.add_argument('--save_folder', type=str, default='.')
    parser.add_argument('--plot', action="store_true", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)

    sim = SpringSim(n_balls=args.n_balls, ga_values_factor=args.ga_values_factor, K=args.K, b=args.b)

    np.random.seed(args.seed)

    print("Generating {} simulations".format(args.num_sim))
    locations, velocities, interactions, group_assignments, group_relationships = generate_dataset(
        args.num_sim,
        args.length,
        args.sample_freq)

    # TODO convert and save data
    #  data to dataframe (DONE)
    #  group assignments to groups (for each sim?)
    df = get_simulation_dataframe(locations, velocities)
    groups = get_group_list(group_assignments)

    data = {
        'loc': locations,
        'vel': velocities,
        'inter': interactions,
        'ga': group_assignments,
        'gr': group_relationships
    }
    suffix = '{}_{}_{}'.format(args.n_balls, args.K, args.b * 100)
    save_data(args.save_folder, df, groups, suffix, data)

    if args.plot:
        plot(sim, locations[0], velocities[0], interactions[0], group_assignments[0])
