import argparse

import matplotlib.pyplot as plt
import numpy as np


def get_group_trajectory(agents_per_group, num_steps, group_velocity, initial_position, final_position, waypoints):
    # Generate initial positions and velocities for the group
    initial_positions = np.random.normal(loc=initial_position, scale=2, size=(agents_per_group, 2))
    final_positions = np.random.normal(loc=final_position, scale=2, size=(agents_per_group, 2))
    initial_velocities = np.random.normal(loc=group_velocity, scale=0.2, size=(agents_per_group, 2))

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
                position += velocity
                position += np.random.normal(scale=0.3, size=2)

                # Store the updated position and velocity
                positions[a, t + 1, :] = position
                velocities[a, t + 1, :] = velocity
                # Add small noise to the velocity vector
                # velocity += np.random.normal(scale=0.3, size=2)
            current_waypoint = next_waypoint

    for step in reversed(range(num_steps)):
        if positions[0, step, 0] == 0 and positions[0, step, 1] == 0:
            positions[:, step, :] = final_positions

    return positions, velocities


def generate_random_waypoints(max_num_waypoints, x_range, y_range):
    num_waypoints = np.random.randint(0, max_num_waypoints + 1)
    waypoints = []
    for _ in range(num_waypoints):
        waypoint = np.array([np.random.uniform(*x_range), np.random.uniform(*y_range)])
        waypoints.append(waypoint)
    return waypoints


def simulate_group_trajectory(agents_per_group, num_steps, group_velocity, max_num_waypoints, x_range, y_range):
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
    waypoints = generate_random_waypoints(max_num_waypoints=max_num_waypoints, x_range=x_range, y_range=y_range)
    # Simulate trajectory for a single group
    positions, velocities = get_group_trajectory(agents_per_group=agents_per_group,
                                                 num_steps=num_steps,
                                                 group_velocity=group_velocity,
                                                 initial_position=start_position,
                                                 final_position=final_position,
                                                 waypoints=waypoints)
    return positions, velocities


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--velocity', type=float, default=0.1)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    np.random.seed(args.seed)
    num_steps = args.frames
    group_velocity = args.velocity
    x_range = (0, 100)  # Range for x-coordinate of waypoints
    y_range = (0, 100)  # Range for y-coordinate of waypoints

    positions = []
    velocities = []
    groups = []

    agents = 0
    num_groups = 4
    for group in range(num_groups):
        agents_per_group = np.random.randint(3, 7)
        group_positions, group_velocities = simulate_group_trajectory(agents_per_group=agents_per_group,
                                                                      num_steps=num_steps,
                                                                      group_velocity=group_velocity,
                                                                      max_num_waypoints=3,
                                                                      x_range=x_range,
                                                                      y_range=y_range)
        positions.append(group_positions)
        velocities.append(group_velocities)
        groups.append(list(range(agents, agents + len(group_positions))))
        agents += len(group_positions)

    positions = np.concatenate(positions)
    velocities = np.concatenate(velocities)

    # Visualize trajectory
    markers = ['.', ',', 'o', 'v', '^', '<', '>', 's', '+', 'x', 'D', 'd', 'p', '*', 'h', 'H']
    for group_idx, group in enumerate(groups):
        marker = np.random.choice(markers)
        markers.remove(marker)
        for i, agent in enumerate(group):
            if i == 0:
                plt.plot(positions[agent, :, 0], positions[agent, :, 1], marker=marker, label='group {}'.format(group_idx))
            else:
                plt.plot(positions[agent, :, 0], positions[agent, :, 1], marker=marker)

    plt.title('Trajectory Simulation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
