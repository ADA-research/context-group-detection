import matplotlib.pyplot as plt
import numpy as np


def get_group_trajectory(agents_per_group, num_steps, group_velocity, initial_position, final_position, waypoints):
    # Generate initial positions and velocities for the group
    initial_positions = np.random.normal(loc=initial_position, scale=2, size=(agents_per_group, 2))
    velocities = np.random.normal(loc=group_velocity, scale=0.2, size=(agents_per_group, 2))

    # Initialize arrays to store trajectory data
    positions = np.zeros((agents_per_group, num_steps, 2), dtype=np.float64)  # (agents_per_group, num_steps, x/y)
    all_velocities = np.zeros((agents_per_group, num_steps, 2),
                              dtype=np.float64)  # (agents_per_group, num_steps, vx/vy)

    # Simulate the trajectory for each agent in the group
    for a in range(agents_per_group):
        position = initial_positions[a, :]
        positions[a, 0, :] = position
        all_velocities[a, 0, :] = velocities[a, :]

        # Create a list of waypoints including the final position
        all_waypoints = waypoints + [final_position]

        steps = int((num_steps - 1) / len(all_waypoints))

        for i in range(len(all_waypoints)):
            # Calculate the direction vector towards the next waypoint
            direction = all_waypoints[i] - position
            distance = np.linalg.norm(direction)
            direction /= distance

            # Update position based on velocity and direction
            for t in range(i * steps, (i + 1) * steps):
                position += velocities[a, :] * direction

                # Update velocity towards the next waypoint
                velocities[a, :] += direction * group_velocity

                # Store the updated position and velocity
                positions[a, t + 1, :] = position
                all_velocities[a, t + 1, :] = velocities[a, :]

    return positions, all_velocities


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
    start_position = np.array([np.random.uniform(*x_range), np.random.uniform(*y_range)])
    final_position = np.array([np.random.uniform(*x_range), np.random.uniform(*y_range)])
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


num_steps = 1000
group_velocity = 0.1
x_range = (0, 100)  # Range for x-coordinate of waypoints
y_range = (0, 100)  # Range for y-coordinate of waypoints

positions = []
velocities = []
groups = []
agent_id = 0

num_groups = 2
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
    groups.append(list(range(agent_id, agent_id + len(group_positions))))
    agent_id += len(group_positions)

positions = np.concatenate(positions)
velocities = np.concatenate(velocities)

# Visualize trajectory
for a in range(len(positions)):
    if a in groups[0]:
        marker = 'o'
    else:
        marker = 'x'
    plt.plot(positions[a, :, 0], positions[a, :, 1], marker='o')

plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Trajectory Simulation')
plt.show()
