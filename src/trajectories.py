import numpy as np
import utils

def generate_circular_trajectory(target_positions: np.array, simulation_steps: np.array,
                                  circle_center: tuple[float, float], circle_radius: float, 
                                  trajectory_speed: float = None) -> np.array:
    total_steps = len(simulation_steps)
    
    # If trajectory_speed is None, use linear speed (m/s) along the circle perimeter
    if trajectory_speed is None:
        # Calculate the perimeter of the circle (2 * pi * radius)
        perimeter = 2 * np.pi * circle_radius
        
        # Calculate the distance the agent should move per simulation step based on linear speed
        distance_per_step = perimeter / total_steps
        
        for i in range(1, total_steps):
            # Calculate the total distance moved up to the current step
            distance_travelled = i * distance_per_step
            
            # Calculate the corresponding angle based on distance travelled along the perimeter
            angle = distance_travelled / circle_radius
            
            # Update target position using the angle
            target_positions[i, :] = [
                circle_center[0] + circle_radius * np.cos(angle),  # Counter-clockwise motion
                circle_center[1] + circle_radius * np.sin(angle)
            ]
    else:
        # If trajectory_speed is provided, calculate angular velocity for a full circle
        for i in range(1, total_steps):
            time = simulation_steps[i]/100 # Convert simulation step to time (assuming 100 ms per step)
            # Calculate the angular position based on trajectory speed
            angle = trajectory_speed * time

            # Update target position on the circle using parametric equations
            target_positions[i, :] = [
                circle_center[0] + circle_radius * np.cos(angle),
                circle_center[1] + circle_radius * np.sin(angle)
            ]
    
    return target_positions

def generate_noisy_circular_trajectory(target_positions: np.array, simulation_steps: np.array,
                                circle_center: tuple[float,float], circle_radius: float, 
                                trajectory_speed: float, noise_mean: float, noise_sigma: float) -> np.array:
    total_steps = len(simulation_steps)

    if trajectory_speed is None:
        # Calculate the perimeter of the circle (2 * pi * radius)
        perimeter = 2 * np.pi * circle_radius
        
        # Calculate the distance the agent should move per simulation step based on linear speed
        distance_per_step = perimeter / total_steps
        
        for i in range(1, total_steps):
            # Calculate the total distance moved up to the current step
            distance_travelled = i * distance_per_step
            
            # Calculate the corresponding angle based on distance travelled along the perimeter
            angle = distance_travelled / circle_radius
            
            # Update target position using the angle and add noise
            target_positions[i, :] = [
                circle_center[0] + circle_radius * np.cos(angle) + generate_noise(noise_mean, noise_sigma),  # Counter-clockwise motion
                circle_center[1] + circle_radius * np.sin(angle) + generate_noise(noise_mean, noise_sigma)
            ]
    else:
        # If trajectory_speed is provided, calculate angular velocity for a full circle
        for i in range(1, total_steps):
            time = simulation_steps[i]/100 # Convert simulation step to time (assuming 100 ms per step)
            # Calculate the angular position based on trajectory speed
            angle = trajectory_speed * time

            # update target position on the circle using parametric equations and add noise
            target_positions[i, :] = [
                circle_center[0] + circle_radius * np.cos(angle) + generate_noise(noise_mean, noise_sigma),
                circle_center[1] + circle_radius * np.sin(angle) + generate_noise(noise_mean, noise_sigma)
            ]
    
    return target_positions
            
def generate_linear_trajectory(target_positions: np.array, simulation_steps: np.array,
                                start_position: tuple[float, float] = (0, 0), end_position: tuple[float, float] = (0, 0)) -> np.array:
    total_steps = len(simulation_steps)
    start_position = np.array(start_position)
    end_position = np.array(end_position)
    # Calculate the direction vector from start to end position
    direction_vector = tuple(float(end_position[i] - start_position[i]) for i in range(len(start_position)))
    # Normalize the direction vector
    direction_vector /= np.linalg.norm(direction_vector)
    # Calculate the distance to move per step
    distance_per_step = np.linalg.norm(end_position - start_position) / total_steps
    for i in range(1, total_steps):
        # Calculate the position at each step
        target_positions[i, :] = start_position + i * distance_per_step * direction_vector
    return target_positions

def generate_noisy_linear_trajectory(target_positions: np.array, simulation_steps: np.array,
                                start_position: tuple[float, float] = (0, 0), end_position: tuple[float, float] = (0, 0), 
                                noise_mean: float = 0, noise_sigma: float = 0) -> np.array:
    total_steps = len(simulation_steps)
    start_position = np.array(start_position)
    end_position = np.array(end_position)
    # Calculate the direction vector from start to end position
    direction_vector = tuple(float(end_position[i] - start_position[i]) for i in range(len(start_position)))
    # Normalize the direction vector
    direction_vector /= np.linalg.norm(direction_vector)
    # Calculate the distance to move per step
    distance_per_step = np.linalg.norm(end_position - start_position) / total_steps
    for i in range(1, total_steps):
        # Calculate the position at each step
        target_positions[i, :] = start_position + i * distance_per_step * direction_vector + generate_noise(noise_mean, noise_sigma)
    return target_positions

def generate_sine_trajectory(target_positions: np.array, simulation_steps: np.array,
                                start_position: tuple[float, float] = (0, 0), trajectory_speed: float = 1,
                                amplitude: float = 1, frequency: float = 1, phase_shift: float = 0) -> np.array:
    total_steps = len(simulation_steps)
    start_position = np.array(start_position)

    # Generate the sine wave trajectory
    for i in range(total_steps):
        time = simulation_steps[i]/100 # Convert simulation step to time (assuming 100 ms per step)

        # Time determines distance between points in the sine wave
        target_positions[i, 0] = start_position[0] + trajectory_speed * time
        target_positions[i, 1] = start_position[1] + amplitude * np.sin(trajectory_speed * frequency * time + phase_shift)

    return target_positions

def generate_noisy_sine_trajectory(target_positions: np.array, simulation_steps: np.array,
                                start_position: tuple[float, float] = (0, 0), trajectory_speed: float = 1,
                                amplitude: float = 1, frequency: float = 1, phase_shift: float = 0,
                                noise_mean: float = 0, noise_sigma: float = 0) -> np.array:
    total_steps = len(simulation_steps)
    start_position = np.array(start_position)

    # Generate the sine wave trajectory with noise
    for i in range(total_steps):
        time = simulation_steps[i]/100 # Convert simulation step to time (assuming 100 ms per step)

        # Time determines distance between points in the sine wave
        target_positions[i, 0] = start_position[0] + trajectory_speed * time + generate_noise(noise_mean, noise_sigma)
        target_positions[i, 1] = start_position[1] + amplitude * np.sin(trajectory_speed * frequency * time + phase_shift) + generate_noise(noise_mean, noise_sigma)

    return target_positions

def generate_noise(noise_mean: float, noise_sigma: float) -> float:
    return np.random.uniform(-noise_sigma, noise_sigma) + noise_mean
    # return np.random.normal(noise_mean, noise_sigma)

# region UNUSED
def compute_quadratic_potential(relative_position: np.array, relative_velocity: np.array, 
                                    lambda_: float) -> float:
    return 0.5 * lambda_ * (utils.magnitude(relative_position) ** 2) # Compute quadratic potential
# endregion