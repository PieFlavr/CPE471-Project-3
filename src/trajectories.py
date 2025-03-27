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
            # Calculate the angular position based on trajectory speed
            angle = trajectory_speed * i
            
            # Ensure the angle stays within [0, 2pi]
            angle = angle % (2 * np.pi)
            
            # Update target position on the circle using parametric equations
            target_positions[i, :] = [
                circle_center[0] + circle_radius * np.cos(angle),
                circle_center[1] + circle_radius * np.sin(angle)
            ]
    
    return target_positions

def generate_noisy_circular_trajectory(target_positions: np.array, simulation_steps: np.array,
                                circle_center: tuple[float,float], circle_radius: float, 
                                trajectory_speed: float, noise_mean: float, noise_sigma: float) -> np.array:
    for i in range (1, len(simulation_steps)):
        time = simulation_steps[i]/100 # Convert time to seconds
        target_positions[i,:] = [circle_center[0] - circle_radius * np.cos(time * trajectory_speed) + np.random.uniform(-noise_mean, noise_sigma) + noise_mean,
                                circle_center[1] + circle_radius * np.sin(time * trajectory_speed) + np.random.uniform(-noise_mean, noise_sigma) + noise_mean]
    return target_positions

def generate_noise(noise_mean: float, noise_sigma: float) -> float:
    return np.random.uniform(-noise_sigma, noise_sigma) + noise_mean
    # return np.random.normal(noise_mean, noise_sigma)

# region UNUSED
def compute_quadratic_potential(relative_position: np.array, relative_velocity: np.array, 
                                    lambda_: float) -> float:
    return 0.5 * lambda_ * (utils.magnitude(relative_position) ** 2) # Compute quadratic potential
# endregion