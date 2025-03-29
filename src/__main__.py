"""
__main__.py

Description: 
Author:  
Date: 

"""

from __init__ import * # Import everything from the __init__.py file

def main():
    """
    Main function to run the application.
    """

    if __name__ == "__main__":
        print("Hello, World!")  

        """
        ====================================================================================================
        CONSTANT HYPERPARAMETERS       OOO   USER MODIFIABLE   OOO
        ====================================================================================================
        """
        # Simulation Parameters
        enable_saving = 1 # Flag to enable saving, 0 = no save, 1 = save
        block_figures = False # Flag to enable blocking figures, 0 = no block, 1 = block
        show_figures = False # Flag to enable showing figures, 0 = no show, 1 = show
        fps = 120 # Frames per second for animation
        base_file_path = 'training_data' # Base file path for saving figures and animations

        noise_simulation_set = (0, 1) # Range of noise types to simulate, 0 = no noise, 1 = noise
        trajectory_simulation_set = (1, 2) # Range of trajectory types to simulate, 0 = circular, 1 = linear, 2 = sine

        noise_simulation_set_names = { 0: "No_Noise", 1: "Noise"}
        trajectory_simulation_set_names = { 0: "Circular", 1: "Linear", 2: "Sine"}

        # General Paramters
        dims = 2 # Number of dimension for abitrary world
        delta_time = 0.05 # Fixed delta time step for simulation (s)
        simulation_time = 10*100 # Total simulation time (ms)
        lambda_ = 8.5 # Scaling factor of positive potential fields
        max_agent_velocity = 50 # Maximum velocity of agent (m/s)

        # Noise Parameters
        noise_mean = 0 # Mean of noise, default centered at 0 (Gaussian)
        noise_sigma = 0.5 # Standard deviation of noise

        """
        ====================================================================================================
        DERIVED HYPERPARAMETERS
        ====================================================================================================
        """
        # Simulation Step Parameters
        simulation_steps = range(0, simulation_time, int(delta_time*100)) # Run time for simulation (ms)
        relative_theta = np.array(array_duplicate(0, len(simulation_steps)), np.float64) # Initialize potential field per simulation_step

        """
        ====================================================================================================
        MAIN PROGRAM
        ====================================================================================================
        """ 
        print("Running Simulations...")
        
        sim_noise_type_range = noise_simulation_set
        sim_trajectory_type_range = trajectory_simulation_set

        for noise_type in sim_noise_type_range:
            for trajectory_type in sim_trajectory_type_range:
                print(f"Running Simulation with Noise Type: {noise_type}, Trajectory Type: {trajectory_type}...")
                """
                ====================================================================================================
                AGENT AND TARGET PARAMETERS RESET
                ====================================================================================================
                """
                # region PARAMETER_RESET
                print("Resetting Agent and Target Parameters...")
                # Virtual Target Parameters
                target_position = np.array(array_duplicate([0,0], len(simulation_steps)), np.float64) # Initialize virtual target positions at each time step (m)
                target_velocity = 1.2 # Set velocity of virtual target (m/s), for this project, constant!
                target_theta = np.array(array_duplicate(0, len(simulation_steps)), np.float64) # Initialize virtual target heading at each time step
                target_diff = np.array(array_duplicate([0,0], len(simulation_steps)), np.float64) # Initialize virtual target difference at each time step
                # target_diff is solely for velocity and theta computation
                
                # Agent Parameters
                agent_position = np.array(array_duplicate([0,0], len(simulation_steps)), np.float64) # Initialize agent positions at each time step (m)
                agent_velocity = np.array(array_duplicate(0,len(simulation_steps)), np.float64) # Initialize agent velocity at each time step
                agent_theta = np.array(array_duplicate(0, len(simulation_steps)), np.float64) # Initialize agent heading at each time step

                # Relative State Parameters
                relative_position = np.array(array_duplicate([0,0], len(simulation_steps)), np.float64) # Initialize relative position at each time step
                relative_velocity = np.array(array_duplicate([0,0], len(simulation_steps)), np.float64) # Initialize relative velocity at each time step

                # Initial Compute of Relative States
                relative_position[0,:] = target_position[0,:] - agent_position[0,:] # Compute initial relative position
                relative_velocity[0,:] = [(target_velocity * np.cos(target_theta[0])) # Compute initial relative velocities
                                                - (agent_velocity[0] * np.cos(agent_theta[0])),
                                            (target_velocity * np.sin(target_theta[0]))
                                                - (agent_velocity[0] * np.sin(agent_theta[0]))]
                
                # Other Tracked States
                error = np.array(array_duplicate(0, len(simulation_steps)), np.float64) # Initialize error tracking at each time step
                #endregion PARAMETER_RESET

                """
                ====================================================================================================
                TRAJECTORY GENERATION
                ====================================================================================================
                """ 
                # region TRAJECTORY_GENERATION
                if (noise_type == 1):
                    print("Noise Enabled!")
                    match trajectory_type:
                        case 0: # Circular
                            target_position = generate_noisy_circular_trajectory(target_position, simulation_steps,
                                                                                circle_center=(60, 30), circle_radius=15, 
                                                                                trajectory_speed=None, noise_mean=noise_mean, noise_sigma=noise_sigma)
                        case 1: # Linear
                            target_position = generate_noisy_linear_trajectory(target_position, simulation_steps,
                                                                                start_position=(0, 100), end_position=(100, 0), 
                                                                                noise_mean=noise_mean, noise_sigma=noise_sigma)
                        case 2: # Sine
                            target_position = generate_noisy_sine_trajectory(target_position, simulation_steps,
                                                                                start_position=(10, 40), trajectory_speed=5,
                                                                                amplitude=15, frequency=0.5, phase_shift=0.5,
                                                                                noise_mean=noise_mean, noise_sigma=noise_sigma)
                        case _:
                            print("Invalid trajectory type!")
                else: 
                    print("Noise Disabled!")
                    match trajectory_type:
                        case 0: # Circular
                            target_position = generate_circular_trajectory(target_position, simulation_steps,
                                                                            circle_center=(60, 30), circle_radius=15, 
                                                                            trajectory_speed=None)
                        case 1: # Linear
                            target_position = generate_linear_trajectory(target_position, simulation_steps,
                                                                        start_position=(0, 100), end_position=(100, 0))
                        case 2: # Sine
                            target_position = generate_sine_trajectory(target_position, simulation_steps,
                                                                        start_position=(10, 40), trajectory_speed=5,
                                                                        amplitude=15, frequency=0.5, phase_shift=0.5)
                # endregion TRAJECTORY_GENERATION
                
                """
                ====================================================================================================
                MAIN SIMULATION SEQUENCE
                ====================================================================================================
                """ 

                for i in range (1, len(simulation_steps)):
                    # time = (simulation_steps[i])/100 # Convert time to seconds
                    time = delta_time # Using time as a step as opposed to total time

                    """
                    ====================================================================================================
                    COMPUTING TARGET STATES
                    ====================================================================================================
                    """ 
                    # region LOGICAL_TARGET_STATE_COMPUTATION
                    target_diff[i,:] = target_position[i,:] - target_position[i-1,:] # Compute target difference
                    target_theta[i] = np.arctan2(target_diff[i,1], target_diff[i,0]) # "back" compute target heading
                    target_velocity = magnitude(target_diff[i,:])                    # Compute target velocity
                    # endregion LOGICAL_TARGET_STATE_COMPUTATION

                    """
                    ====================================================================================================
                    COMPUTING AGENT TARGET POTENTIAL STATES
                    ====================================================================================================
                    """
                    # region CONTROLLER_COMPUTATION
                    # Compute known position and velocity
                    agent_position[i,:] = agent_position[i-1,:] + np.array([
                        agent_velocity[i-1] * np.cos(agent_theta[i-1]) * time,
                        agent_velocity[i-1] * np.sin(agent_theta[i-1]) * time
                    ]) + (noise_type * np.array([generate_noise(noise_mean, noise_sigma), generate_noise(noise_mean, noise_sigma)]))
                    # print("Agent Position: ", agent_position[i,:], "Agent Velocity: ", agent_velocity[i-1], "Agent Theta: ", agent_theta[i-1])
                    
                    # Compute relative states
                    relative_position[i,:] = target_position[i,:] - agent_position[i,:] + (noise_type * np.array([generate_noise(noise_mean, noise_sigma), generate_noise(noise_mean, noise_sigma)])) # Compute relative position
                    relative_velocity[i,:] = [(target_velocity * np.cos(target_theta[i])) # Compute relative velocities
                                                - (agent_velocity[i-1] * np.cos(agent_theta[i-1])),
                                            (target_velocity * np.sin(target_theta[i]))
                                                - (agent_velocity[i-1] * np.sin(agent_theta[i-1]))]
                    relative_theta[i] = np.arctan2(relative_position[i,1], relative_position[i,0]) # Compute relative heading

                    # Compute error
                    error[i] = magnitude(relative_position[i,:]) # Compute error

                    # Compute desired states
                    desired_agent_velocity = math.sqrt(target_velocity
                                                        + (2* lambda_ 
                                                            * magnitude(relative_position[i,:]) * target_velocity
                                                                * abs(math.cos(target_theta[i]))
                                                        + (lambda_**2) * (magnitude(relative_position[i,:])**2))) # Compute desired velocity magnitude
                    agent_velocity[i] = min(desired_agent_velocity, max_agent_velocity) # Compute agent velocity
                    desired_agent_theta = relative_theta[i] + math.asin((target_velocity
                                                                            * math.sin(target_theta[i] - agent_theta[i-1]))
                                                                        / (agent_velocity[i] if agent_velocity[i] >= target_velocity else target_velocity)) # Compute desired heading
                    agent_theta[i] = desired_agent_theta # Compute agent heading
                    # endregion CONTROLLER_COMPUTATION

                """
                ====================================================================================================
                FIGURE GENERATION AND ANIMATION
                ====================================================================================================
                """ 
                # region FIGURES

                subfolder_path = None 
                if enable_saving == 1:
                    subfolder_path = os.path.join(base_file_path, noise_simulation_set_names[noise_type] + "_" + trajectory_simulation_set_names[trajectory_type])
                    if not os.path.exists(subfolder_path):
                        os.makedirs(subfolder_path, exist_ok=True)
                        print("Subfolder Created: " + subfolder_path)

                # region ANIMATION
                if show_figures:
                    trajectory, trajectory_animation = None, None
                    trajectory, trajectory_animation = animate_field(agent_position, target_position, # i hate this but it prevents issue with double plotting
                                title="Agent and Target Trajectories", 
                                subtitle=(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type]),
                                figsize=(10,10), fps = fps) 
                    
                    plt.gcf().canvas.setWindowTitle(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type] + " - Agent and Target Trajectories")
                    plt.show(block=block_figures) 
                    plt.close() if not show_figures else None
                # endregion ANIMATION
                
                # region ERROR_DISTANCE
                error_distance = plt.plot(simulation_steps, error, label="Error Distance", color="red")
                plt.xlabel("Simulation Steps (ms)")
                plt.ylabel("Error Distance (m)")
                plt.title("Error Distance vs Simulation Steps")
                plt.suptitle(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type])
                plt.legend(loc = 'upper right')
                
                if enable_saving == 1:
                    file_name = (noise_simulation_set_names[noise_type] + "_" + trajectory_simulation_set_names[trajectory_type] + "_error_distance.png")
                    file_path = os.path.join(subfolder_path, file_name)
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    print(file_name + " Figure Saved!")

                plt.gcf().canvas.setWindowTitle(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type] + " - Error Distance")
                plt.show(block=block_figures)
                plt.close() if not show_figures else None
                # endregion ERROR_DISTANCE
                
                # region AGENT_VELOCITY
                velocity = plt.plot(simulation_steps, agent_velocity, label="Agent Velocity", color="blue")
                plt.xlabel("Simulation Steps (ms)")
                plt.ylabel("Agent Velocity (m/s)")
                plt.title("Agent Velocity vs Simulation Steps")
                plt.suptitle(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type])
                plt.legend(loc = 'upper right')

                if enable_saving == 1:
                    file_name = (noise_simulation_set_names[noise_type] + "_" + trajectory_simulation_set_names[trajectory_type] + "_agent_velocity.png")
                    file_path = os.path.join(subfolder_path, file_name)
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    print(file_name + " Figure Saved!")

                plt.gcf().canvas.setWindowTitle(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type] + " - Agent Velocity")
                plt.show(block=block_figures)
                plt.close() if not show_figures else None
                # endregion AGENT_VELOCITY

                # region ALL_HEADINGS
                heading_fig, ax = plt.subplots()
                ax.plot(simulation_steps, agent_theta, label="Agent Heading", color="blue", linestyle = '-')
                ax.plot(simulation_steps, target_theta, label="Target Heading", color="red", linestyle = ':')
                ax.plot(simulation_steps, relative_theta, label="Relative Heading", color="green", linestyle = '--')
                ax.set_xlabel("Simulation Steps (ms)")
                ax.set_ylabel("Heading (rad)")
                ax.set_title("Heading vs Simulation Steps")
                plt.suptitle(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type])
                ax.legend(loc = 'upper right')

                if enable_saving == 1:
                    file_name = (noise_simulation_set_names[noise_type] + "_" + trajectory_simulation_set_names[trajectory_type] + "_heading.png")
                    file_path = os.path.join(subfolder_path, file_name)
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    print(file_name + " Figure Saved!")

                plt.gcf().canvas.setWindowTitle(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type] + " - Heading")
                plt.show(block=block_figures)
                plt.close() if not show_figures else None
                # endregion ALL_HEADINGS
                
                # region ANIMATION_SAVING
                if enable_saving == 1:
                    print("Regenerating Animation...")
                    file_name = (noise_simulation_set_names[noise_type] + "_" + trajectory_simulation_set_names[trajectory_type] + "_trajectory.gif")
                    file_path = os.path.join(subfolder_path, file_name)
                    trajectory = None
                    trajectory, animation = animate_field(agent_position, target_position,
                                title="Agent and Target Trajectories", 
                                subtitle=(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type]),
                                figsize=(10,10), fps = fps)
                    print("Animation Regenerated!")

                    print("Saving Animation...")
                    animation.save(file_path, writer='pillow', fps=fps)
                    plt.gcf().canvas.setWindowTitle(noise_simulation_set_names[noise_type] + ", " + trajectory_simulation_set_names[trajectory_type] + " - Agent and Target Trajectories")
                    plt.show(block=False)
                    plt.close()
                    print("Animation Saved!")
                # endregion ANIMATION_SAVING
                #endregion FIGURES
    pass

main()