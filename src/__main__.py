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
        CONSTANT HYPERPARAMETERS
        ====================================================================================================
        """
        # General Paramters
        dims = 2 # Number of dimension for abitrary world
        delta_time = 0.05 # Fixed delta time step for simulation (s)
        simulation_time = 10*100 # Total simulation time (ms)
        lambda_ = 8.5 # Scaling factor of positive potential fields
        max_agent_velocity = 50 # Maximum velocity of agent (m/s)

        # Noise Parameters
        noise_mean = 0.5 # Mean of noise
        noise_sigma = 0.5 # Standard deviation of noise

        # Miscallaneous Parameters
        time_scaling = 1.495 # Scaling factor for time applied when converting (ms to s)

        """
        ====================================================================================================
        DERIVED HYPERPARAMETERS
        ====================================================================================================
        """
        # Simulation Step Parameters
        simulation_steps = range(0, simulation_time, int(delta_time*100)) # Run time for simulation (ms)
        artificial_error = np.array(array_duplicate(0, len(simulation_steps)), np.float64) # Initialize artificial error per simulation_step
        phi = np.array(array_duplicate(0, len(simulation_steps)), np.float64) # Initialize potential field per simulation_step

        """
        ====================================================================================================
        AGENT AND TARGET PARAMETERS
        ====================================================================================================
        """
        # Virtual Target Parameters
        target_position = np.array(array_duplicate([0,0], len(simulation_steps)), np.float64) # Initialize virtual target positions at each time step
        target_velocity = 1.2 # Set velocity of virtual target (m/s), for this project, constant!
        target_theta = np.array(array_duplicate(0, len(simulation_steps)), np.float64) # Initialize virtual target heading at each time step
        target_diff = np.array(array_duplicate([0,0], len(simulation_steps)), np.float64) # Initialize virtual target difference at each time step
        
        # Agent Parameters
        agent_position = np.array(array_duplicate([0,0], len(simulation_steps)), np.float64) # Initialize agent positions at each time step
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
        
        """
        ====================================================================================================
        MAIN PROGRAM
        ====================================================================================================
        """ 
        
        for i in range (1, len(simulation_steps)):
            time = (simulation_steps[i] * time_scaling)/100 # Convert time to seconds

            """
            ====================================================================================================
            CIRCULAR TRAJECTORY
            ====================================================================================================
            """ 
            # WITHOUT NOISE
            ## Setting the Virtual Target Trajectory 
            target_position_x = 60 - 15 * np.cos(time)
            target_position_y = 30 + 15 * np.sin(time)
            target_position[i,:] = [target_position_x, target_position_y] 

            # WITH NOISE
            # target_position_x = 60 - 15 * np.cos(time) + np.random.uniform(-noise_mean, noise_sigma) + noise_mean
            # target_position_y = 30 + 15 * np.sin(time) + np.random.uniform(-noise_mean, noise_sigma) + noise_mean
            # target_position[i,:] = [target_position_x, target_position_y] 

            """
            ====================================================================================================
            COMPUTING TARGET HEADING
            ====================================================================================================
            """ 
            target_diff[i,:] = target_position[i,:] - target_position[i-1,:] # Compute target difference
            target_theta[i] = np.arctan2(target_diff[i,1], target_diff[i,0]) # "back" compute target heading
            target_velocity = magnitude(target_diff[i,:])                    # Compute target velocity
    pass

main()