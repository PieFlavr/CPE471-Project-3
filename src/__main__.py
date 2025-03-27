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
        simulation_time = 5*100 # Total simulation time (ms)
        lambda_ = 8.5 # Scaling factor of positive potential fields
        max_agent_velocity = 50 # Maximum velocity of agent (m/s)

        # Noise Parameters
        noise_mean = 0 # Mean of noise, default centered at 0 (Gaussian)
        noise_sigma = 0.5 # Standard deviation of noise
        do_noise = 1 # Flag to enable noise, 0 = no noise, 1 = noise

        """
        ====================================================================================================
        DERIVED HYPERPARAMETERS
        ====================================================================================================
        """
        # Simulation Step Parameters
        simulation_steps = range(0, simulation_time, int(delta_time*100)) # Run time for simulation (ms)
        relative_heading = np.array(array_duplicate(0, len(simulation_steps)), np.float64) # Initialize potential field per simulation_step

        """
        ====================================================================================================
        AGENT AND TARGET PARAMETERS
        ====================================================================================================
        """
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
        
        """
        ====================================================================================================
        MAIN PROGRAM
        ====================================================================================================
        """ 

        """
        ====================================================================================================
        TRAJECTORY GENERATION
        ====================================================================================================
        """ 
        # WITHOUT NOISE
        ## Setting the Virtual Target Trajectory 
        target_position = generate_circular_trajectory(target_position, simulation_steps,
                                                        circle_center=(60, 30), circle_radius=15, 
                                                        trajectory_speed=None)
        
        # WITHOUT NOISE
        # WITH NOISE
        # target_position = generate_noisy_circular_trajectory(target_position, simulation_steps,
        #                                                     circle_center = (60,30), circle_radius = 15, 
        #                                                     trajectory_speed = 1, noise_mean = noise_mean, noise_sigma = noise_sigma)

        
        for i in range (1, len(simulation_steps)):
            # time = (simulation_steps[i])/100 # Convert time to seconds
            time = delta_time # Using time as a step as opposed to total time

            """
            ====================================================================================================
            COMPUTING TARGET STATES
            ====================================================================================================
            """ 
            target_diff[i,:] = target_position[i,:] - target_position[i-1,:] # Compute target difference
            target_theta[i] = np.arctan2(target_diff[i,1], target_diff[i,0]) # "back" compute target heading
            target_velocity = magnitude(target_diff[i,:])                    # Compute target velocity
            
            """
            ====================================================================================================
            COMPUTING AGENT TARGET POTENTIAL STATES
            ====================================================================================================
            """
            # Compute known position and velocity
            agent_position[i,:] = agent_position[i-1,:] + np.array([
                agent_velocity[i-1] * np.cos(agent_theta[i-1]) * time,
                agent_velocity[i-1] * np.sin(agent_theta[i-1]) * time
            ]) + (do_noise * np.array([generate_noise(noise_mean, noise_sigma), generate_noise(noise_mean, noise_sigma)]))
            # print("Agent Position: ", agent_position[i,:], "Agent Velocity: ", agent_velocity[i-1], "Agent Theta: ", agent_theta[i-1])
            
            # Compute relative states
            relative_position[i,:] = target_position[i,:] - agent_position[i,:] + (do_noise * np.array([generate_noise(noise_mean, noise_sigma), generate_noise(noise_mean, noise_sigma)])) # Compute relative position
            relative_velocity[i,:] = [(target_velocity * np.cos(target_theta[i])) # Compute relative velocities
                                        - (agent_velocity[i-1] * np.cos(agent_theta[i-1])),
                                    (target_velocity * np.sin(target_theta[i]))
                                        - (agent_velocity[i-1] * np.sin(agent_theta[i-1]))]
            relative_heading[i] = np.arctan2(relative_position[i,1], relative_position[i,0]) # Compute relative heading
            # Compute desired states
            desired_agent_velocity = math.sqrt(target_velocity
                                                + (2* lambda_ 
                                                    * magnitude(relative_position[i,:]) * target_velocity
                                                        * abs(math.cos(target_theta[i]))
                                                + (lambda_**2) * (magnitude(relative_position[i,:])**2))) # Compute desired velocity magnitude
            agent_velocity[i] = min(desired_agent_velocity, max_agent_velocity) # Compute agent velocity
            desired_agent_theta = relative_heading[i] + math.asin((target_velocity
                                                                    * math.sin(target_theta[i] - agent_theta[i-1]))
                                                                / (agent_velocity[i] if agent_velocity[i] >= target_velocity else target_velocity)) # Compute desired heading
            agent_theta[i] = desired_agent_theta # Compute agent heading
        
        plot, animation = animate_field(agent_position, target_position,
                      title="Agent and Target Trajectories", 
                      subtitle="Trajectories of Agent and Target",
                      figsize=(10,10), fps = 48)
        plot.show(block=True)
        print("Saving Animation...")
        animation.save("project.gif", writer='pillow', fps=48)
        print("Animation Saved!")
    pass

main()