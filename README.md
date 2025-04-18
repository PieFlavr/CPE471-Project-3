# Artificial Potential Fields

## Overview

This project implements an Artificial Potential Field Controller primarily used to track a virtual target for the robot agent to follow.

## Project Goals

1) [X] Design/Implement an Artificial Potential Controller to allow a mobile robot to follow/track a virtual moving target.
2) [X] Write Matlab, Python, or C/Cpp code to implement your designed potential controller
3) [X] Write a report of the project to cover the following items:
    * [X] (50 points total) Noise free environment (robot is assumed to localize itself accurately without noise, and be able to sense target position accurately).
        * [X] **(25 points) Plan the target to move in a linear/line trajectory and plot the tracking results:**
        1. [X] trajectories of the target and robot
        2. [X] tracking error between the target and robot
        3. [X] robot’s heading
        4. [X] robot’s velocity
        * [X] **(25 points) Plan the target to move in the sine wave trajectory and plot the tracking results:**
        1. [X] trajectories of the target and robot
        2. [X] tracking error between the target and robot
        3. [X] robot’s heading
        4. [X] robot’s velocity
    * [X] (50 points total) Noisy environment (robot is assumed to be able to sense the target, but with noise ). You can use Gaussian noise model (randn function), a similar noise function in the project 1.
        * [X] **(25 points) Plan the target to move in a linear/line trajectory and plot the tracking results:**
        1. [X] trajectories of the target and robot
        2. [X] tracking error between the target and robot
        3. [X] robot’s heading
        4. [X] robot’s velocity
        * [X] **(25 points) Plan the target to move in the sine wave trajectory and plot the tracking results:**
        1. [X] trajectories of the target and robot
        2. [X] tracking error between the target and robot
        3. [X] robot’s heading
        4. [X] robot’s velocity
    * [X] Put all source code/software in the Appendix with instruction of running the code.

## Features

1) [X] Generates Animation Display
    * **DEFAULT**: blocks until user closes figure windows
    * **DISABLING BLOCKING/SHOWING**: modify `show_figures` and `block_figures`
    * **ANIMATION FPS**: modify `fps`
2) [X] Modularized Trajectory Generation (circular, line, sine, noisy, not noisy)
    * **DEFAULT**: by default, runs only line and sine both noisy not noisy
    * **SPECIFYING SIMULATION TYPES**: must modify `noise_simulation_set` and `trajectory_simulation_set` to change
3) [X] Implemented Error/Distance, Velocity (robot), and Heading Tracking (relative, agent, target)
4) [X] Implemented Plotting of Above Statistics
5) [X] Implemented Proper File Saving of GIFs and Figures
    * **DEFAULT**: is saved to `training_data` in appropriate subfolders and enabled
    * **FILE NAMES**: can also be modified, though not recommended, via `noise_simulation_set_names` and `trajectory_simulation_set_names`

## Project Submission Files

The data files analyzed in the report are organized in a folder named `project_data`. The report itself is a .pdf file in the main directory named `Pinto CPE471 - Project 3 Report.pdf`. The `project_data` folder includes includes all the generated data used in the report. The `training_data` is where data generated by running the code is written to.

## Dependencies

This project requies a **Python** version of 3.12.9 or higher to run.
You are going to need the following dependencies for this project. Whether older version work or not are unknown as they have not been tested, but these were the versions used when developing this.

* **NumPy**: 2.2.2 or higher
* **Matplotlib**: 3.10.0 or higher

From the main directory, you can simply install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the project from the main directory, just run the following command

```bash
python src/__main__.py
```

Once the projects starts, by default, it will in sequence:

1) Run the following simulations as follows:
    1. No Noise Linear
    2. No Noise Sine
    3. Noisy Linear
    4. Noisy Sine
2) For each simulation, in order performs the following:
    1. Generates Trajectory Animation
    2. Generates Error Distance Plot
    3. Saves Error Distance Plot
    4. Generates Velocity Plot
    5. Saves Velocity Plot
    6. Generates Heading Plot
    7. Saves Heading Pot
    8. Regenerates Trajectory Animation
    9. Saves Trajectory Animation

## Configuration

The main configuration settings for the Grid World environment and reinforcement learning algorithms can be found in the `main` function, namely as the `CONSTANT HYPERPARAMETERS`. You can modify these settings which are found in the `main` function to suit your specific requirements.
