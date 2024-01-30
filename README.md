# Trajectory Optimization for Adaptive Informative Path Planning with Multimodal Sensing

<!-- ![](https://github.com/josh0tt/TO_AIPPMS/blob/main/img/main_figure.jpg) -->
<p align="center">
  <img alt="Main" src="https://github.com/josh0tt/TO_AIPPMS/blob/main/img/main_figure.jpg" width="65%">
</p>

# Description
Repository for the IEEE-CoDIT 2024 submission "Trajectory Optimization for Adaptive Informative Path Planning with Multimodal Sensing." 

We consider the problem of an autonomous agent equipped with multiple sensors, each with different sensing accuracy and energy costs. The agent's goal is to explore the environment and gather information subject to its resource constraints in unknown, partially observable environments. The challenge lies in reasoning about the effects of sensing and movement while respecting the agent's resource and dynamic constraints. We formulate the problem as a trajectory optimization problem and solve it using a projection-based trajectory optimization approach where the objective is to reduce the variance of the Gaussian process world belief. Our approach outperforms previous approaches in long horizon trajectories by achieving an overall variance reduction of up to 85% and reducing the root-mean square error in the environment belief by 50%

# Overview
We directly compare our approach with five other methods.
1. **MCTS-DPW.** MCTS-DPW with Gaussian process beliefs which was able to significantly outperform previous AIPPMS approaches.[^1] [^2]   


2. **Ergodic.** We directly compare with an ergodic control approach using projection-based trajectory optimization as introduced by Miller and adapted by Dressel.[^3][^4] We use evenly spaced drill measurements since the ergodic metric does not explicitly consider multimodal sensing capabilities.


3. **Ergodic-GP.** The Ergodic Gaussian process (Ergodic-GP) method is a modified ergodic control approach that also uses a Gaussian process to update the information distribution as samples are received online. The updated information distribution is then used to plan an ergodic trajectory at the next planning iteration. This approach allows the agent to combine the ergodic metric for trajectory generation with the Gaussian process for drill site selection.
    
4. **GP-PTO (our contribution).** The Gaussian Process Projection-based Trajectory Optimization (GP-PTO) method is our contribution outlined in the previous section. 

5. **GP-PTO Offline (our contribution).** The Gaussian Process Projection-based Trajectory Optimization Offline (GP-PTO Offline) method is a slight modification of the GP-PTO method that simply allows the optimization to run longer and then executes the best trajectory without replanning online. 

7. **Random.** The random method was implemented as a baseline. At each step, the agent chooses randomly from a set of actions. The action space is constrained so that the agent will always be able to reach the goal location within the specified energy resource constraints.  

<p align="center">
  <img alt="Variance" src="https://github.com/josh0tt/TO_AIPPMS/blob/main/img/all_trajectories.jpg" width="100%">
</p>

# Instructions
1. Specify your desired `path_name` inside parameters.jl for example `/Users/data/results/`
2. Run the following command to create the simulated environment maps:
```
julia build_maps.jl
```

3. This has been tested with Julia 1.9.0. You will need to make sure you have installed all of the Julia packages that are used. For example, Distributed, Zygote, Optim, JLD2, etc. see the [Julia documentation](https://docs.julialang.org/en/v1/stdlib/Pkg/) for more details.

# Directory Structure
**ErgodicControl**: contains the files for the ergodic control approach adapted from [ErgodicControl.jl](https://github.com/dressel/ErgodicControl.jl). Experiments can be run with:
```
julia run_ergodic_opt.jl
```
**ErgodicGP** contains the files for the Ergodic-GP approach. Experiments can be run with: 
```
julia run_traj_opt_mpc_trials.jl
``` 
**GP_PTO**: contains the files for our contribution. Experiments can be run with: 
```
julia run_traj_opt_mpc_trials.jl
```
**SBO_AIPPMS**: contains the files for MCTS-DPW.[^2] From within /SBO_AIPPMS/GP_BMDP_Rover, experiments can be run with:
```
julia Trials_RoverBDMP.jl
```

# Examples
Visualization of the GP-PTO iterative optimization process for two different intializations and random seeds. The trajectory is shown in magenta, drill locations are shown in blue, and the contour plot shows the Gaussian process variance. 

<p align="center">
  <img alt="Variance" src="https://github.com/josh0tt/TO_AIPPMS/blob/main/img/traj1.gif" width="65%">
</p>

This visualization was made by running `run_traj_opt.jl` and then `traj_gif.jl` from `GP_PTO/src/`.

# References

[^1]: > Shushman Choudhury, Nate Gruver, and Mykel J. Kochenderfer. "Adaptive informative path planning with multimodal sensing." Proceedings of the International Conference on Automated Planning and Scheduling. Vol. 30. 2020.

[^2]: > Joshua Ott, Edward Balaban, and Mykel. J. Kochenderfer, “Sequential Bayesian optimization for adaptive informative path planning with multimodal sensing,” in Proceedings of the IEEE International Conference on Robotics & Automation (ICRA), 2023.

[^3]: > Louis Dressel and Mykel J. Kochenderfer, “On the Optimality of Ergodic Trajectories for Information Gathering Tasks,” in American Control Conference (ACC), IEEE, 2018, pp. 1855–1861.

[^4]: > Lauren M. Miller and Todd D. Murphey, “Trajectory optimization for continuous ergodic exploration,” in American Control Conference (ACC), IEEE, 2013, pp. 4196–4201.

