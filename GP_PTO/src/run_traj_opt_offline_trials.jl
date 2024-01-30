using Distributed
@everywhere using LinearAlgebra, Random, Distributions
@everywhere using Interpolations
@everywhere using StatsBase
using Images, Plots
@everywhere using KernelFunctions
@everywhere using Zygote
using DelimitedFiles
using CSV
using StatProfilerHTML
@everywhere using Optim, LineSearches
@everywhere using JLD2
using SharedArrays

include("../../CustomGP.jl")
include("GaussianProcessControl.jl")
@everywhere include("../../parameters.jl")


println("#############################################################")
println("####################    Checking σ's   ######################")
println("#############################################################")
println("σ_drill: ", σ_drill)
println("σ_spec: ", σ_spec)

@everywhere function run_traj_opt(gpm, replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging, true_interp_map, rng, trial_num)

    i = 1 # counter to indicate when it's time to replan
    t = 1
    map_size = size(true_interp_map)
    global optim_time_limit
    optim_time_limit = 30.0

    tm = TrajectoryManager(x0, xf, h, N, CornerInitializer());
    tm.N = N - sample_cost*init_num_drills
    tm.rng = rng
    tm.x0 += [abs(rand(tm.rng, Normal(0, 0.01))), abs(rand(tm.rng, Normal(0, 0.01)))]
    tm.barrier_cost=100000.0
    tm.sample_cost = sample_cost

    max_iters = 5000#10000
    verbose = true
    xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions, νₚ_best = pto_trajectory_offline(gpm, tm, verbose=verbose, logging=logging, max_iters=max_iters)
   
    executed_drill_locations = []
    executed_traj = []
    traj_length = []
    t_indexes = []

    post_GP = f_prior

    gp_X = [] 
    gp_y = []
    gp_ν = []

    # Finish executing the trajectory
    x = [best_xd[i][1] for i in 1:length(best_xd)]
    y = [best_xd[i][2] for i in 1:length(best_xd)]
    sample_actions_xy = [convert_perc2xy(gpm, tm, best_sample_actions[i], best_xd) for i in 1:length(best_sample_actions)]
    drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]

    for i = 1:length(best_xd)
        if i in drill_idx
            x_samp = [x[i],y[i]]
            
            pos_x = Int(round(x_samp[1]*100 + 1, digits=0))
            if pos_x > map_size[2]
                pos_x = map_size[2]
            elseif pos_x < 1
                pos_x = 1
            end

            pos_y = map_size[1] - Int(round(x_samp[2]*100 + 1, digits=0))
            if pos_y > map_size[1]
                pos_y = map_size[1]
            elseif pos_y < 1
                pos_y = 1
            end

            pos = LinearIndices(map_size)[pos_y, pos_x]

            y_samp = true_interp_map[pos]
            ν_samp = gpm.σ_drill
            post_GP = posterior(post_GP, [x_samp] , [y_samp], [gpm.σ_drill])
            append!(executed_drill_locations, [[x_samp[1], x_samp[2]]])
        else
            x_samp = [x[i],y[i]]

            pos_x = Int(round(x[i]*100 + 1, digits=0))
            if pos_x > map_size[2]
                pos_x = map_size[2]
            elseif pos_x < 1
                pos_x = 1
            end

            pos_y = map_size[1] - Int(round(y[i]*100 + 1, digits=0))
            if pos_y > map_size[1]
                pos_y = map_size[1]
            elseif pos_y < 1
                pos_y = 1
            end

            pos = LinearIndices(map_size)[pos_y, pos_x]

            y_samp = true_interp_map[pos] + rand(rng, Normal(0, σ_spec))
            # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
            # for normal dist whereas our GP setup uses σ²_n
            ν_samp = gpm.σ_spec^2
            post_GP = posterior(post_GP, [x_samp] , [y_samp], [gpm.σ_spec^2])
        end

        gp_X = vcat(gp_X, [x_samp])
        gp_y = vcat(gp_y, [y_samp])
        gp_ν = vcat(gp_ν, [ν_samp])
    end
    append!(executed_traj, best_xd)

    append!(traj_length, length(best_xd))
    append!(t_indexes, t)

    return executed_traj, executed_drill_locations, gp_X, gp_y, gp_ν 

end

planning_time = SharedArray{Float64}(1, num_trials)
num_actions = SharedArray{Float64}(1, num_trials)
@sync @distributed for i = 1:num_trials
    @show i

    rng = MersenneTwister(1234+i)
    true_map = load(path_name * "/true_maps/true_map$(i).jld")["true_map"]

    plot_scale = 0:0.01:1
    true_interp_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))

    for i in 1:size(true_interp_map)[1]
        for j in 1:size(true_interp_map)[2]
            idx_i = Int(floor(i/10) + 1)
            idx_j = Int(floor(j/10) + 1)
            true_interp_map[i,j] = true_map[idx_i, idx_j]
        end
    end

    d = Domain(domain_min, domain_max, domain_bins)
    values, t = @timed run_traj_opt(GaussianProcessManagerR2(d, GP, σ_drill, σ_spec), replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging, true_interp_map, rng, i)
    executed_traj, executed_drill_locations, gp_X, gp_y, gp_ν = values
    planning_time[i] = t
    num_actions[i] = length(executed_traj)

    x = [executed_traj[i][1] for i in 1:length(executed_traj)]
    y = [executed_traj[i][2] for i in 1:length(executed_traj)]
    drill_idx = [find_nearest_traj_drill_pts(x, y, executed_drill_locations, i) for i in 1:length(executed_drill_locations)]

    JLD2.save(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld", "final_gp_X", gp_X)
    JLD2.save(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld", "final_gp_y", gp_y)
    JLD2.save(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld", "final_gp_ν", gp_ν)
    JLD2.save(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld", "executed_traj", executed_traj)
    JLD2.save(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld", "executed_drill_locations", drill_idx)

end 

avg_planning_time = sum(planning_time)/sum(num_actions)
@show avg_planning_time