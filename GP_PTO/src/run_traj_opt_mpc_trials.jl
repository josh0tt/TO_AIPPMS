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

@everywhere function run_traj_opt_mpc(gpm, replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging, true_interp_map, rng, trial_num)

    i = 1 # counter to indicate when it's time to replan
    t = 1
    map_size = size(true_interp_map)

    tm = TrajectoryManager(x0, xf, h, N, CornerInitializer());
    tm.N = N - sample_cost*init_num_drills
    tm.rng = rng
    tm.x0 += [abs(rand(tm.rng, Normal(0, 0.01))), abs(rand(tm.rng, Normal(0, 0.01)))]
    tm.barrier_cost=100000.0
    tm.sample_cost = sample_cost

    xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions, νₚ_best = pto_trajectory(gpm, tm, verbose=verbose, logging=logging, max_iters=max_iters)
    ####################################
    # Keep GP_PTO running instead of taking the best traj
    best_xd = xd
    best_ud = ud
    best_sample_actions = sample_actions
    ####################################
    q = objective == "expected_improvement" ? 4 : 2
    νₚ_best = query_no_data(f_prior)[q]

    ν_storage = zeros(bins_x+1, bins_y+1, 1+convert(Int, floor(N/replan_rate)))
    xd_storage = []
    sample_action_storage = -1 .* ones(N+1, 1+convert(Int, floor(N/replan_rate)))

    executed_drill_locations = []
    executed_traj = []
    traj_length = []
    t_indexes = []

    post_GP = f_prior

    gp_X = [] 
    gp_y = []
    gp_ν = []

    step_idx = 1
    steps_taken = 0
    Tb = N - sample_cost*length(best_sample_actions) - steps_taken
    while steps_taken < N
        @show steps_taken
        if i >= replan_rate
            x = [best_xd[i][1] for i in 1:length(best_xd)]
            y = [best_xd[i][2] for i in 1:length(best_xd)]
            sample_actions_xy = [convert_perc2xy(gpm, tm, best_sample_actions[i], best_xd) for i in 1:length(best_sample_actions)]
	        drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]

            xd_storage = vcat(xd_storage, [best_xd]) 
            sample_action_storage[1:length(drill_idx), step_idx] = drill_idx
            ν_storage[:,:, step_idx] = νₚ_best
            
            if any(i .== drill_idx) # the next location is a drill
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
                ν_samp = σ_drill
                post_GP = posterior(post_GP, [x_samp] , [y_samp], [gpm.σ_drill])
                append!(executed_drill_locations, [[x_samp[1], x_samp[2]]])
                deleteat!(sample_actions_xy, first(findall(x->x==i, drill_idx))) # remove the sample location that was executed
                
                steps_taken += sample_cost + 1 # you move and take a drill
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
                ν_samp = σ_spec^2
                # y_samp = GP.m([x[i],y[i]])
                # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
                # for normal dist whereas our GP setup uses σ²_n
                post_GP = posterior(post_GP, [x_samp] , [y_samp], [gpm.σ_spec^2])

                steps_taken += 1 
            end
    
            gp_X = vcat(gp_X, [x_samp])
            gp_y = vcat(gp_y, [y_samp])
            gp_ν = vcat(gp_ν, [ν_samp])


            gpm.GP = post_GP 
            x0 = best_xd[replan_rate+1]
            Tb = N - sample_cost*length(sample_actions_xy) - steps_taken
            append!(executed_traj, [best_xd[1]])
            append!(traj_length, length(best_xd))
            append!(t_indexes, t)
            
            tm.x0 = x0
            tm.N = Tb
            @show tm.N
            @show best_sample_actions
            
            if Tb <= 1
                break
            end 

            #######################################
            # USING PREVIOUS TRAJ AS INITIALIZATION
            #######################################
            xd0 = best_xd[(replan_rate+1):end]
            ud0 = best_ud[(replan_rate+1):end]
            # convert to xy, shorten traj, then convert back
            sample_actions0 = length(sample_actions_xy) > 0 ? convert_xy2perc(gpm, tm, sample_actions_xy, xd0) : Float64[]
            xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions, νₚ_best = pto_trajectory(gpm, tm, xd0, ud0, sample_actions0; verbose=verbose, logging=logging, max_iters, gps_crit=0.0, dd_crit=1e-6)
            
            ####################################
            # Keep GP_PTO running instead of taking the best traj
            best_xd = xd
            best_ud = ud
            best_sample_actions = sample_actions
            ####################################
            drill_added = length(best_ud) < Tb ? true : false

            #######################################
            # REINITIALIZING TRAJ
            #######################################
            # @show tm
            # @show t 
            # @show length(best_ud)
            # @show length(best_xd)
            # @show best_sample_actions 
            # tm = TrajectoryManager(x0, xf, h, Tb, CornerInitializer());
            # tm.barrier_cost=1000.0
            # tm.sample_cost = sample_cost
            # sample_actions0 = [1/((3-length(executed_drill_locations) + 1)) * i for i in 1:(3-length(executed_drill_locations))]
            # xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions, νₚ_best = pto_trajectory(gpm, tm, sample_actions0; verbose=true, logging=false, max_iters, gps_crit=0.0, dd_crit=1e-6)

            i = 1
            q = objective == "expected_improvement" ? 4 : 2 
            νₚ_best = query(post_GP)[q]

        else
            i += 1
            drill_added = false
        end
        t = drill_added ? N - length(best_ud) : N - length(best_ud) + 1
        step_idx += 1
    end

    # Finish executing the trajectory (2 steps)
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

    xd_storage = vcat(xd_storage, [best_xd]) #xd_storage[1:length(best_xd), convert(Int, floor(t/replan_rate))] = best_xd
    sample_action_storage[1:length(drill_idx), convert(Int, floor(t/replan_rate))] = drill_idx
    ν_storage[:,:, convert(Int, floor(t/replan_rate))] = νₚ_best
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
    values, t = @timed run_traj_opt_mpc(GaussianProcessManagerR2(d, GP, σ_drill, σ_spec), replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging, true_interp_map, rng, i)
    executed_traj, executed_drill_locations, gp_X, gp_y, gp_ν = values
    planning_time[i] = t
    num_actions[i] = length(executed_traj)

    x = [executed_traj[i][1] for i in 1:length(executed_traj)]
    y = [executed_traj[i][2] for i in 1:length(executed_traj)]
    drill_idx = [find_nearest_traj_drill_pts(x, y, executed_drill_locations, i) for i in 1:length(executed_drill_locations)]

    JLD2.save(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld", "final_gp_X", gp_X)
    JLD2.save(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld", "final_gp_y", gp_y)
    JLD2.save(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld", "final_gp_ν", gp_ν)
    JLD2.save(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld", "executed_traj", executed_traj)
    JLD2.save(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld", "executed_drill_locations", drill_idx)

end 

avg_planning_time = sum(planning_time)/sum(num_actions)
@show avg_planning_time

