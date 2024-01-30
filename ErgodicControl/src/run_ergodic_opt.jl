using Distributed
include("ErgodicControl.jl")
# using .ErgodicControl
using Plots
ENV["GKSwstype"]="nul" 

using KernelFunctions
using LinearAlgebra
using JLD2
using Random
using Distributions
using Interpolations
include("../../CustomGP.jl")
include("../../parameters.jl")
include("plot_ergodic.jl")

function convert_perc2xy(sample_action, xd)
	if sample_action >= 1.0
		return xd[end]
	elseif convert(Int, floor(length(xd)*sample_action)) <= 1
		return xd[1]
	end

	x_hist = [xd[i][1] for i in 1:length(xd)]
	y_hist = [xd[i][2] for i in 1:length(xd)]
	p = [i/length(xd) for i in 1:length(xd)]
	interp_linear_x = linear_interpolation(p, x_hist,extrapolation_bc=Line())
	interp_linear_y = linear_interpolation(p, y_hist,extrapolation_bc=Line())
	sa_x = interp_linear_x(sample_action)
	sa_y = interp_linear_y(sample_action)
	sample_action_xy = [sa_x, sa_y]

	return sample_action_xy
end

function find_nearest_traj_drill_pts(x_hist, y_hist, sample_actions_xy, action_idx)
	dist_to_drill_pt = [norm(sample_actions_xy[action_idx] - [x_hist[i], y_hist[i]]) for i in 1:length(x_hist)]
	min_dist_idx = argmin(dist_to_drill_pt)

	return min_dist_idx
end

function generate_ergodic_samples(xd, sample_actions, true_interp_map, rng)
    x = [xd[i][1] for i in 1:length(xd)]
    y = [xd[i][2] for i in 1:length(xd)]
    sample_actions_xy = [convert_perc2xy(sample_actions[i], xd) for i in 1:length(sample_actions)]
    drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]

    map_size = size(true_interp_map)

    gp_X = [] 
    gp_y = []
    gp_ν = []

    for i in 1:length(xd)

        if any(drill_idx .== i)
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
        end
        gp_X = vcat(gp_X, [x_samp])
        gp_y = vcat(gp_y, [y_samp])
        gp_ν = vcat(gp_ν, [ν_samp])
    end

    return gp_X, gp_y, gp_ν, drill_idx
end

function run_ergodic_opt(K, em, replan_rate, N, h, sample_cost, phi)

    R = 0.1*[1 0; 0 1]#0.01 * [1 0; 0 1]
    planning_time = zeros(1, num_trials)
    num_actions = zeros(1, num_trials)

    for i = 1:num_trials
        @show i

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

        Tb = N - sample_cost*init_num_drills #- steps_taken
        # tm = TrajectoryManager(x0, xf, h, Tb, ConstantInitializer([0.0,0.0]));
        tm = TrajectoryManager(x0, xf, h, Tb, CornerInitializer());
        tm.rng = MersenneTwister(1234+i)
        tm.x0 += [abs(rand(tm.rng, Normal(0, 0.01))), abs(rand(tm.rng, Normal(0, 0.01)))]
        tm.barrier_cost=1000.0
        tm.sample_cost = sample_cost
        xd, ud = pto_trajectory(em, tm, verbose=false, logging=true);

        em = ErgodicManagerR2(em.domain, phi, K, [bins_x+1, bins_y+1])

        replan_flag = true
        while replan_flag
            tm.R = R 
            values, t = @timed pto_trajectory(em, tm, verbose=true, logging=true, max_iters=500);
            xd, ud = values
            planning_time[i] += t

            ud_mag = [sqrt(ud[i][1]^2 + ud[i][2]^2) for i in 1:length(ud)]
    
            if maximum(ud_mag) > ud_max
                @show tm.R 
                println("exceeded control input")
                # double the control effort penalty 
                R = 2 .* R
            else
                gp_X, gp_y, gp_ν, drill_idx = generate_ergodic_samples(xd, [0.25, 0.5, 0.75], true_interp_map, tm.rng)

                JLD2.save(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld", "final_gp_X", gp_X)
                JLD2.save(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld", "final_gp_y", gp_y)
                JLD2.save(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld", "final_gp_ν", gp_ν)
                JLD2.save(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld", "executed_traj", xd)
                JLD2.save(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld", "executed_drill_locations", drill_idx)
            
                replan_flag = false
            end
        end
        num_actions[i] = length(xd)
    end
    avg_planning_time = sum(planning_time)/sum(num_actions)
    @show avg_planning_time
end

d = Domain(domain_min, domain_max, domain_bins)
map_size = map_size_sboaippms
run_ergodic_opt(K_fc, ErgodicManagerR2(d, phi, K_fc, [bins_x+1, bins_y+1]), replan_rate, N, h, sample_cost, phi)
