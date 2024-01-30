using Distributed
include("../../CustomGP.jl")
include("rover_pomdp.jl")
include("belief_mdp.jl")
include("plot_RoverBMDP.jl")
using Random
using BasicPOMCP
using POMDPs
using Statistics
using Distributions
using Plots
using KernelFunctions
using MCTS
using DelimitedFiles
using JLD2

include("../../parameters.jl")

POMDPs.isterminal(bmdp::BeliefMDP, b::RoverBelief) = isterminal(bmdp.pomdp, b)

################################################################################
# Map Building
################################################################################

function get_neighbors(idx::Int, map_size::Tuple{Int, Int})
	pos = [CartesianIndices(map_size)[idx].I[1], CartesianIndices(map_size)[idx].I[2]]
	neighbors = [pos+[0,1], pos+[0,-1], pos+[1,0], pos+[-1,0]]
	bounds_neighbors = []
	for i in 1:length(neighbors)
		if inbounds(map_size, RoverPos(neighbors[i][1], neighbors[i][2]))
			append!(bounds_neighbors, [neighbors[i]])
		end
	end

	bounds_neighbors_idx = [LinearIndices(map_size)[bounds_neighbors[i][1], bounds_neighbors[i][2]] for i in 1:length(bounds_neighbors)]
	return bounds_neighbors_idx
end

function inbounds(map_size::Tuple{Int, Int}, pos::RoverPos)
    if map_size[1] >= pos[1] > 0 && map_size[2] >= pos[2] > 0
        # i = abs(s[2] - pomdp.map_size[1]) + 1
        # j = s[1]
        return true
    else
        return false
    end
end

function build_map(rng::RNG, number_of_sample_types::Int, map_size::Tuple{Int, Int}) where {RNG<:AbstractRNG}
	sample_types = collect(0:(1/number_of_sample_types):(1-1/number_of_sample_types))
	init_map = rand(rng, sample_types, map_size[1], map_size[2])
	new_map = zeros(map_size)

	p_neighbors = 0.95

	for i in 1:(map_size[1]*map_size[2])
		if i == 1
			continue
		else
			if rand(rng) < p_neighbors
				neighbor_values = init_map[get_neighbors(i, map_size)]
				new_map[i] = round(mean(neighbor_values),digits=1)
				#true_map[i] = true_map[i-1]
			else
				continue
			end
		end
	end

	return new_map
end

################################################################################
# Running Trials
################################################################################

function run_rover_bmdp(rng::RNG, bmdp::BeliefMDP, policy, isterminal::Function) where {RNG<:AbstractRNG}

    belief_state = initialstate(bmdp).val

    # belief_state = initial_belief_state(bmdp, rng)
	state_hist = [deepcopy(belief_state.pos)]
	gp_hist = [deepcopy(belief_state.location_belief)]
	action_hist = []
	reward_hist = []
	total_reward_hist = []
	total_planning_time = 0

    total_reward = 0.0
    while true
        a, t = @timed policy(belief_state)

		total_planning_time += t

        if isterminal(belief_state)
            break
        end

		#new_belief_state, sim_reward = POMDPs.gen(bmdp, belief_state, a, rng)

		####################################################################
		# This update accesses the true map AFTER the simulations have been completed
		# in the rollouts update_belief() is used which overrides the o from generate_o to only use the GP predicted mean for collecting samples
		s = rand(rng, bmdp.pomdp, belief_state)
		sp, o, r = @gen(:sp, :o, :r)(bmdp.pomdp, s, a, rng)
		new_belief_state = update_belief_no_obs_override(bmdp.pomdp, belief_state, a, o, rng)
		####################################################################

		# just use these to get the true reward NOT the simulated reward
		s = RoverState(belief_state.pos, bmdp.pomdp.true_map, belief_state.cost_expended, belief_state.drill_samples)
		sp = RoverState(new_belief_state.pos, bmdp.pomdp.true_map, new_belief_state.cost_expended, new_belief_state.drill_samples)
		true_reward = reward(bmdp.pomdp, s, a, sp)

		# if a == :drill
		# 	println("State: ", convert_pos_idx_2_pos_coord(bmdp.pomdp, belief_state.pos))
		# 	println("Cost Expended: ", belief_state.cost_expended)
		# 	println("Actions available: ", actions(bmdp.pomdp, belief_state))
		# 	println("Action: ", a)
		# 	println("True reward: ", true_reward)
		# 	println("Sim reward: ", sim_reward)
		# 	println("True Value: ", bmdp.pomdp.true_map[new_belief_state.pos])
		#
		# 	if belief_state.location_belief.X == []
	    #         μ_init, ν_init = query_no_data(belief_state.location_belief)
	    #     else
	    #         μ_init, ν_init, S_init = query(belief_state.location_belief)
	    #     end
		# 	if new_belief_state.location_belief.X == []
		# 		μ_post, ν_post = query_no_data(new_belief_state.location_belief)
		# 	else
		# 		μ_post, ν_post, S_post = query(new_belief_state.location_belief)
		# 	end
		# 	println("Mean Value Before Drill: ", μ_init[s.pos])
		# 	println("Mean Value After Drill: ", μ_post[s.pos])
		#
		# 	println("Drill Samples: ", new_belief_state.drill_samples)
		# 	println("")
		# end


        # new_state = generate_s(pomdp, state, a, rng)
        # loc_reward = reward(pomdp, state, a, new_state)
        # obs = generate_o(pomdp, state, a, new_state, rng)
		#
		# # println("Reward: ", loc_reward)
		# # println("Observation: ", obs)
		# # println("True Value: ", pomdp.true_map[new_state.pos])
		# # println("Drill Samples: ", new_state.drill_samples)
		# #
		# # # println("Particles: ", belief_state.location_belief.particles[:, new_state.pos])
		# # # println("Weights: ", belief_state.location_belief.weights[:, new_state.pos])
		# # println("")
		#
        # belief_state = update_belief(pomdp, belief_state, a, obs, rng)

		# println("New Particles: ", belief_state.location_belief.particles[:, new_state.pos])
		# println("New Weights: ", belief_state.location_belief.weights[:, new_state.pos])
		# println("")

        total_reward += true_reward
        belief_state = new_belief_state

        if isterminal(belief_state)
            break
        end
		state_hist = vcat(state_hist, deepcopy(belief_state.pos))
		gp_hist = vcat(gp_hist, deepcopy(belief_state.location_belief))
		# # location_states_hist = vcat(location_states_hist, deepcopy(state.location_states))
		action_hist = vcat(action_hist, deepcopy(a))
		reward_hist = vcat(reward_hist, deepcopy(true_reward))
		total_reward_hist = vcat(total_reward_hist, deepcopy(total_reward))


    end

    return total_reward, state_hist, gp_hist, action_hist, reward_hist, total_reward_hist, total_planning_time, length(reward_hist)

end


function get_gp_bmdp_policy(bmdp, rng, max_depth=20, queries = 100)
	planner = solve(MCTS.DPWSolver(depth=max_depth, n_iterations=queries, rng=rng, k_state=0.5, k_action=10000.0, alpha_state=0.5), bmdp)
	# planner = solve(MCTSSolver(depth=max_depth, n_iterations=queries, rng=rng), bmdp)

	return b -> action(planner, b)
end

function raster_policy(pomdp, b)
	pos = convert_pos_idx_2_pos_coord(pomdp, b.pos)

	if b.cost_expended % 30 == 0
		if pos == RoverPos(1,1)
			return :right
		else
			return :drill
		end
	end

	# in an odd column
	if pos[1] % 2 == 0
		# at the top of the column
		if pos[2] == pomdp.map_size[2]
			return :right
		else
			return :up
		end
	# in an even column
	else
		# at the bottom of the column
		if pos[2] == 1
			# at the origin
			if pos[1] == 1
				return :right
			else
				return :right
			end
		else
			return :down
		end
	end
end

function random_policy(pomdp, b)
	possible_actions = POMDPs.actions(pomdp, b)
	return rand(pomdp.rng, possible_actions)
end

function solver_test_RoverBMDP(pref::String, f_prior::GaussianProcess, rollout_depth::Int, rollout_iterations::Int; number_of_sample_types::Int=10, map_size::Tuple{Int, Int}=(10,10), seed::Int64=1234, num_graph_trials=50, total_budget = 100.0, use_ssh_dir=false, plot_results=true, run_raster=false, run_random=false)

	gp_mcts_rewards = Vector{Float64}(undef, 0)
	rmse_hist_gp_mcts = []
	trace_hist_gp_mcts = []
	total_planning_time_gp_mcts = 0
	total_plans_gp_mcts = 0

	raster_rewards = Vector{Float64}(undef, 0)
	rmse_hist_raster = []
	trace_hist_raster = []
	total_planning_time_raster = 0
	total_plans_raster = 0

	random_rewards = Vector{Float64}(undef, 0)
	rmse_hist_random = []
	trace_hist_random = []
	total_planning_time_random = 0
	total_plans_random = 0

    i = 1
    idx = 1
    while idx <= num_graph_trials
        @show i
        rng = MersenneTwister(seed+i)

		# true_map = build_map(rng, number_of_sample_types, map_size)
		true_map = load(path_name * "/true_maps/true_map$(idx).jld")["true_map"]


		#########################################################################################
		# take a sample at the starting location to be consistent with other TO_AIPPMS methods
        spec_pos = (1,1)
        σ²_n = σ_spec^2
		o = true_map[1] + round(rand(rng, Normal(0, σ_spec)), digits=1)
        f_init = posterior(f_prior, [[spec_pos[1], spec_pos[2]]], [o], [σ²_n])
		#########################################################################################

        pomdp = RoverPOMDP(true_map=true_map, f_prior=f_init, cost_budget=total_budget, sample_types=sample_types = collect(0:(1/number_of_sample_types):(1-1/number_of_sample_types)), drill_time = sample_cost, rng=rng)

		bmdp = BeliefMDP(pomdp, RoverBeliefUpdater(pomdp), belief_reward)

        gp_bmdp_isterminal(s) = POMDPs.isterminal(pomdp, s)

		gp_bmdp_policy = get_gp_bmdp_policy(bmdp, rng, rollout_depth, rollout_iterations)

		# GP-MCTS-DPW
		gp_mcts_reward, state_hist, gp_hist, action_hist, reward_hist, total_reward_hist, planning_time, num_plans = run_rover_bmdp(rng, bmdp, gp_bmdp_policy, gp_bmdp_isterminal)
		total_planning_time_gp_mcts += planning_time
		println("average planning time: ", planning_time/num_plans)
		total_plans_gp_mcts += num_plans
		# rmse_hist_gp_mcts = vcat(rmse_hist_gp_mcts, [calculate_rmse_along_traj(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)])
		# trace_hist_gp_mcts = vcat(trace_hist_gp_mcts, [calculate_trace_Σ(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)])
		
		executed_traj_gp_mcts = [[convert_pos_idx_2_pos_coord(pomdp, state_hist[i])[1], convert_pos_idx_2_pos_coord(pomdp, state_hist[i])[2]] for i in 1:length(state_hist)]
		executed_drills_gp_mcts = action_hist .== :drill
		# JLD2.save(path_name * "/true_maps/true_map$(idx).jld", "true_map", true_map)
		JLD2.save(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(idx).jld", "final_gp_X", gp_hist[end].X)
		JLD2.save(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(idx).jld", "final_gp_y", gp_hist[end].y)
		JLD2.save(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(idx).jld", "final_gp_ν", gp_hist[end].ν)
		JLD2.save(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj_gp_mcts$(idx).jld", "executed_traj_gp_mcts", executed_traj_gp_mcts)
		JLD2.save(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drills_gp_mcts$(idx).jld", "executed_drills_gp_mcts", executed_drills_gp_mcts)
	

		if plot_results
			# plot_trial(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "gp_mcts_dpw", use_ssh_dir)
			# plot_trial_with_mean(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "gp_mcts_dpw", use_ssh_dir)
			# plot_true_map(pomdp.true_map,i, "gp_mcts_dpw", use_ssh_dir)
			# plot_error_map(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "gp_mcts_dpw", use_ssh_dir)
			# plot_RMSE_trajectory(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "gp_mcts_dpw", use_ssh_dir)
			# plot_trace_trajectory(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "gp_mcts_dpw", use_ssh_dir)
		end
		@show gp_mcts_reward

		# RASTER POLICY
		if run_raster
			rast_p = b -> raster_policy(pomdp, b)
			raster_reward, state_hist, gp_hist, action_hist, reward_hist, total_reward_hist, planning_time, num_plans = run_rover_bmdp(rng, bmdp, rast_p, gp_bmdp_isterminal)
			total_planning_time_raster += planning_time
			total_plans_raster += num_plans
			rmse_hist_raster = vcat(rmse_hist_raster, [calculate_rmse_along_traj(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)])
			trace_hist_raster = vcat(trace_hist_raster, [calculate_trace_Σ(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)])
			if plot_results
				# plot_trial(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "raster", use_ssh_dir)
				# plot_trial_with_mean(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "raster", use_ssh_dir)
				# plot_true_map(pomdp.true_map,i, "raster", use_ssh_dir)
				# plot_error_map(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "raster", use_ssh_dir)
				# plot_RMSE_trajectory(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "raster", use_ssh_dir)
				# plot_trace_trajectory(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "raster", use_ssh_dir)
			end
			@show raster_reward
			push!(raster_rewards, raster_reward)
		end

		# RANDOM POLICY
		if run_random
			random_p = b -> random_policy(pomdp, b)
			random_reward, state_hist, gp_hist, action_hist, reward_hist, total_reward_hist, planning_time, num_plans = run_rover_bmdp(rng, bmdp, random_p, gp_bmdp_isterminal)
			total_planning_time_random += planning_time
			total_plans_random += num_plans
			rmse_hist_random = vcat(rmse_hist_random, [calculate_rmse_along_traj(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)])
			trace_hist_random = vcat(trace_hist_random, [calculate_trace_Σ(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)])
			
			executed_traj_random = [[convert_pos_idx_2_pos_coord(pomdp, state_hist[i])[1], convert_pos_idx_2_pos_coord(pomdp, state_hist[i])[2]] for i in 1:length(state_hist)]
			executed_drills_random = action_hist .== :drill
			JLD2.save(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(idx).jld", "final_gp_X", gp_hist[end].X)
			JLD2.save(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(idx).jld", "final_gp_y", gp_hist[end].y)
			JLD2.save(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(idx).jld", "final_gp_ν", gp_hist[end].ν)
			JLD2.save(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj_random$(idx).jld", "executed_traj_random", executed_traj_random)
			JLD2.save(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drills_random$(idx).jld", "executed_drills_random", executed_drills_random)
	
			@show random_reward
			push!(random_rewards, random_reward)
		end

        i = i+1
        idx = idx+1

        push!(gp_mcts_rewards, gp_mcts_reward)

    end

	if plot_results
		# plot_RMSE_trajectory_history(rmse_hist_gp_mcts, "gp_mcts_dpw", use_ssh_dir)
		# plot_RMSE_trajectory_history(rmse_hist_raster, "raster", use_ssh_dir)
		# plot_trace_Σ_history(trace_hist_gp_mcts, "gp_mcts_dpw", use_ssh_dir)
		# plot_trace_Σ_history(trace_hist_raster, "raster", use_ssh_dir)
	end

	println("GP-MCTS-DPW average planning time: ", total_planning_time_gp_mcts/total_plans_gp_mcts)
	@show mean(gp_mcts_rewards)

	if run_raster
		println("Raster average planning time: ", total_planning_time_raster/total_plans_raster)
		@show mean(raster_rewards)
	end

	if run_random
		println("Random average planning time: ", total_planning_time_random/total_plans_random)
		@show mean(random_rewards)
	end
end

solver_test_RoverBMDP("test", f_prior_sboaippms, rollout_depth, rollout_iterations, number_of_sample_types=number_of_sample_types, map_size=map_size_sboaippms, total_budget=total_budget, use_ssh_dir=use_ssh_dir, plot_results=plot_results, run_raster=run_raster, run_random = run_random, num_graph_trials=num_trials)
