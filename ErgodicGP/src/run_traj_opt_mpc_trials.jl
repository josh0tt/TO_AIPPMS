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
include("ErgodicGaussianProcessControl.jl")
@everywhere include("../../parameters.jl")

println("#############################################################")
println("####################    Checking σ's   ######################")
println("#############################################################")
println("σ_drill: ", σ_drill)
println("σ_spec: ", σ_spec)


@everywhere function run_traj_opt_mpc(egpm, replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging, true_interp_map, rng, trial_num)
    i = 1 # counter to indicate when it's time to replan
    t = 1
    map_size = size(true_interp_map)
    R = 0.1*[1 0; 0 1]#0.01 * [1 0; 0 1]

    # tm = TrajectoryManager(x0, xf, h, N, CornerInitializer());
    tm = TrajectoryManager(x0, xf, h, N, CornerInitializer());
    tm.N = N - sample_cost*init_num_drills
    tm.rng = rng
    tm.R = R
    tm.x0 += [abs(rand(tm.rng, Normal(0, 0.01))), abs(rand(tm.rng, Normal(0, 0.01)))]
    tm.barrier_cost=1000.0
    tm.sample_cost = sample_cost
    max_iters=500
    
    xd = nothing; ud = nothing; sample_actions=nothing; best_score = nothing; best_xd = nothing; best_ud = nothing; best_sample_actions = nothing;
    replan_flag = true
    while replan_flag
        tm.R = R
        tm.N = N - sample_cost*init_num_drills
        xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions = pto_trajectory(egpm, tm, verbose=verbose, logging=logging, max_iters=max_iters)
        # best_xd, best_ud = smc_trajectory(egpm, tm; verbose, umax=0.10)

        ud_mag = [sqrt(ud[i][1]^2 + ud[i][2]^2) for i in 1:length(ud)]


        if maximum(ud_mag) > ud_max
            R = 2 .* R
            # tm.x0 += [abs(rand(tm.rng, Normal(0, 0.01))), abs(rand(tm.rng, Normal(0, 0.01)))]
            replan_flag = true
            @show maximum(ud_mag)
            println("🚨maximum control input exceeded")
        else
            replan_flag = false
        end
    end
    q = objective == "expected_improvement" ? 4 : 2 
    νₚ_best = query_no_data(f_prior)[q]

    ν_storage = zeros(bins_x+1, bins_y+1, 1+convert(Int, floor(N/replan_rate)))

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
    Tb = N - sample_cost*length(sample_actions) - steps_taken
    while steps_taken < N 
        @show steps_taken
        if i >= replan_rate
            x = [xd[i][1] for i in 1:length(xd)]
            y = [xd[i][2] for i in 1:length(xd)]
            sample_actions_xy = [convert_perc2xy(egpm, tm, sample_actions[i], xd) for i in 1:length(sample_actions)]
            drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]

            if any(i .== drill_idx) # the next location is a drill
                x_samp = [x[i],y[i]]

                # You need to convert x,y pos to matrix indices to access true map pos

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
                #y_samp = GP.m([sample_action_xy[1],sample_action_xy[2]])
                post_GP = posterior(post_GP, [x_samp] , [y_samp], [egpm.σ_drill])
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
                post_GP = posterior(post_GP, [x_samp] , [y_samp], [egpm.σ_spec^2])

                steps_taken += 1 
            end

            gp_X = vcat(gp_X, [x_samp])
            gp_y = vcat(gp_y, [y_samp])
            gp_ν = vcat(gp_ν, [ν_samp])

            q = objective == "expected_improvement" ? 4 : 2 
            phi = Matrix(reshape(query(post_GP)[q], (bins_x+1, bins_y+1))')
            egpm = ErgodicGPManagerR2(egpm.domain, post_GP, egpm.σ_drill, egpm.σ_spec, phi, egpm.K) # create new egpm since we need to decompose new phi kpixl, kpixy, etc.
            
            x0 = xd[replan_rate+1]
            Tb = N - sample_cost*length(sample_actions_xy) - steps_taken#N-t # N stays fixed at the as a total budget that we then subtract from depending on steps and drills #length(best_ud)-t #N-t # remember ud is of length N, xd is length N+1
            append!(executed_traj, [xd[i]])
            append!(traj_length, length(xd))
            append!(t_indexes, t)
            
            tm.x0 = x0
            tm.N = Tb
            @show tm.N
            @show sample_actions

            if Tb <= 1
                break
            end 


            #######################################
            # USING PREVIOUS TRAJ AS INITIALIZATION
            #######################################
            # NOTE: this outperforms REINITIALIZING in both trace and rmse
            xd0 = xd[(replan_rate+1):end]
            ud0 = ud[(replan_rate+1):end]
            # convert to xy, shorten traj, then convert back
            sample_actions0 = length(sample_actions_xy) > 0 ? convert_xy2perc(egpm, tm, sample_actions_xy, xd0) : Float64[]
            replan_flag = true
            while replan_flag
                tm.N = Tb
                tm.R = R
                xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions = pto_trajectory(egpm, tm, xd0, ud0, sample_actions0; verbose=verbose, logging=logging, max_iters, gps_crit=0.0, dd_crit=1e-6)
                ud_mag = [sqrt(ud[i][1]^2 + ud[i][2]^2) for i in 1:length(ud)]
                @show tm.x0
                @show length(xd)
                
                if maximum(ud_mag) > ud_max
                    R = 2 .* R
                    # tm.x0 += [abs(rand(tm.rng, Normal(0, 0.01))), abs(rand(tm.rng, Normal(0, 0.01)))]
                    replan_flag = true
                    @show maximum(ud_mag)
                    println("🚨maximum control input exceeded")
                else
                    replan_flag = false
                end
            end

            drill_added = length(ud) < Tb ? true : false

            #######################################
            # REINITIALIZING TRAJ
            #######################################
            # # @show tm
            # # @show t 
            # # @show length(best_ud)
            # # @show length(best_xd)
            # # @show best_sample_actions 
            # # tm = TrajectoryManager(x0, xf, h, Tb, ConstantInitializer([0.0,0.0]));
            # # tm.barrier_cost=1000.0
            # # tm.sample_cost = sample_cost
        
            # sample_actions0 = [1/((3-length(executed_drill_locations) + 1)) * i for i in 1:(3-length(executed_drill_locations))]
            # # if t == 50
            # #     @show length(egpm.GP.X)
            # #     xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions = pto_trajectory(egpm, tm, sample_actions0; verbose=true, logging=true, max_iters, gps_crit=0.0, dd_crit=1e-6)
            # #     break
            # # end

            # # xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions = pto_trajectory(egpm, tm, sample_actions0; verbose=verbose, logging=logging, max_iters, gps_crit=0.0, dd_crit=1e-6)
            # # best_xd = xd
            # # best_sample_actions = sample_actions
            # # best_ud = ud
            # replan_flag = true
            # while replan_flag
            #     # JLD2.save("/data/executed_traj_tests/phi.jld", "phi", egpm.phi)
            #     # JLD2.save("/data/executed_traj_tests/x0.jld", "x0", x0)

            #     tm.R = R
            #     xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions = pto_trajectory(egpm, tm, sample_actions0; verbose=verbose, logging=logging, max_iters, gps_crit=0.0, dd_crit=1e-6)
            #     # best_xd, best_ud = smc_trajectory(egpm, tm; verbose, umax=0.10)

            #     if length(xd) <= 2
            #         break
            #     end
            #     ud_mag = [sqrt(ud[i][1]^2 + ud[i][2]^2) for i in 1:length(ud)]
            #     @show tm.R
            #     @show tm.x0
            #     @show length(xd)
            #     if maximum(ud_mag) > ud_max
            #         R = 2 .* R
            #         # tm.x0 += [rand(tm.rng, Normal(0, 0.01)), rand(tm.rng, Normal(0, 0.01))]
            #         replan_flag = true
            #         @show maximum(ud_mag)
            #         println("🚨maximum control input exceeded")
            #     else
            #         replan_flag = false
            #     end
            # end
            # drill_added = length(ud) < Tb ? true : false

            i = 1
            νₚ_best = egpm.phi' #query(post_GP)[2]
        else
            i += 1
            drill_added = false
        end
        t = drill_added ? N - length(ud) : N - length(ud) + 1
        step_idx += 1
        #t += 1 # NOTE: the drill vs step cost is considered in the optimization, once the drills are added you don't need to include it in t as well #any(1 .== drill_idx) ? sample_cost : 1
    end

    # Finish executing the trajectory (2 steps)
    x = [xd[i][1] for i in 1:length(xd)]
    y = [xd[i][2] for i in 1:length(xd)]
    sample_actions_xy = [convert_perc2xy(egpm, tm, sample_actions[i], xd) for i in 1:length(sample_actions)]
    drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]

    for i = 1:length(xd)
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
            ν_samp = egpm.σ_drill
            post_GP = posterior(post_GP, [x_samp] , [y_samp], [egpm.σ_drill])
            append!(executed_drill_locations, [[x_samp[1],x_samp[2]]])
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
            ν_samp = egpm.σ_spec^2
            post_GP = posterior(post_GP, [x_samp] , [y_samp], [egpm.σ_spec^2])
        end

        gp_X = vcat(gp_X, [x_samp])
        gp_y = vcat(gp_y, [y_samp])
        gp_ν = vcat(gp_ν, [ν_samp])
    end
    append!(executed_traj, xd)

    ν_storage[:,:, convert(Int, floor(t/replan_rate))] = νₚ_best
    append!(traj_length, length(xd))
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
    values, t = @timed run_traj_opt_mpc(ErgodicGPManagerR2(d, GP, σ_drill, σ_spec, phi, K_fc), replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging, true_interp_map, rng, i)
    executed_traj, executed_drill_locations, gp_X, gp_y, gp_ν = values
    planning_time[i] = t
    num_actions[i] = length(executed_traj)

    x = [executed_traj[i][1] for i in 1:length(executed_traj)]
    y = [executed_traj[i][2] for i in 1:length(executed_traj)]
    drill_idx = [find_nearest_traj_drill_pts(x, y, executed_drill_locations, i) for i in 1:length(executed_drill_locations)]

    JLD2.save(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld", "final_gp_X", gp_X)
    JLD2.save(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld", "final_gp_y", gp_y)
    JLD2.save(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld", "final_gp_ν", gp_ν)
    JLD2.save(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld", "executed_traj", executed_traj)
    JLD2.save(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld", "executed_drill_locations", drill_idx)

end 

avg_planning_time = sum(planning_time)/sum(num_actions)
@show avg_planning_time


