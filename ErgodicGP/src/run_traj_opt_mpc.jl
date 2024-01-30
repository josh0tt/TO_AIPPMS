using LinearAlgebra, Random, Distributions
using Interpolations
using StatsBase
using Images, Plots
using KernelFunctions
using Zygote
using DelimitedFiles
using CSV
using StatProfilerHTML
using Optim, LineSearches
using JLD2

include("../../CustomGP.jl")
include("ErgodicGaussianProcessControl.jl")
include("../../parameters.jl")

println("#############################################################")
println("####################    Checking σ's   ######################")
println("#############################################################")
println("σ_drill: ", σ_drill)
println("σ_spec: ", σ_spec)

function run_traj_opt_mpc(egpm, replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging)

    i = 1 # counter to indicate when it's time to replan
    t = 1

    tm = TrajectoryManager(x0, xf, h, N, CornerInitializer());
    tm.barrier_cost=1000.0
    tm.sample_cost = sample_cost
   
    xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions = pto_trajectory(egpm, tm, verbose=verbose, logging=logging, max_iters=max_iters)
    νₚ_best = query_no_data(f_prior)[2]

    ν_storage = zeros(bins_x+1, bins_y+1, 1+convert(Int, floor(N/replan_rate)))
    xd_storage = Matrix{Vector{Float64}}(undef, N+1, 1+convert(Int, floor(N/replan_rate)))
    sample_action_storage = -1 .* ones(N+1, 1+convert(Int, floor(N/replan_rate)))#Matrix{Float64}(undef, N+1, 1+convert(Int, floor(N/replan_rate)))
    executed_drill_locations = []
    executed_traj = []
    traj_length = []
    t_indexes = []

    post_GP = f_prior

    Tb = N
    while t < N
        if i >= replan_rate
            x = [best_xd[i][1] for i in 1:length(best_xd)]
            y = [best_xd[i][2] for i in 1:length(best_xd)]
            @show best_sample_actions
            sample_actions_xy = [convert_perc2xy(egpm, tm, best_sample_actions[i], best_xd) for i in 1:length(best_sample_actions)]
	        drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]

            xd_storage[1:length(best_xd), convert(Int, floor(t/replan_rate))] = best_xd
            sample_action_storage[1:length(drill_idx), convert(Int, floor(t/replan_rate))] = drill_idx
            ν_storage[:,:, convert(Int, floor(t/replan_rate))] = νₚ_best

            if any(1 .== drill_idx) # the next location is a drill
                sample_action_xy = convert_perc2xy(egpm, tm, first(best_sample_actions[1 .== drill_idx]), best_xd)
                x_samp = [[sample_action_xy[1],sample_action_xy[2]]]
                y_samp = GP.m([sample_action_xy[1],sample_action_xy[2]])
                post_GP = posterior(post_GP, x_samp , [y_samp], [egpm.σ_drill])
                append!(executed_drill_locations, [[sample_action_xy[1],sample_action_xy[2]]])
                deleteat!(sample_actions_xy, first(findall(x->x==1, drill_idx))) # remove the sample location that was executed
            else
                x_samp = [[x[i],y[i]]]
                y_samp = GP.m([x[i],y[i]])
                # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
                # for normal dist whereas our GP setup uses σ²_n
                post_GP = posterior(post_GP, x_samp , [y_samp], [egpm.σ_spec^2])
            end

            egpm.GP = post_GP #GaussianProcessManagerR2(gpm.domain, post_GP, gpm.σ_drill, gpm.σ_spec)
            egpm.phi = Matrix(reshape(query(post_GP)[2], (bins_x+1, bins_y+1))')
            egpm.phi[end,end] = phi_end

            x0 = best_xd[replan_rate+1]
            Tb = N-t # N stays fixed at the as a total budget that we then subtract from depending on steps and drills #length(best_ud)-t #N-t # remember ud is of length N, xd is length N+1
            append!(executed_traj, [x0])
            append!(traj_length, length(best_xd))
            append!(t_indexes, t)
            
            tm.x0 = x0
            tm.N = Tb 
            
            #######################################
            # USING PREVIOUS TRAJ AS INITIALIZATION
            #######################################
            xd0 = best_xd[(replan_rate+1):end]
            ud0 = best_ud[(replan_rate+1):end]
            # convert to xy, shorten traj, then convert back
            sample_actions0 = length(sample_actions_xy) > 0 ? convert_xy2perc(egpm, tm, sample_actions_xy, xd0) : Float64[]
            xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions = pto_trajectory(egpm, tm, xd0, ud0, sample_actions0; verbose=verbose, logging=logging, max_iters, gps_crit=0.0, dd_crit=1e-6)
            drill_added = length(best_ud) == Tb ? false : true
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
            # xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions = pto_trajectory(egpm, tm, sample_actions0; verbose=verbose, logging=logging, max_iters, gps_crit=0.0, dd_crit=1e-6)

            i = 1
            νₚ_best = egpm.phi' #query(post_GP)[2]
        else
            i += 1
        end
        t = drill_added ? N - length(best_ud) : N - length(best_ud) + 1
        #t += 1 # NOTE: the drill vs step cost is considered in the optimization, once the drills are added you don't need to include it in t as well #any(1 .== drill_idx) ? sample_cost : 1
    end

    # Finish executing the trajectory (2 steps)
    x = [best_xd[i][1] for i in 1:length(best_xd)]
    y = [best_xd[i][2] for i in 1:length(best_xd)]
    sample_actions_xy = [convert_perc2xy(egpm, tm, best_sample_actions[i], best_xd) for i in 1:length(best_sample_actions)]
    drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]

    for i = 1:length(best_xd)
        if i in drill_idx
            x_samp = [[sample_actions_xy[i][1],sample_actions_xy[i][2]]]
            y_samp = GP.m([sample_actions_xy[i][1],sample_actions_xy[i][2]])
            post_GP = posterior(post_GP, x_samp , [y_samp], [egpm.σ_drill])
            append!(executed_drill_locations, [[sample_actions_xy[i][1],sample_actions_xy[i][1]]])
        else
            x_samp = [[x[i],y[i]]]
            y_samp = GP.m([x[i],y[i]])
            # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
            # for normal dist whereas our GP setup uses σ²_n
            post_GP = posterior(post_GP, x_samp , [y_samp], [egpm.σ_spec^2])
        end
    end
    append!(executed_traj, best_xd)

    xd_storage[1:length(best_xd), convert(Int, floor(t/replan_rate))] = best_xd
    sample_action_storage[1:length(drill_idx), convert(Int, floor(t/replan_rate))] = drill_idx
    ν_storage[:,:, convert(Int, floor(t/replan_rate))] = νₚ_best
    append!(traj_length, length(best_xd))
    append!(t_indexes, t)

    anim = @animate for i ∈ 1:length(t_indexes)#i ∈ 1:(size(ν_storage)[3]-1)
        @show i
        # xd = xd_storage[1:(N-(i-1)*replan_rate),i]
        xd = xd_storage[1:traj_length[i],t_indexes[i]]
        ν = ν_storage[:,:,t_indexes[i]]
        drill_idxs = sample_action_storage[:,t_indexes[i]] # these are drill_idxs
        drill_idxs = Int.(drill_idxs[drill_idxs .!= -1])
        @show drill_idxs

        dx = (x_max(egpm) - x_min(egpm))/(egpm.domain.cells[1]-1)
        dy = (y_max(egpm) - y_min(egpm))/(egpm.domain.cells[2]-1)
        x = [xd[i][1] for i in 1:length(xd)]
        y = [xd[i][2] for i in 1:length(xd)]
        contourf(0:dx:1, 0:dy:1, reshape(vec(ν), (egpm.domain.cells[1], egpm.domain.cells[2]))', colorbar = true, c = cgrad(:inferno, rev = true), xlims = (0, 1), ylims = (0, 1),legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim=(0,1))
        scatter!(x, y,legend=false, color=:red)
        scatter!(x[drill_idxs], y[drill_idxs],legend=false, color=:yellow)
        # scatter!([sample_actions_xy[i][1] for i in 1:length(sample_actions_xy)], [sample_actions_xy[i][2] for i in 1:length(sample_actions_xy)], legend=false, color=:yellow)

    end
    Plots.gif(anim, path_name * "/pto_mpc.gif", fps = 20)

    dx = (x_max(egpm) - x_min(egpm))/(egpm.domain.cells[1]-1)
    dy = (y_max(egpm) - y_min(egpm))/(egpm.domain.cells[2]-1)
    ν = ν_storage[:,:,end-1]
    x = [executed_traj[i][1] for i in 1:length(executed_traj)]
    y = [executed_traj[i][2] for i in 1:length(executed_traj)]
    x_drill = [executed_drill_locations[i][1] for i in 1:length(executed_drill_locations)]
    y_drill = [executed_drill_locations[i][2] for i in 1:length(executed_drill_locations)]
    contourf(0:dx:1, 0:dy:1, reshape(vec(ν), (egpm.domain.cells[1], egpm.domain.cells[2]))', colorbar = true, c = cgrad(:inferno, rev = true), xlims = (0, 1), ylims = (0, 1),legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim=(0,1))
    scatter!(x, y,legend=false, color=:red)
    scatter!(x_drill, y_drill,legend=false, color=:yellow)
    savefig(path_name *"/executed_traj.png")

    JLD2.save(path_name * "/ErgodicGP/executed_traj.jld", "executed_traj", executed_traj)
    JLD2.save(path_name * "/ErgodicGP/executed_drill_locations.jld", "executed_drill_locations", executed_drill_locations)
    JLD2.save(path_name * "/ErgodicGP/gp_hist/final_gp_X.jld", "final_gp_X", post_GP.X)
    JLD2.save(path_name * "/ErgodicGP/gp_hist/final_gp_y.jld", "final_gp_y", post_GP.y)
    JLD2.save(path_name * "/ErgodicGP/gp_hist/final_gp_ν.jld", "final_gp_ν", post_GP.ν)

end

d = Domain(domain_min, domain_max, domain_bins)
run_traj_opt_mpc(ErgodicGPManagerR2(d, GP, σ_drill, σ_spec, phi, K_fc), replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging)