# plot_trajectories.jl
using Distributed
using SharedArrays
using Plots
using DelimitedFiles
using StatsBase
using JLD2
using KernelFunctions
using LinearAlgebra
using Colors
using Distributions

include("../CustomGP.jl")
include("../parameters.jl")

theme(:dao)

plot_gifs = false
# path_name = "/data/results_ssh_TO_AIPPMS/results_ssh_TO_AIPPMS/variance"

function plot_trial_mean(xd, drill_idx, gp_X, gp_y, gp_ν, trial_num, trial_name)
	k = with_lengthscale(SqExponentialKernel(), 0.1) # NOTE: check length scale
	plot_scale = 0:0.01:1#1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    m(x) = 0.0 
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2]) 
    KXqXq = K(X_plot, X_plot, k)

	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
    gp = GaussianProcess(m, μ(X_plot, m), k, gp_X, X_plot, gp_y, gp_ν, K(gp_X, gp_X, k), K(X_plot, gp_X, k), KXqXq);

	# Gaussian Process
	# contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', c = cgrad(:inferno, rev = false), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, colorbar=false) # xlims = (1, 10), ylims = (1, 10)
	heatmap(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, axis=false, ticks=false, colorbar=false) # xlims = (1, 10), ylims = (1, 10)

	# Agent location
    plt = plot!([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, color=:orchid1, linestyle=:solid, linewidth=3)
    # plt = scatter!([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, color=:orchid1, linewidth=3)

    scatter_idx = findlast(drill_idx .<= length(xd))
    if scatter_idx != nothing
        plt = scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, color=:green, markeralpha=1, markersize=6)
    end

    return plt
end

function plot_trial_objective(xd, drill_idx, gp_X, gp_y, gp_ν, trial_num, trial_name)
	k = with_lengthscale(SqExponentialKernel(), 0.1) # NOTE: check length scale
	plot_scale = 0:0.01:1#1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    m(x) = 0.0 
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2]) 
    KXqXq = K(X_plot, X_plot, k)

	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
    gp = GaussianProcess(m, μ(X_plot, m), k, gp_X, X_plot, gp_y, gp_ν, K(gp_X, gp_X, k), K(X_plot, gp_X, k), KXqXq);

	# Gaussian Process
    q = objective == "expected_improvement" ? 4 : 2
    if objective == "expected_improvement"
	    contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,0.01), grid=false, levels = collect(-0.002:0.001:0.01), axis=false, ticks=false, colorbar=false) # xlims = (1, 10), ylims = (1, 10)
    else
        heatmap(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, axis=false, ticks=false, colorbar=false) # xlims = (1, 10), ylims = (1, 10)
        # contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, colorbar=false) # xlims = (1, 10), ylims = (1, 10)
    end
	# Agent location
    plt = plot!([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, color=:orchid1, linestyle=:solid, linewidth=3)
    # plt = scatter!([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, color=:orchid1, linewidth=3)

    scatter_idx = findlast(drill_idx .<= length(xd))
    if scatter_idx != nothing
        plt = scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, color=:green, markeralpha=1, markersize=6)
    end

    return plt
end


i = 14
@show i

true_map = load(path_name * "/true_maps/true_map$(i).jld")["true_map"]


##########################
# GP MCTS
##########################
executed_traj_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj_gp_mcts$(i).jld")["executed_traj_gp_mcts"]
executed_drills_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drills_gp_mcts$(i).jld")["executed_drills_gp_mcts"]
gp_X_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
gp_y_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
gp_ν_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]

# for SBO_AIPPMS we have to scale (1, 11)x(1, 11) grid to (0, 1)x(0, 1)
executed_traj_gp_mcts .-= [[1.0,1.0]]
executed_traj_gp_mcts = executed_traj_gp_mcts ./ 10

# rescale the X's
gp_X_gp_mcts .-= [[1.0,1.0]]
gp_X_gp_mcts = gp_X_gp_mcts ./ 10

# terminal_element = [-0.2, -0.2]
# traj_length = first(findall(x->x==terminal_element, executed_traj_gp_mcts)) - 1
push!(executed_traj_gp_mcts, [1.0,1.0]) # mcts logging clips last one so get it to the goal for visualization
xd = executed_traj_gp_mcts#[1:traj_length]
drill_idxs = findall(x->x==1.0, executed_drills_gp_mcts)


if plot_gifs
    plot_trial_mean(xd, drill_idxs, gp_X_gp_mcts, gp_y_gp_mcts, gp_ν_gp_mcts, i, "SBO_AIPPMS")
    plot_trial_objective(xd, drill_idxs, gp_X_gp_mcts, gp_y_gp_mcts, gp_ν_gp_mcts, i, "SBO_AIPPMS")
end

##########################
# Ergodic
##########################

executed_traj_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
executed_drills_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
gp_X_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
gp_y_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
gp_ν_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]

xd = executed_traj_ergodic
drill_idxs = executed_drills_ergodic

if plot_gifs
    plot_trial_mean(xd, drill_idxs, gp_X_ergodic, gp_y_ergodic, gp_ν_ergodic, i, "Ergodic")
    plot_trial_objective(xd, drill_idxs, gp_X_ergodic, gp_y_ergodic, gp_ν_ergodic, i, "Ergodic")
end

##########################
# Ergodic GP
##########################
executed_traj_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
executed_drills_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
gp_X_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
gp_y_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
gp_ν_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]

xd = executed_traj_ergodicGP
drill_idxs = executed_drills_ergodicGP

if plot_gifs
    plot_trial_mean(xd, drill_idxs, gp_X_ergodicGP, gp_y_ergodicGP, gp_ν_ergodicGP, i, "ErgodicGP")
    plot_trial_objective(xd, drill_idxs, gp_X_ergodicGP, gp_y_ergodicGP, gp_ν_ergodicGP, i, "ErgodicGP")
end

##########################
# GP PTO
##########################
executed_traj_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
executed_drills_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
gp_X_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
gp_y_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
gp_ν_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]

xd = executed_traj_gp_pto
drill_idxs = executed_drills_gp_pto

if plot_gifs
    plot_trial_mean(xd, drill_idxs, gp_X_gp_pto, gp_y_gp_pto, gp_ν_gp_pto, i, "GP_PTO")
    plot_trial_objective(xd, drill_idxs, gp_X_gp_pto, gp_y_gp_pto, gp_ν_gp_pto, i, "GP_PTO")
end


##########################
# GP PTO Offline
##########################
executed_traj_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
executed_drills_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
gp_X_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
gp_y_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
gp_ν_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]

xd = executed_traj_gp_pto_offline
drill_idxs = executed_drills_gp_pto_offline

if plot_gifs
    plot_trial_mean(xd, drill_idxs, gp_X_gp_pto_offline, gp_y_gp_pto_offline, gp_ν_gp_pto_offline, i, "GP_PTO_offline")
    plot_trial_objective(xd, drill_idxs, gp_X_gp_pto_offline, gp_y_gp_pto_offline, gp_ν_gp_pto_offline, i, "GP_PTO_offline")
end

##########################
# Random
##########################
executed_traj_random = load(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj_random$(i).jld")["executed_traj_random"]
executed_drills_random = load(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drills_random$(i).jld")["executed_drills_random"]
gp_X_random = load(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
gp_y_random = load(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
gp_ν_random = load(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]

# for SBO_AIPPMS we have to scale (1, 11)x(1, 11) grid to (0, 1)x(0, 1)
executed_traj_random .-= [[1.0,1.0]]
executed_traj_random = executed_traj_random ./ 10

# rescale the X's
gp_X_random .-= [[1.0,1.0]]
gp_X_random = gp_X_random ./ 10

push!(executed_traj_random, [1.0,1.0]) # mcts logging clips last one so get it to the goal for visualization
xd = executed_traj_random#[1:traj_length]
drill_idxs = findall(x->x==1.0, executed_drills_random)

if plot_gifs
    plot_trial_mean(xd, drill_idxs, gp_X_random, gp_y_random, gp_ν_random, i, "Random")
    plot_trial_objective(xd, drill_idxs, gp_X_random, gp_y_random, gp_ν_random, i, "Random")
end


############################################################################################
dir_names = ["SBO_AIPPMS", "Ergodic", "ErgodicGP", "GP_PTO", "GP_PTO_offline", "Random"]
executed_trajectories = [executed_traj_gp_mcts, executed_traj_ergodic, executed_traj_ergodicGP, executed_traj_gp_pto, executed_traj_gp_pto_offline, executed_traj_random]
executed_drills = [executed_drills_gp_mcts, executed_drills_ergodic, executed_drills_ergodicGP, executed_drills_gp_pto, executed_drills_gp_pto_offline, executed_drills_random]
gp_Xs = [gp_X_gp_mcts, gp_X_ergodic, gp_X_ergodicGP, gp_X_gp_pto, gp_X_gp_pto_offline, gp_X_random]
gp_ys = [gp_y_gp_mcts, gp_y_ergodic, gp_y_ergodicGP, gp_y_gp_pto, gp_y_gp_pto_offline, gp_y_random]
gp_νs = [gp_ν_gp_mcts, gp_ν_ergodic, gp_ν_ergodicGP, gp_ν_gp_pto, gp_ν_gp_pto_offline, gp_ν_random]
mean_plts = []
var_plts = []
for i in 1:length(dir_names)
    xd = executed_trajectories[i]
    if dir_names[i] == "SBO_AIPPMS" || dir_names[i] == "Random"
        drill_idxs = findall(x->x==1.0, executed_drills[i])
    else
        drill_idxs = executed_drills[i]
    end

    plt = plot_trial_mean(xd, drill_idxs, gp_Xs[i], gp_ys[i], gp_νs[i], i, dir_names[i])
    push!(mean_plts, plt)

    plt = plot_trial_objective(xd, drill_idxs, gp_Xs[i], gp_ys[i], gp_νs[i], i, dir_names[i])
    push!(var_plts, plt)
end

plot(mean_plts..., legend=false, size=(2400,400), layout=(1,6))
# savefig(path_name * "/Mean_Traj.pdf")
savefig(path_name * "/main_figure/Mean_Traj_$(i).pdf")


plot(var_plts..., legend=false, size=(2400,400), layout=(1,6))
# savefig(path_name * "/Var_Traj.pdf")
savefig(path_name * "/main_figure/Var_Traj_$(i).pdf")

plot_scale = 0:0.01:1#1:0.1:10
true_interp_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))

for i in 1:size(true_interp_map)[1]
    for j in 1:size(true_interp_map)[2]
        idx_i = Int(floor(i/10) + 1)
        idx_j = Int(floor(j/10) + 1)
        true_interp_map[i,j] = true_map[idx_i, idx_j]
    end
end
# heatmap(collect(plot_scale), collect(plot_scale), true_interp_map, c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, axis=false, ticks=false, colorbar=false, size=(400,400))

theme(:default)
heatmap(true_interp_map, c = cgrad(:inferno, rev = true), grid=false, axis=false, ticks=false, colorbar=false,aspectratio = :equal,clim=(0,1))
# heatmap(true_map, c = cgrad(:inferno, rev = true), legend = false, aspectratio = :equal, clim=(0,1), grid=false, axis=false, ticks=false, colorbar=false, size=(400,400))
savefig(path_name * "/main_figure/true_map_$(i).pdf")
