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
	plot_scale = range(0,1,200)#1:0.1:10
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
	plot_scale = range(0,1,200)#1:0.1:10
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



############################################################################################
dir_names = ["GP_PTO"]
executed_trajectories = [executed_traj_gp_pto]
executed_drills = [executed_drills_gp_pto]
gp_Xs = [gp_X_gp_pto]
gp_ys = [gp_y_gp_pto]
gp_νs = [gp_ν_gp_pto]
mean_plts = []
var_plts = []
for i in 1:length(dir_names)
    xd = executed_trajectories[i]
    drill_idxs = executed_drills[i]
    println("here")

    plt = plot_trial_mean(xd, drill_idxs, gp_Xs[i], gp_ys[i], gp_νs[i], i, dir_names[i])
    push!(mean_plts, plt)

    println("here2")

    plt = plot_trial_objective(xd, drill_idxs, gp_Xs[i], gp_ys[i], gp_νs[i], i, dir_names[i])
    push!(var_plts, plt)
    println("here3")

end
plot(mean_plts..., legend=false, size=(2400,400), layout=(1,6))
# savefig(path_name * "/Mean_Traj.pdf")
savefig(path_name * "/main_figure/Mean_Traj_$(i).pdf")


plot(var_plts..., legend=false, size=(2400,400), layout=(1,6))
# savefig(path_name * "/Var_Traj.pdf")
savefig(path_name * "/main_figure/Var_Traj_$(i).pdf")

println("here4")

# heatmap(collect(plot_scale), collect(plot_scale), true_interp_map, c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, axis=false, ticks=false, colorbar=false, size=(400,400))

plot_scale = range(0,1,640)
true_map_plt = heatmap(collect(plot_scale), collect(plot_scale), true_map, c = cgrad(:inferno, rev = true), grid=false, axis=false, ticks=false, colorbar=false,aspectratio = :equal, clim=(0,1))
# heatmap(true_map, c = cgrad(:inferno, rev = true), legend = false, aspectratio = :equal, clim=(0,1), grid=false, axis=false, ticks=false, colorbar=false, size=(400,400))

println("here5")

plot([true_map_plt, mean_plts[1], var_plts[1]]..., legend=false, size=(1200,400), layout=(1,3))
savefig(path_name * "/main_figure/large.pdf")
