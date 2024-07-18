# main_figure.jl
using Distributed
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
using SharedArrays

include("../../CustomGP.jl")
include("GaussianProcessControl.jl")
include("../../parameters.jl")


println("#############################################################")
println("####################    Checking σ's   ######################")
println("#############################################################")
println("σ_drill: ", σ_drill)
println("σ_spec: ", σ_spec)

function run_traj_opt(gpm, replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging, true_interp_map, rng, trial_num)

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

    max_iters = 200 # just for main page plotting !
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

function initial_traj(gpm, replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging, true_interp_map, rng, trial_num)
    tm = TrajectoryManager(x0, xf, h, N, CornerInitializer());
    tm.N = N - sample_cost*init_num_drills
    tm.rng = rng
    tm.x0 += [abs(rand(tm.rng, Normal(0, 0.01))), abs(rand(tm.rng, Normal(0, 0.01)))]
    tm.barrier_cost=100000.0
    tm.sample_cost = sample_cost

    map_size = size(true_interp_map)

    xd, ud = initialize(tm.initializer, gpm, tm)
    sample_actions = [0.25, 0.5, 0.75]
    executed_drill_locations = []
    executed_traj = []
    traj_length = []
    t_indexes = []

    post_GP = f_prior

    gp_X = [] 
    gp_y = []
    gp_ν = []

    x = [xd[i][1] for i in 1:length(xd)]
    y = [xd[i][2] for i in 1:length(xd)]
    sample_actions_xy = [convert_perc2xy(gpm, tm, sample_actions[i], xd) for i in 1:length(sample_actions)]
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
    append!(executed_traj, xd)

    append!(traj_length, length(xd))
    append!(t_indexes, t)

    return executed_traj, executed_drill_locations, gp_X, gp_y, gp_ν 
end


ud_max = 0.05 #0.1

i = 1
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

x = [executed_traj[i][1] for i in 1:length(executed_traj)]
y = [executed_traj[i][2] for i in 1:length(executed_traj)]
drill_idx = [find_nearest_traj_drill_pts(x, y, executed_drill_locations, i) for i in 1:length(executed_drill_locations)]


k = with_lengthscale(SqExponentialKernel(), 0.1) # NOTE: check length scale
X_plot = [[i,j] for i = plot_scale, j = plot_scale]
plot_size = size(X_plot)
m(x) = 0.0 
X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2]) 
KXqXq = K(X_plot, X_plot, k)


############################################################################
# Just make the plot
############################################################################
# increase GP query resolution for plotting
xd = executed_traj
gp = GaussianProcess(m, μ(X_plot, m), k, gp_X, X_plot, gp_y, gp_ν, K(gp_X, gp_X, k), K(X_plot, gp_X, k), KXqXq);

# Gaussian Process
plt_contour = plot()
q = objective == "expected_improvement" ? 4 : 2
if objective == "expected_improvement"
    plt_contour = contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, clim=(0,0.01), grid=false, levels = collect(-0.002:0.001:0.01), axis=false, ticks=false, size=(400,400), aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
else
    plt_contour = contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, size=(400,400), aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
end
# Agent location
plt_contour = scatter!([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, color=:orchid1, linestyle=:solid, linewidth=3)

scatter_idx = findlast(drill_idx .<= length(xd))
if scatter_idx != nothing
    plt_contour = scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, color=:green, markeralpha=1, markersize=6, colorbar=false)
end

savefig("/data/results/main_figure/contour_plot.pdf")

plt_points = plot()
plt_points = scatter([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, xaxis=false, yaxis=false, grid=false, ticks= false, color=:orchid1, linestyle=:solid, linewidth=3)
scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, ticks= false, color=:green, markeralpha=1, markersize=6, size=(400,400), aspectratio = :equal, background_color = :transparent)
savefig("/data/results/main_figure/traj_only.png")

# ############################################################################
# make the plot for xd0
# ############################################################################
executed_traj, executed_drill_locations, gp_X, gp_y, gp_ν = initial_traj(GaussianProcessManagerR2(d, GP, σ_drill, σ_spec), replan_rate, N, h, sample_cost, f_prior, max_iters, verbose, logging, true_interp_map, rng, i)


x = [executed_traj[i][1] for i in 1:length(executed_traj)]
y = [executed_traj[i][2] for i in 1:length(executed_traj)]
drill_idx = [find_nearest_traj_drill_pts(x, y, executed_drill_locations, i) for i in 1:length(executed_drill_locations)]


k = with_lengthscale(SqExponentialKernel(), 0.1) # NOTE: check length scale
X_plot = [[i,j] for i = plot_scale, j = plot_scale]
plot_size = size(X_plot)
m(x) = 0.0 
X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2]) 
KXqXq = K(X_plot, X_plot, k)

# increase GP query resolution for plotting
xd = executed_traj
gp = GaussianProcess(m, μ(X_plot, m), k, gp_X, X_plot, gp_y, gp_ν, K(gp_X, gp_X, k), K(X_plot, gp_X, k), KXqXq);

# Gaussian Process
plt_contour = plot()
q = objective == "expected_improvement" ? 4 : 2
if objective == "expected_improvement"
    plt_contour = contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, clim=(0,0.01), grid=false, levels = collect(-0.002:0.001:0.01), axis=false, ticks=false, size=(400,400), aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
else
    plt_contour = contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:inferno, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, size=(400,400), aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
end
# Agent location
plt_contour = scatter!([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, color=:orchid1, linestyle=:solid, linewidth=3)

scatter_idx = findlast(drill_idx .<= length(xd))
if scatter_idx != nothing
    plt_contour = scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, color=:green, markeralpha=1, markersize=6, colorbar=false)
end

savefig("/data/results/main_figure/contour_plot_xd0.pdf")

plt_points = plot()
plt_points = scatter([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, xaxis=false, yaxis=false, grid=false, ticks= false, color=:orchid1, linestyle=:solid, linewidth=3)
scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, ticks= false, color=:green, markeralpha=1, markersize=6, size=(400,400), aspectratio = :equal, background_color = :transparent)
savefig("/data/results/main_figure/traj_only_xd0.png")

plt_points_disturbed = plot()
plt_points_disturbed = scatter([xd[i][1] for i in 1:length(xd)] + rand(rng, Normal(0.0, 0.05), length(xd)),[xd[i][2] for i in 1:length(xd)] + rand(rng, Normal(0.0, 0.05), length(xd)),legend=false, xaxis=false, yaxis=false, grid=false, ticks= false, color=:gray, linestyle=:solid, linewidth=3)
scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, ticks= false, color=:gray, markeralpha=1, markersize=6, size=(400,400), aspectratio = :equal, background_color = :transparent)
savefig("/data/results/main_figure/disturbed_traj_only_xd0.png")