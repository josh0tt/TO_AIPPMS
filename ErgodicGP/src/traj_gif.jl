using LinearAlgebra, Random, Distributions
using Interpolations
using StatsBase
using Images, Plots
ENV["GKSwstype"]="nul" 
# using POMDPs
# # using POMCPOW
# using POMDPPolicies
# using POMDPSimulators
#using POMDPModels
using DelimitedFiles
using KernelFunctions
using Zygote
using DelimitedFiles
using CSV

# NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
# for normal dist whereas our GP setup uses σ²_n
include("../../CustomGP.jl")
include("ErgodicGaussianProcessControl.jl")
include("../../parameters.jl")



println("#############################################################")
println("####################    Checking σ's   ######################")
println("#############################################################")
println("σ_drill: ", σ_drill)
println("σ_spec: ", σ_spec)


n_skip = 2

# X_plot = [[i,j] for i = range(0, 1, length=101), j = range(0, 0.5, length=51)]
plot_size = (101,101)
X_plot = [[i,j] for i = range(0, 1, length=plot_size[1]), j = range(0, 1, length=plot_size[2])]
X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])

KXqXq = K(X_plot, X_plot, k)
plot_GP = GaussianProcess(m, μ(X_plot, m), k, [], X_plot, [], [], [], [], KXqXq);

gps_actual = readdlm("data/gps_actual.csv", ',', Float64)
xfs = readdlm(path_name * "/xfs.csv", ',', Float64)
xsa = readdlm(path_name * "/xsa.csv", ',', Float64)
cs = readdlm(path_name * "/cs.csv", ',', Float64)
bs = readdlm(path_name * "/bs.csv", ',', Float64)
ts_actual = readdlm(path_name * "/temp_score_actual.csv", ',', Float64)
traj_length = readdlm(path_name * "/traj_length.csv", Float64)
num_samples = readdlm(path_name * "/num_samples.csv", Float64)

all_traj = readdlm(path_name * "/temp.csv", ',', Float64)
all_sample_actions = readdlm(path_name * "/temp_sample_actions.csv", ',', Float64)
ts_actual = [ts_actual[i] for i in 1:length(ts_actual)]

N = Int(traj_length[1]) # trajectory length
n = Int(num_samples[1]) # number of sample locations 
num_trajectories = length(traj_length) #round(Int, num_rows / N)
# num_trajectories = 80

# start with the first trajectory...
xd = [[all_traj[i,1], all_traj[i,2]] for i in 1:N]
x_hist = [xd[i][1] for i in 1:length(xd)]
y_hist = [xd[i][2] for i in 1:length(xd)]

sample_actions = [[all_sample_actions[i,1], all_sample_actions[i,2]] for i in 1:n]
#Plots.scatter(x,y,legend=false)

# gp_hist = []
# for i ∈ 1:length(x_hist)
#     global gp_hist, x_hist, y_hist, pomdp
#
#     gp_hist = vcat(gp_hist, plot_GP)
#     plot_GP = posterior(plot_GP, [[x_hist[i],y_hist[i]]], [GP.m([x_hist[i],y_hist[i]])], [pomdp.σ²_spec])
# end

# function make_gif(x_hist, y_hist, gp_hist, filename)
#     contourf(reshape(query_no_data(gp_hist[1])[2], (101,51))', colorbar = true, c = cgrad(:viridis, rev = true), xlims = (1, 101), ylims = (1, 51),legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim=(0,1))
#
#     anim_var = @animate for i ∈ 2:length(x_hist)
#
#         f_posterior = gp_hist[i]
#         contourf(reshape(query(f_posterior)[2], (101,51))', colorbar = true, c = cgrad(:viridis, rev = true), xlims = (1, 101), ylims = (1, 51),legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim=(0,1))
#         #contourf(reshape(query(f_posterior)[2], (11,11))', colorbar = true, c = cgrad(:viridis, rev = true), xlims = (1, 11), ylims = (1, 11),legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim=(0,1))
#         #contourf(1:10:101, 1:10:51, query(f_posterior)[2], colorbar = true, c = cgrad(:viridis, rev = true), xlims = (1, 101), ylims = (1, 51),legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal)
#
#         scatter!(x_hist[1:i], y_hist[1:i], legend=false, color=:red)
#         # #scatter!(rand(-10:n_skip:10, 10), [0 for _ in 1:10],label="Some points")
#         # if GW_actions[i][1] == :drill
#         #     scatter!(x_hist[1:i], y_hist[1:i], legend=false, color=:red)
#         # else
#         #     scatter!(x_hist[1:i], y_hist[1:i], legend=false, color=:red)
#         # end
#
#     end
#
#     gif(anim_var, filename, fps = 10)
#     #gif(anim_var, "Figures/anim_var_high.gif", fps = 10)
# end
#
# make_gif(gp_hist, x_hist, y_hist, "Results/gifs/traj_opt.gif")

################################################################################
##########################        Uncomment this     ###########################
################################################################################
# then do the rest
#anim = @animate for traj_idx = 2:n_skip:num_trajectories

function find_nearest_traj_drill_pts(x_hist, y_hist, sample_actions, action_idx)
	dist_to_drill_pt = [norm(sample_actions[action_idx] - [x_hist[i], y_hist[i]]) for i in 1:length(x_hist)]
	#push!(min_dist_idx, argmin(dist_to_drill_pt))
	min_dist_idx = argmin(dist_to_drill_pt)

	return min_dist_idx
end 

start_idx = 1
end_idx = Int(traj_length[1])
start_idx_act = 1
end_idx_act = Int(num_samples[1])

anim = @animate for traj_idx = 1:n_skip:num_trajectories
    global start_idx, end_idx, start_idx_act, end_idx_act

    if traj_idx == 1
        start_idx = 1
        end_idx = Int(traj_length[1])

        start_idx_act = 1
        end_idx_act = Int(num_samples[1])
    else
        # start_idx = end_idx + 1*n_skip + Int(sum(traj_length[(traj_idx-(n_skip-1)):(traj_idx-1)]))
        start_idx = end_idx + 1 + Int(sum(traj_length[(traj_idx-(n_skip-1)):(traj_idx-1)]))
        end_idx = start_idx + Int(traj_length[traj_idx])

        start_idx_act = end_idx_act + 1 + Int(sum(num_samples[(traj_idx-(n_skip-1)):(traj_idx-1)]))
        end_idx_act = start_idx_act + Int(num_samples[traj_idx]) - 1

    end

    println(traj_idx)

    xd = [[all_traj[i,1], all_traj[i,2]] for i in start_idx:end_idx]
    x = [xd[i][1] for i in 1:length(xd)]
    y = [xd[i][2] for i in 1:length(xd)]


    sample_actions = [[all_sample_actions[i,1], all_sample_actions[i,2]] for i in start_idx_act:(end_idx_act)]
    drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions, i) for i in 1:length(sample_actions)]


    f_posterior = plot_GP
    for i in 1:length(x)
        if (i in drill_idx)
            σ²_drill = σ_drill #^2 dont square this causes singular exception in GP update
            f_posterior = posterior(f_posterior, [[x[i],y[i]]], [GP.m([x[i],y[i]])], [σ²_drill])
        else
            σ²_spec = σ_spec^2
            f_posterior = posterior(f_posterior, [[x[i],y[i]]], [GP.m([x[i],y[i]])], [σ²_spec])
        end
    end

    dx = 1/(plot_size[1]-1)
    dy = 1/(plot_size[2]-1)
    contourf(0:dx:1, 0:dy:1, reshape(query(f_posterior)[2], plot_size)', colorbar = true, c = cgrad(:inferno, rev = true), xlims = (0, 1), ylims = (0, 1),legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim=(0,1))

    xd_shifted = xd.*100 + repeat([[1.0, 1.0]], length(xd))
    #xd_fair = [[round(xd_shifted[i][1], digits=0), round(xd_shifted[i][2], digits=0)] for i in 1:length(xd_shifted)]
    x_shifted = [xd_shifted[i][1] for i in 1:length(xd_shifted)]
    y_shifted = [xd_shifted[i][2] for i in 1:length(xd_shifted)]
    sample_actions_shifted = sample_actions.*100 + repeat([[1.0, 1.0]], length(sample_actions))


    x_no_shift = [xd[i][1] for i in 1:length(xd)]
    y_no_shift = [xd[i][2] for i in 1:length(xd)]
    scatter!(x_no_shift, y_no_shift,legend=false, color=:red)
    scatter!(x_no_shift[drill_idx],y_no_shift[drill_idx],legend=false, color=:green)
    scatter!([sample_actions[i][1] for i in 1:length(sample_actions)], [sample_actions[i][2] for i in 1:length(sample_actions)], legend=false, color=:yellow)

end
#
Plots.gif(anim, path_name * "/traj_opt.gif", fps = 20)

anim = @animate for traj_idx = 1:n_skip:num_trajectories

    sorted_scores_idx = sortperm([gps_actual[traj_idx], xfs[traj_idx], xsa[traj_idx], cs[traj_idx], bs[traj_idx]])
    score_names = ["GP Actual", "End Point", "Sample Selection Spacing", "Control Effort", "Boundary"]
    scores = [gps_actual[1:n_skip:traj_idx], xfs[1:n_skip:traj_idx], xsa[1:n_skip:traj_idx], cs[1:n_skip:traj_idx], bs[1:n_skip:traj_idx]]

    for i in 1:(length(score_names)-1)
        fill_bound = ts_actual[1:n_skip:traj_idx]
        for j in 1:i
            fill_bound -= scores[j]
        end

        if i == 1
            plot(1:n_skip:traj_idx, fill_bound + scores[i], fillrange = fill_bound, fillalpha = 0.35, c = i, label = score_names[i])
        else
            plot!(1:n_skip:traj_idx, fill_bound+ scores[i], fillrange = fill_bound, fillalpha = 0.35, c = i, label = score_names[i])
        end
    end

end
#
Plots.gif(anim, path_name * "/total_score_actual.gif", fps = 20)


anim = @animate for traj_idx = 1:n_skip:num_trajectories
    bar([0], [gps_actual[traj_idx]], label="GP Actual")
    bar!([1], [xfs[traj_idx]], label="End Point")
    bar!([2], [xsa[traj_idx]], label="Sample Selection Spacing")
    bar!([3], [cs[traj_idx]], label="Control Effort")
    bar!([4], [bs[traj_idx]], label="Boundary", ylabel="Score", ylim=(0,10))
end
#
Plots.gif(anim, path_name * "/actual_score_hist.gif", fps = 20)


anim = @animate for traj_idx = 1:n_skip:num_trajectories
    plot(traj_length[1:n_skip:traj_idx], label="Trajectory Length")
    plot!(num_samples[1:n_skip:traj_idx], label="Number of Samples")
end
#
Plots.gif(anim, path_name * "/traj_length_vs_sample_actions.gif", fps = 20)