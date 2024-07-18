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

include("../../CustomGP.jl")
include("GaussianProcessControl.jl")
include("../../parameters.jl")


println("#############################################################")
println("####################    Checking σ's   ######################")
println("#############################################################")
println("σ_drill: ", σ_drill)
println("σ_spec: ", σ_spec)

d = Domain(domain_min, domain_max, domain_bins)
gpm = GaussianProcessManagerR2(d, GP, σ_drill, σ_spec)
# tm = TrajectoryManager(x0, xf, h, N, ConstantInitializer([0.001,0.001]));
# tm.Qf = [1e6 0; 0 1e6]
tm = TrajectoryManager(x0, xf, h, N, CornerInitializer());
tm.barrier_cost=1000.0
tm.sample_cost = sample_cost


@time xd, ud, sample_actions, best_score_actual, best_xd, best_ud, best_sample_actions, νₚ_best = pto_trajectory(gpm, tm, verbose=true, logging=true, max_iters=100)

x = [best_xd[i][1] for i in 1:length(best_xd)]
y = [best_xd[i][2] for i in 1:length(best_xd)]
# drill_idx = partialsortperm(best_sample_actions, 1:3, rev=true)
best_sample_actions_xy = [convert_perc2xy(gpm, tm, best_sample_actions[i], xd) for i in 1:length(best_sample_actions)]
drill_idx = [find_nearest_traj_drill_pts(x, y, best_sample_actions_xy, i) for i in 1:length(best_sample_actions_xy)]

Plots.scatter(x,y,legend=false, color=:red)
# actual drill points
Plots.scatter!(x[drill_idx],y[drill_idx],legend=false, color=:green)
# requested drill points
Plots.scatter!([best_sample_actions_xy[i][1] for i in 1:length(best_sample_actions_xy)],[best_sample_actions_xy[i][2] for i in 1:length(best_sample_actions_xy)],legend=false, color=:yellow)

savefig(path_name * "/projected_traj_opt.png")

q = objective == "expected_improvement" ? 4 : 2
v_init = sum(query_no_data(GP)[q])
v_post = query_sequence(gpm, [x; y], [drill_idx[i]/length(xd) for i in 1:length(drill_idx)], gpm.GP, tm, xd) #query_sequence([x; y],GP)
variance_reduction = sum(v_init .- v_post)
println("Variance Reduction of Best Trajectory SA_L10: ", variance_reduction)