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

d = Domain(domain_min, domain_max, domain_bins)
phi = load(path_name * "/executed_traj_tests/phi.jld")["phi"]
x0 = load(path_name * "/executed_traj_tests/x0.jld")["x0"]
N = 50
egpm = ErgodicGPManagerR2(d, GP, σ_drill, σ_spec, phi, K_fc)
@show egpm.phik
tm = TrajectoryManager(x0, xf, h, N, ConstantInitializer([0.0,0.0]));
tm.barrier_cost=1000.0
tm.sample_cost = sample_cost

@time xd, ud, sample_actions, best_score, best_xd, best_ud, best_sample_actions = pto_trajectory(egpm, tm, verbose=true, logging=true, max_iters=500)
ud_mag = [sqrt(best_ud[i][1]^2 + best_ud[i][2]^2) for i in 1:length(best_ud)]
@show maximum(ud_mag)

x = [best_xd[i][1] for i in 1:length(best_xd)]
y = [best_xd[i][2] for i in 1:length(best_xd)]
best_sample_actions_xy = [convert_perc2xy(egpm, tm, best_sample_actions[i], xd) for i in 1:length(best_sample_actions)]
drill_idx = [find_nearest_traj_drill_pts(x, y, best_sample_actions_xy, i) for i in 1:length(best_sample_actions_xy)]

Plots.scatter(x,y,legend=false, color=:red)
# actual drill points
Plots.scatter!(x[drill_idx],y[drill_idx],legend=false, color=:green)
# requested drill points
Plots.scatter!([best_sample_actions_xy[i][1] for i in 1:length(best_sample_actions_xy)],[best_sample_actions_xy[i][2] for i in 1:length(best_sample_actions_xy)],legend=false, color=:yellow)

savefig(path_name * "/projected_traj_opt.png")


v_init = sum(query_no_data(GP)[2])
v_post = sum(query_sequence(egpm, [x; y], [drill_idx[i]/length(xd) for i in 1:length(drill_idx)], egpm.GP, tm, xd)) #query_sequence([x; y],GP)
variance_reduction = v_init - v_post
println("Variance Reduction of Best Trajectory SA_L10: ", variance_reduction)