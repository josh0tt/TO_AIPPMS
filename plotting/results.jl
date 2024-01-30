# results.jl
using Distributed
using SharedArrays
@everywhere using Plots
using DelimitedFiles
@everywhere using StatsBase
@everywhere using JLD2
@everywhere using KernelFunctions
@everywhere using LinearAlgebra
@everywhere using Colors
@everywhere using Distributions
using Measures 

@everywhere include("../CustomGP.jl")
@everywhere include("../parameters.jl")

objective = "variance" #"expected_improvement" # "variance"
save_path = path_name #"/Users/data/additional_results/expected_improvement/"

load_gp_mcts = true
load_ergodic = true
load_ergodic_gp = true
load_gp_pto = true
load_gp_pto_offline = true
load_random = true

plot_gifs = false
plot_final_subplot = true

theme(:dao)
default(titlefont=font(24, "Computer Modern"))
default(guidefont=font(22, "Computer Modern"))
default(tickfont=font(14, "Computer Modern"))
default(legendfont=font(12, "Computer Modern"))
default(linewidth=3)

@everywhere function plot_trial_mean(xd, drill_idx, gp_X, gp_y, gp_ν, trial_num, trial_name)
	k = with_lengthscale(SqExponentialKernel(), 0.1) # NOTE: check length scale
	plot_scale = 0:0.01:1#1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    m(x) = 0.0 
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2]) 
    KXqXq = K(X_plot, X_plot, k)

    anim = @animate for i = 1:length(xd)

		# increase GP query resolution for plotting
		if gp_X[1:i] == []
			gp = GaussianProcess(m, μ(X_plot, m), k, [], X_plot, [], [], [], [], KXqXq);
		else
            gp = GaussianProcess(m, μ(X_plot, m), k, gp_X[1:i], X_plot, gp_y[1:i], gp_ν[1:i], K(gp_X[1:i], gp_X[1:i], k), K(X_plot, gp_X[1:i], k), KXqXq);
		end


		# # Display Total Reward
		# if i == 1
		# 	title = "Mean \n Total Reward: $(total_reward_hist[1])"
		# else
		# 	title = "Mean \n Total Reward: $(total_reward_hist[i-1])"
		# end


		if gp_X[1:i] == []
			contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)
			# contourf(collect(1:0.3:9.7), collect(1:0.3:9.7), reshape(query_no_data(gp_hist[i])[2], (30,30))', colorbar = true, c =  cgrad(:davos, rev = false), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
		else
			contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)
		end

		# Drill Positions
		scatter_idx = findlast(drill_idx .<= i)
		if scatter_idx != nothing
            scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, color=:green, markeralpha=1, markersize=6)
			# scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1, markersize=6)
		end

        # Agent location
        scatter!([xd[i][1]], [xd[i][2]], legend=false, color=:orchid1, title="Mean", markersize=7)
		# scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title, markersize=7)

	end
    Plots.gif(anim, path_name * "/$(trial_name)/total_budget_$(total_budget)/σ_spec_$(σ_spec)/mean/mean$(trial_num).gif", fps=2) 


	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
    gp = GaussianProcess(m, μ(X_plot, m), k, gp_X, X_plot, gp_y, gp_ν, K(gp_X, gp_X, k), K(X_plot, gp_X, k), KXqXq);

	# Gaussian Process
	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Mean") # xlims = (1, 10), ylims = (1, 10)

	# Agent location
    plot!([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, color=:orchid1, linestyle=:solid, linewidth=3)

    scatter_idx = findlast(drill_idx .<= length(xd))
    if scatter_idx != nothing
        scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, color=:green, markeralpha=1, markersize=6)
    end

    savefig(path_name * "/$(trial_name)/total_budget_$(total_budget)/σ_spec_$(σ_spec)/mean/mean$(trial_num).pdf") 

end

@everywhere function plot_trial_objective(xd, drill_idx, gp_X, gp_y, gp_ν, trial_num, trial_name)
	k = with_lengthscale(SqExponentialKernel(), 0.1) # NOTE: check length scale
	plot_scale = 0:0.01:1#1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    m(x) = 0.0 
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2]) 
    KXqXq = K(X_plot, X_plot, k)

    anim = @animate for i = 1:length(xd)

		# increase GP query resolution for plotting
		if gp_X[1:i] == []
			gp = GaussianProcess(m, μ(X_plot, m), k, [], X_plot, [], [], [], [], KXqXq);
		else
            gp = GaussianProcess(m, μ(X_plot, m), k, gp_X[1:i], X_plot, gp_y[1:i], gp_ν[1:i], K(gp_X[1:i], gp_X[1:i], k), K(X_plot, gp_X[1:i], k), KXqXq);
		end

		# Gaussian Process Variance
        q = objective == "expected_improvement" ? 4 : 2
        if objective == "expected_improvement"
            if gp_X[1:i] == []
                contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,0.01), levels = collect(-0.002:0.001:0.01), grid=false, axis=false, ticks=false, title="Expected Improvement") # xlims = (1, 10), ylims = (1, 10)
                # contourf(collect(1:0.3:9.7), collect(1:0.3:9.7), reshape(query_no_data(gp_hist[i])[2], (30,30))', colorbar = true, c =  cgrad(:davos, rev = false), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
            else
                contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,0.01), levels = collect(-0.002:0.001:0.01), grid=false, axis=false, ticks=false, title="Expected Improvement") # xlims = (1, 10), ylims = (1, 10)
            end
        else
            if gp_X[1:i] == []
                contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)
                # contourf(collect(1:0.3:9.7), collect(1:0.3:9.7), reshape(query_no_data(gp_hist[i])[2], (30,30))', colorbar = true, c =  cgrad(:davos, rev = false), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
            else
                contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)
            end
        end

		# Drill Positions
		scatter_idx = findlast(drill_idx .<= i)
		if scatter_idx != nothing
            scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, color=:green, markeralpha=1, markersize=6)
			# scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1, markersize=6)
		end

        # Agent location
        scatter!([xd[i][1]], [xd[i][2]], legend=false, color=:orchid1, title="Variance", markersize=7)
		# scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title, markersize=7)

	end
    Plots.gif(anim, path_name * "/$(trial_name)/total_budget_$(total_budget)/σ_spec_$(σ_spec)/" * objective * "/" * objective * "$(trial_num).gif", fps=2) 


	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
    gp = GaussianProcess(m, μ(X_plot, m), k, gp_X, X_plot, gp_y, gp_ν, K(gp_X, gp_X, k), K(X_plot, gp_X, k), KXqXq);

	# Gaussian Process
    q = objective == "expected_improvement" ? 4 : 2
    if objective == "expected_improvement"
	    contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,0.01), grid=false, levels = collect(-0.002:0.001:0.01), axis=false, ticks=false, title="Expected Improvement") # xlims = (1, 10), ylims = (1, 10)
    else
        contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[q], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.00, 1.05), ylims = (0.00, 1.05), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)
    end
	# Agent location
    plot!([xd[i][1] for i in 1:length(xd)],[xd[i][2] for i in 1:length(xd)],legend=false, color=:orchid1, linestyle=:solid, linewidth=3)

    scatter_idx = findlast(drill_idx .<= length(xd))
    if scatter_idx != nothing
        scatter!([xd[drill_idx[i]][1] for i in collect(1:scatter_idx)], [xd[drill_idx[i]][2] for i in collect(1:scatter_idx)], legend=false, color=:green, markeralpha=1, markersize=6)
    end

    savefig(path_name * "/$(trial_name)/total_budget_$(total_budget)/σ_spec_$(σ_spec)/" * objective * "/" * objective * "$(trial_num).pdf") 

end

@everywhere function calculate_RMSE_TrΣ(true_map, xd, gp_X, gp_y, gp_ν, trial_name)
	k = with_lengthscale(SqExponentialKernel(), 0.1) # NOTE: check length scale
	plot_scale = 0:0.01:1#1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    m(x) = 0.0 
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2]) 
    KXqXq = K(X_plot, X_plot, k)

    error_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))
    true_interp_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))
    RMSE_hist = []
    trace_hist = []
    EI_hist = []

    for i in 1:size(true_interp_map)[1]
        for j in 1:size(true_interp_map)[2]
            idx_i = Int(floor(i/10) + 1)
            idx_j = Int(floor(j/10) + 1)
            true_interp_map[i,j] = true_map[idx_i, idx_j]
        end
    end

    if trial_name == "SBO_AIPPMS"
        for i = 1:length(xd)
            # increase GP query resolution for plotting
            if gp_X[1:i] == []
                gp = GaussianProcess(m, μ(X_plot, m), k, [], X_plot, [], [], [], [], KXqXq);
                mean_map, ν, S, EI = query_no_data(gp)
                trace_Σ = sum(ν)
                EI = sum(EI)
            else
                gp = GaussianProcess(m, μ(X_plot, m), k, gp_X[1:i], X_plot, gp_y[1:i], gp_ν[1:i], K(gp_X[1:i], gp_X[1:i], k), K(X_plot, gp_X[1:i], k), KXqXq);
                mean_map, ν, S, EI = query(gp)
                trace_Σ = sum(ν)
                EI = sum(EI)
            end

            error_map = abs.(true_interp_map - reshape(mean_map, size(true_interp_map)))
            RMSE = 0
            for i in 1:length(error_map)
                RMSE += error_map[i]^2
            end
            RMSE = sqrt(RMSE/length(error_map))

            append!(RMSE_hist, RMSE)
            append!(trace_hist, trace_Σ)
            append!(EI_hist, EI)
        end

        # once it's reached it's cost budget it stays there for the remaining time units 
        RMSE_end = RMSE_hist[end]
        trace_end = trace_hist[end]
        EI_end = EI_hist[end]
        for i = (length(xd)+1):total_budget
            append!(RMSE_hist, RMSE_end)
            append!(trace_hist, trace_end)
            append!(EI_hist, EI_end)
        end
        
    else
        for i = 1:length(xd)
            # increase GP query resolution for plotting
            if gp_X[1:i] == []
                gp = GaussianProcess(m, μ(X_plot, m), k, [], X_plot, [], [], [], [], KXqXq);
                mean_map, ν, S, EI = query_no_data(gp)
                trace_Σ = sum(ν)
                EI = sum(EI)
            else
                gp = GaussianProcess(m, μ(X_plot, m), k, gp_X[1:i], X_plot, gp_y[1:i], gp_ν[1:i], K(gp_X[1:i], gp_X[1:i], k), K(X_plot, gp_X[1:i], k), KXqXq);
                mean_map, ν, S, EI = query(gp)
                trace_Σ = sum(ν)
                EI = sum(EI)
            end

            # error_map = abs.(true_interp_map - reshape(mean_map, size(true_interp_map)))
            error_map = abs.(true_interp_map - reverse(reshape(mean_map, size(true_interp_map))', dims=1))

            RMSE = 0
            for i in 1:length(error_map)
                RMSE += error_map[i]^2
            end
            RMSE = sqrt(RMSE/length(error_map))

            append!(RMSE_hist, RMSE)
            append!(trace_hist, trace_Σ)
            append!(EI_hist, EI)

        end

        # once it's reached it's cost budget it stays there for the remaining time units 
        RMSE_end = RMSE_hist[end]
        trace_end = trace_hist[end]
        EI_end = EI_hist[end]
        for i = (length(xd)+1):total_budget
            append!(RMSE_hist, RMSE_end)
            append!(trace_hist, trace_end)
            append!(EI_hist, EI_end)
        end

    end
	return RMSE_hist, trace_hist, EI_hist
end

@everywhere function plot_drill_stats(drill_hist, trial_name, color)
    drill_hist_conc = []

    for i in 1:num_trials
        drill_hist_conc = vcat(drill_hist_conc, drill_hist[i][:])
    end
    bins = collect(0.0:(101/35):100)
    # histogram!(bins, drill_hist_conc, label=trial_name, color=color, xlim=(0,100), ylim=(0,50), xlabel="Trajectory Step", ylabel="Number of Drills", title="Drill Samples Along Trajectory")    
    global σ_spec
    global total_budget
    if σ_spec == 0.1 && total_budget == 30
        histogram!(drill_hist_conc, bins=bins, label=trial_name, color=color, xlim=(0,100), alpha=0.75, legend=true)    
    else
        histogram!(drill_hist_conc, bins=bins, label=trial_name, color=color, xlim=(0,100), alpha=0.75)
    end    
end

@everywhere function plot_traj_length_stats(traj_length_hist, trial_name, color)
    histogram!(traj_length_hist, nbins=15, label=trial_name, color=color, xlim=(50,100), ylim=(0,50), xlabel="Trajectory Length", ylabel="Number of Trajectories", title="Trajectory Length")    
end


@everywhere function plot_RMSE_trajectory_history_together(rmse_hist, trial_name, min_length, color)
	μ = []
	σ = []
	for i in 1:min_length
		mn = []
    	for j in 1:length(rmse_hist)
			append!(mn, rmse_hist[j][i])
		end
		append!(μ, mean(mn))
		append!(σ, sqrt(var(mn)))
	end

    title = ""
    xlabel = ""
    ylabel = ""

	plot!(collect(1:min_length), μ, ribbon = σ, xlabel=xlabel, ylabel=ylabel,title=title, legend=false, label=trial_name, color=color,fillalpha=0.2, size=(400,400), xlim=(1,total_budget))#, ylim=(0.23,0.5))
end

@everywhere function plot_objective_trajectory_history_together(trace_hist, trial_name, min_length, color, objective)
	μ = []
	σ = []
	for i in 1:min_length
		mn = []
    	for j in 1:length(trace_hist)
			append!(mn, trace_hist[j][i])
		end
		append!(μ, mean(mn))
		append!(σ, sqrt(var(mn)))
	end

    title = ""
    xlabel = ""
    ylabel = ""

    if objective == "expected_improvement"
        plot!(collect(1:min_length), μ ./ maximum(μ), ribbon = σ ./ maximum(μ), yscale=:log10, xlabel=xlabel, ylabel=ylabel,title=title, legend=false, label=trial_name, color=color,fillalpha=0.2, size=(400,400), xlim=(1,total_budget))#, ylim = (0.15, 1.0))#, ylim=(2000,8300))
    else
        plot!(collect(1:min_length), μ ./ maximum(μ), ribbon = σ ./ maximum(μ), xlabel=xlabel, ylabel=ylabel,title=title, legend=false, label=trial_name, color=color,fillalpha=0.2, size=(400,400), xlim=(1,total_budget))#, ylim = (0.15, 1.0))#, ylim=(2000,8300))
    end
end

function convert_shared_2_concat(hist)
    new_hist = []
    for i in 1:size(hist)[2]
        if length(hist[:,i]) == total_budget
            idx = length(hist[:,i])
        else
            idx = first(findall(x->x==0.0, hist[:,i])) - 1
        end
        new_hist = vcat(new_hist, [hist[1:idx,i]])
    end
    return new_hist
end


# GP MCTS
if !load_gp_mcts
    rmse_hist_gp_mcts = SharedArray{Float64}(N, num_trials)
    trace_hist_gp_mcts = SharedArray{Float64}(N, num_trials)
    EI_hist_gp_mcts = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_gp_mcts.jld")["rmse_hist_gp_mcts"]
    trace_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_gp_mcts.jld")["trace_hist_gp_mcts"]
    EI_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_gp_mcts.jld")["EI_hist_gp_mcts"]
    @show num_trials
end

# Ergodic
if !load_ergodic
    rmse_hist_ergodic = SharedArray{Float64}(N, num_trials)
    trace_hist_ergodic = SharedArray{Float64}(N, num_trials)
    EI_hist_ergodic = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_ergodic.jld")["rmse_hist_ergodic"]
    trace_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_ergodic.jld")["trace_hist_ergodic"]
    EI_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_ergodic.jld")["EI_hist_ergodic"]
end

# Ergodic GP
if !load_ergodic_gp
    rmse_hist_ergodicGP = SharedArray{Float64}(N, num_trials)
    trace_hist_ergodicGP = SharedArray{Float64}(N, num_trials)
    EI_hist_ergodicGP = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_ergodicGP.jld")["rmse_hist_ergodicGP"]
    trace_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_ergodicGP.jld")["trace_hist_ergodicGP"]
    EI_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_ergodicGP.jld")["EI_hist_ergodicGP"]
end

# GP PTO 
if !load_gp_pto
    rmse_hist_gp_pto = SharedArray{Float64}(N, num_trials)
    trace_hist_gp_pto = SharedArray{Float64}(N, num_trials)
    EI_hist_gp_pto = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_gp_pto.jld")["rmse_hist_gp_pto"]
    trace_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_gp_pto.jld")["trace_hist_gp_pto"]
    EI_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_gp_pto.jld")["EI_hist_gp_pto"]
end

# GP PTO Offline
if !load_gp_pto_offline
    rmse_hist_gp_pto_offline = SharedArray{Float64}(N, num_trials)
    trace_hist_gp_pto_offline = SharedArray{Float64}(N, num_trials)
    EI_hist_gp_pto_offline = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_gp_pto_offline.jld")["rmse_hist_gp_pto_offline"]
    trace_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_gp_pto_offline.jld")["trace_hist_gp_pto_offline"]
    EI_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_gp_pto_offline.jld")["EI_hist_gp_pto_offline"]
end

# Random
if !load_random
    rmse_hist_random = SharedArray{Float64}(N, num_trials)
    trace_hist_random = SharedArray{Float64}(N, num_trials)
    EI_hist_random = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_random = load(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_random.jld")["rmse_hist_random"]
    trace_hist_random = load(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_random.jld")["trace_hist_random"]
    EI_hist_random = load(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_random.jld")["EI_hist_random"]
end



@sync @distributed for i = 1:num_trials
    @show i

    true_map = load(path_name * "/true_maps/true_map$(i).jld")["true_map"]


    ##########################
    # GP MCTS
    ##########################
    global rmse_hist_gp_mcts
    global trace_hist_gp_mcts
    global EI_hist_gp_mcts

    if !load_gp_mcts
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
        xd = executed_traj_gp_mcts#[1:traj_length]
        drill_idxs = findall(x->x==1.0, executed_drills_gp_mcts)

        RMSE_hist, trace_hist, EI_hist = calculate_RMSE_TrΣ(true_map, xd, gp_X_gp_mcts, gp_y_gp_mcts, gp_ν_gp_mcts, "SBO_AIPPMS")
        # rmse_hist_gp_mcts = vcat(rmse_hist_gp_mcts, [RMSE_hist])
        # trace_hist_gp_mcts = vcat(trace_hist_gp_mcts, [trace_hist])

        rmse_hist_gp_mcts[1:length(RMSE_hist), i] = RMSE_hist
        trace_hist_gp_mcts[1:length(trace_hist), i] = trace_hist
        EI_hist_gp_mcts[1:length(EI_hist), i] = EI_hist


        if plot_gifs
            plot_trial_mean(xd, drill_idxs, gp_X_gp_mcts, gp_y_gp_mcts, gp_ν_gp_mcts, i, "SBO_AIPPMS")
            plot_trial_objective(xd, drill_idxs, gp_X_gp_mcts, gp_y_gp_mcts, gp_ν_gp_mcts, i, "SBO_AIPPMS")
        end
    end

    ##########################
    # Ergodic
    ##########################
    global rmse_hist_ergodic
    global trace_hist_ergodic
    global EI_hist_ergodic

    if !load_ergodic
        executed_traj_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
        executed_drills_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
        gp_X_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
        gp_y_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
        gp_ν_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]
        
        xd = executed_traj_ergodic
        drill_idxs = executed_drills_ergodic

        RMSE_hist, trace_hist, EI_hist = calculate_RMSE_TrΣ(true_map, xd, gp_X_ergodic, gp_y_ergodic, gp_ν_ergodic, "Ergodic")
        # rmse_hist_ergodic = vcat(rmse_hist_ergodic, [RMSE_hist])
        # trace_hist_ergodic = vcat(trace_hist_ergodic, [trace_hist])

        rmse_hist_ergodic[1:length(RMSE_hist), i] = RMSE_hist
        trace_hist_ergodic[1:length(trace_hist), i] = trace_hist
        EI_hist_ergodic[1:length(EI_hist), i] = EI_hist

        if plot_gifs
            plot_trial_mean(xd, drill_idxs, gp_X_ergodic, gp_y_ergodic, gp_ν_ergodic, i, "Ergodic")
            plot_trial_objective(xd, drill_idxs, gp_X_ergodic, gp_y_ergodic, gp_ν_ergodic, i, "Ergodic")
        end
    end

    ##########################
    # Ergodic GP
    ##########################
    global rmse_hist_ergodicGP
    global trace_hist_ergodicGP
    global EI_hist_ergodicGP

    if !load_ergodic_gp
        executed_traj_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
        executed_drills_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
        gp_X_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
        gp_y_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
        gp_ν_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]
        
        xd = executed_traj_ergodicGP
        drill_idxs = executed_drills_ergodicGP
        # drill_idxs = []
        # for i in 1:length(executed_drills_ergodicGP)
        #     idx = findall(x->x==executed_drills_ergodicGP[i], xd)
        #     append!(drill_idxs, idx)
        # end
        # @show drill_idxs

        RMSE_hist, trace_hist, EI_hist = calculate_RMSE_TrΣ(true_map, xd, gp_X_ergodicGP, gp_y_ergodicGP, gp_ν_ergodicGP, "ErgodicGP")
        # rmse_hist_ergodicGP = vcat(rmse_hist_ergodicGP, [RMSE_hist])
        # trace_hist_ergodicGP = vcat(trace_hist_ergodicGP, [trace_hist])

        rmse_hist_ergodicGP[1:length(RMSE_hist), i] = RMSE_hist
        trace_hist_ergodicGP[1:length(trace_hist), i] = trace_hist
        EI_hist_ergodicGP[1:length(EI_hist), i] = EI_hist

        if plot_gifs
            plot_trial_mean(xd, drill_idxs, gp_X_ergodicGP, gp_y_ergodicGP, gp_ν_ergodicGP, i, "ErgodicGP")
            plot_trial_objective(xd, drill_idxs, gp_X_ergodicGP, gp_y_ergodicGP, gp_ν_ergodicGP, i, "ErgodicGP")
        end
    end

    ##########################
    # GP PTO
    ##########################
    global rmse_hist_gp_pto
    global trace_hist_gp_pto
    global EI_hist_gp_pto

    if !load_gp_pto
        executed_traj_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
        executed_drills_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
        gp_X_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
        gp_y_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
        gp_ν_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]
        
        xd = executed_traj_gp_pto
        drill_idxs = executed_drills_gp_pto

        RMSE_hist, trace_hist, EI_hist = calculate_RMSE_TrΣ(true_map, xd, gp_X_gp_pto, gp_y_gp_pto, gp_ν_gp_pto, "GP_PTO")
        # rmse_hist_gp_pto = vcat(rmse_hist_gp_pto, [RMSE_hist])
        # trace_hist_gp_pto = vcat(trace_hist_gp_pto, [trace_hist])

        rmse_hist_gp_pto[1:length(RMSE_hist), i] = RMSE_hist
        trace_hist_gp_pto[1:length(trace_hist), i] = trace_hist
        EI_hist_gp_pto[1:length(EI_hist), i] = EI_hist

        if plot_gifs
            plot_trial_mean(xd, drill_idxs, gp_X_gp_pto, gp_y_gp_pto, gp_ν_gp_pto, i, "GP_PTO")
            plot_trial_objective(xd, drill_idxs, gp_X_gp_pto, gp_y_gp_pto, gp_ν_gp_pto, i, "GP_PTO")
        end
    end


    ##########################
    # GP PTO Offline
    ##########################
    global rmse_hist_gp_pto_offline
    global trace_hist_gp_pto_offline
    global EI_hist_gp_pto_offline

    if !load_gp_pto_offline
        executed_traj_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
        executed_drills_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
        gp_X_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
        gp_y_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
        gp_ν_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]
        
        xd = executed_traj_gp_pto_offline
        drill_idxs = executed_drills_gp_pto_offline

        RMSE_hist, trace_hist, EI_hist = calculate_RMSE_TrΣ(true_map, xd, gp_X_gp_pto_offline, gp_y_gp_pto_offline, gp_ν_gp_pto_offline, "GP_PTO_offline")
        # rmse_hist_gp_pto_offline = vcat(rmse_hist_gp_pto_offline, [RMSE_hist])
        # trace_hist_gp_pto_offline = vcat(trace_hist_gp_pto_offline, [trace_hist])

        rmse_hist_gp_pto_offline[1:length(RMSE_hist), i] = RMSE_hist
        trace_hist_gp_pto_offline[1:length(trace_hist), i] = trace_hist
        EI_hist_gp_pto_offline[1:length(EI_hist), i] = EI_hist

        if plot_gifs
            plot_trial_mean(xd, drill_idxs, gp_X_gp_pto_offline, gp_y_gp_pto_offline, gp_ν_gp_pto_offline, i, "GP_PTO_offline")
            plot_trial_objective(xd, drill_idxs, gp_X_gp_pto_offline, gp_y_gp_pto_offline, gp_ν_gp_pto_offline, i, "GP_PTO_offline")
        end
    end

    ##########################
    # Random
    ##########################
    global rmse_hist_random
    global trace_hist_random
    global EI_hist_random

    if !load_random
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

        xd = executed_traj_random#[1:traj_length]
        drill_idxs = findall(x->x==1.0, executed_drills_random)

        RMSE_hist, trace_hist, EI_hist = calculate_RMSE_TrΣ(true_map, xd, gp_X_random, gp_y_random, gp_ν_random, "Random")

        rmse_hist_random[1:length(RMSE_hist), i] = RMSE_hist
        trace_hist_random[1:length(trace_hist), i] = trace_hist
        EI_hist_random[1:length(EI_hist), i] = EI_hist


        if plot_gifs
            plot_trial_mean(xd, drill_idxs, gp_X_random, gp_y_random, gp_ν_random, i, "Random")
            plot_trial_objective(xd, drill_idxs, gp_X_random, gp_y_random, gp_ν_random, i, "Random")
        end
    end

end

# Convert from SharedArray Structure to original concat version
if !load_gp_mcts
    rmse_hist_gp_mcts = convert_shared_2_concat(rmse_hist_gp_mcts)
    trace_hist_gp_mcts = convert_shared_2_concat(trace_hist_gp_mcts)
    EI_hist_gp_mcts = convert_shared_2_concat(EI_hist_gp_mcts)
end

if !load_ergodic
    rmse_hist_ergodic = convert_shared_2_concat(rmse_hist_ergodic)
    trace_hist_ergodic = convert_shared_2_concat(trace_hist_ergodic)
    EI_hist_ergodic = convert_shared_2_concat(EI_hist_ergodic)
end

if !load_ergodic_gp
    rmse_hist_ergodicGP = convert_shared_2_concat(rmse_hist_ergodicGP)
    trace_hist_ergodicGP = convert_shared_2_concat(trace_hist_ergodicGP)
    EI_hist_ergodicGP = convert_shared_2_concat(EI_hist_ergodicGP)
end

if !load_gp_pto
    rmse_hist_gp_pto = convert_shared_2_concat(rmse_hist_gp_pto)
    trace_hist_gp_pto = convert_shared_2_concat(trace_hist_gp_pto)
    EI_hist_gp_pto = convert_shared_2_concat(EI_hist_gp_pto)
end

if !load_gp_pto_offline
    rmse_hist_gp_pto_offline = convert_shared_2_concat(rmse_hist_gp_pto_offline)
    trace_hist_gp_pto_offline = convert_shared_2_concat(trace_hist_gp_pto_offline)
    EI_hist_gp_pto_offline = convert_shared_2_concat(EI_hist_gp_pto_offline)
end

if !load_random
    rmse_hist_random = convert_shared_2_concat(rmse_hist_random)
    trace_hist_random = convert_shared_2_concat(trace_hist_random)
    EI_hist_random = convert_shared_2_concat(EI_hist_random)
end

##############################################
# PLOT RMSE
##############################################
trial_names = ["MCTS-DPW", "Ergodic", "Ergodic-GP", "GP-PTO", "GP-PTO Offline", "Random"]
# trial_names = ["MCTS-DPW", "Ergodic", "Ergodic-GP", "GP-PTO", "Random"]

hists = [rmse_hist_gp_mcts, rmse_hist_ergodic, rmse_hist_ergodicGP, rmse_hist_gp_pto, rmse_hist_gp_pto_offline, rmse_hist_random]
# hists = [rmse_hist_gp_mcts, rmse_hist_ergodic, rmse_hist_ergodicGP, rmse_hist_gp_pto, rmse_hist_random]

# hists = [rmse_hist_raster_corrected, rmse_hist_gp_mcts_corrected, rmse_hist_basic_corrected, rmse_hist_gcb_corrected]
colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :green, :red, :orchid1, :orange]
# alphas = [0.4, 0.3, 0.2, 0.1]

@show [length(hists[i]) for i in 1:length(hists)]
min_length = []
for hist in hists
    append!(min_length, minimum([length(hist[i]) for i in 1:length(hist)]))
end
# min_length = minimum(min_length)
# @show min_length

Plots.plot()
for i in 1:length(hists)
	plot_RMSE_trajectory_history_together(hists[i], trial_names[i], min_length[i], colors[i])
end


savefig(save_path * "/RMSE_traj_together_total_budget_$(total_budget)_σ_spec_$(σ_spec).pdf")


##############################################
# PLOT Tr(Σ)
##############################################
trial_names = ["MCTS-DPW", "Ergodic", "Ergodic-GP", "GP-PTO", "GP-PTO Offline", "Random"]

hists = [trace_hist_gp_mcts, trace_hist_ergodic, trace_hist_ergodicGP, trace_hist_gp_pto, trace_hist_gp_pto_offline, trace_hist_random]
# hists = [rmse_hist_raster_corrected, rmse_hist_gp_mcts_corrected, rmse_hist_basic_corrected, rmse_hist_gcb_corrected]
colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :green, :red, :orchid1, :orange]
# alphas = [0.4, 0.3, 0.2, 0.1]

min_length = []
for hist in hists
    append!(min_length, minimum([length(hist[i]) for i in 1:length(hist)]))
end
# min_length = minimum(min_length)
# @show min_length

Plots.plot()
for i in 1:length(hists)
	plot_objective_trajectory_history_together(hists[i], trial_names[i], min_length[i], colors[i], "variance")

    hist = hists[i]
    var_reduction_hist = [(maximum(hist[i]) - minimum(hist[i]))/maximum(hist[i]) for i in 1:length(hist)]
    println(trial_names[i] * " Average Variance Reduction: ", mean(var_reduction_hist))
    println(trial_names[i] * " Stddev Variance Reduction: ", std(var_reduction_hist))
end


savefig(save_path * "/Trace_traj_together_total_budget_$(total_budget)_σ_spec_$(σ_spec).pdf")

##############################################
# PLOT EI
##############################################
trial_names = ["MCTS-DPW", "Ergodic", "Ergodic-GP", "GP-PTO", "GP-PTO Offline", "Random"]

hists = [EI_hist_gp_mcts, EI_hist_ergodic, EI_hist_ergodicGP, EI_hist_gp_pto, EI_hist_gp_pto_offline, EI_hist_random]
# hists = [rmse_hist_raster_corrected, rmse_hist_gp_mcts_corrected, rmse_hist_basic_corrected, rmse_hist_gcb_corrected]
colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :green, :red, :orchid1, :orange]
# alphas = [0.4, 0.3, 0.2, 0.1]

min_length = []
for hist in hists
    append!(min_length, minimum([length(hist[i]) for i in 1:length(hist)]))
end
# min_length = minimum(min_length)
# @show min_length

Plots.plot()
for i in 1:length(hists)
	plot_objective_trajectory_history_together(hists[i], trial_names[i], min_length[i], colors[i], "expected_improvement")

    hist = hists[i]
    EI_reduction_hist = [(maximum(hist[i]) - minimum(hist[i]))/maximum(hist[i]) for i in 1:length(hist)]
    println(trial_names[i] * " Average EI Reduction: ", mean(EI_reduction_hist))
    println(trial_names[i] * " Stddev EI Reduction: ", std(EI_reduction_hist))
end


savefig(save_path * "/EI_traj_together_total_budget_$(total_budget)_σ_spec_$(σ_spec).pdf")


##############################################
# PLOT Trajectory Stats
##############################################
trial_names = ["MCTS-DPW", "Ergodic", "GP-PTO", "GP-PTO Offline", "Ergodic-GP", "Random"]
dir_names = ["SBO_AIPPMS", "Ergodic", "GP_PTO", "GP_PTO_offline", "ErgodicGP", "Random"]
# colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :red, :green, :orchid1]
colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :darkorange, :darkseagreen, :orchid1, :orange]


Plots.plot()
for n = 1:length(dir_names)
    name = dir_names[n]
    drill_hist = [findall(x->x.==1.0e-9, load(path_name * "/" * name * "/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"])  for i in 1:num_trials]
    plot_drill_stats(drill_hist, trial_names[n], colors[n])
    savefig(save_path * "/Drill_stats_$(total_budget)_σ_spec_$(σ_spec).pdf")
end

Plots.plot()
for n = 1:length(dir_names)
    name = dir_names[n]
    traj_length_hist = [length(load(path_name * "/" * name * "/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"])  for i in 1:num_trials]
    plot_traj_length_stats(traj_length_hist, trial_names[n], colors[n])
    savefig(save_path * "/Traj_length_stats_$(total_budget)_σ_spec_$(σ_spec).pdf")
end
############################################################################################

if plot_final_subplot
    # σ_s = [0.1, 0.5, 1.0]
    σ_s = [0.1, 0.5]
    budgets = [30.0, 60.0, 100.0]
    plts_rmse = []
    plts_trace = []
    plts_EI = []
    plts_drill_stats = []
    for σ in σ_s
        for budget in budgets
            global σ_spec
            global total_budget
            σ_spec = σ
            total_budget = budget

            rmse_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_gp_mcts.jld")["rmse_hist_gp_mcts"]
            trace_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_gp_mcts.jld")["trace_hist_gp_mcts"]
            EI_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_gp_mcts.jld")["EI_hist_gp_mcts"]

            rmse_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_ergodic.jld")["rmse_hist_ergodic"]
            trace_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_ergodic.jld")["trace_hist_ergodic"]
            EI_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_ergodic.jld")["EI_hist_ergodic"]

            rmse_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_ergodicGP.jld")["rmse_hist_ergodicGP"]
            trace_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_ergodicGP.jld")["trace_hist_ergodicGP"]
            EI_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_ergodicGP.jld")["EI_hist_ergodicGP"]

            rmse_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_gp_pto.jld")["rmse_hist_gp_pto"]
            trace_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_gp_pto.jld")["trace_hist_gp_pto"]
            EI_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_gp_pto.jld")["EI_hist_gp_pto"]

            rmse_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_gp_pto_offline.jld")["rmse_hist_gp_pto_offline"]
            trace_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_gp_pto_offline.jld")["trace_hist_gp_pto_offline"]
            EI_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_gp_pto_offline.jld")["EI_hist_gp_pto_offline"]

            rmse_hist_random = load(path_name * "/Random/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_random.jld")["rmse_hist_random"]
            trace_hist_random = load(path_name * "/Random/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_random.jld")["trace_hist_random"]
            EI_hist_random = load(path_name * "/Random/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_random.jld")["EI_hist_random"]

            ##########################################################################################
            trial_names = ["MCTS-DPW", "Ergodic", "Ergodic-GP", "GP-PTO", "GP-PTO Offline", "Random"]

            hists = [rmse_hist_gp_mcts, rmse_hist_ergodic, rmse_hist_ergodicGP, rmse_hist_gp_pto, rmse_hist_gp_pto_offline, rmse_hist_random]
            colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :green, :red, :orchid1, :orange]

            min_length = []
            for hist in hists
                append!(min_length, minimum([length(hist[i]) for i in 1:length(hist)]))
            end

            plt = Plots.plot()
            for i in 1:length(hists)
                plt = plot_RMSE_trajectory_history_together(hists[i], trial_names[i], min_length[i], colors[i])
            end

            push!(plts_rmse, plt)

            ##########################################################################################
            hists = [trace_hist_gp_mcts, trace_hist_ergodic, trace_hist_ergodicGP, trace_hist_gp_pto, trace_hist_gp_pto_offline, trace_hist_random]
            colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :green, :red, :orchid1, :orange]

            min_length = []
            for hist in hists
                append!(min_length, minimum([length(hist[i]) for i in 1:length(hist)]))
            end

            plt = Plots.plot()
            for i in 1:length(hists)
                plt = plot_objective_trajectory_history_together(hists[i], trial_names[i], min_length[i], colors[i], "variance")
            end

            push!(plts_trace, plt)

            ##########################################################################################
            hists = [EI_hist_gp_mcts, EI_hist_ergodic, EI_hist_ergodicGP, EI_hist_gp_pto, EI_hist_gp_pto_offline, EI_hist_random]
            colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :green, :red, :orchid1, :orange]

            min_length = []
            for hist in hists
                append!(min_length, minimum([length(hist[i]) for i in 1:length(hist)]))
            end

            plt = Plots.plot()
            for i in 1:length(hists)
                plt = plot_objective_trajectory_history_together(hists[i], trial_names[i], min_length[i], colors[i], "expected_improvement")
            end

            push!(plts_EI, plt)

            ##########################################################################################
            if objective == "variance" && σ == 1.0 && budget == 100.0
                trial_names = ["Ergodic", "MCTS-DPW", "Random", "Ergodic-GP", "GP-PTO", "GP-PTO Offline"]
                dir_names = ["Ergodic", "SBO_AIPPMS", "Random", "ErgodicGP", "GP_PTO", "GP_PTO_offline"]
                colors = [:black, RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :orange, :green, :red, :orchid1]
            elseif objective == "variance" && σ == 0.5 && budget == 30.0
                trial_names = ["Ergodic", "Ergodic-GP", "GP-PTO", "GP-PTO Offline", "Random", "MCTS-DPW"]
                dir_names = ["Ergodic", "ErgodicGP", "GP_PTO", "GP_PTO_offline", "Random", "SBO_AIPPMS"]
                colors = [:black, :green, :red, :orchid1, :orange, RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073)]
            else
                trial_names = ["Ergodic", "Random", "Ergodic-GP", "GP-PTO", "GP-PTO Offline", "MCTS-DPW"]
                dir_names = ["Ergodic", "Random", "ErgodicGP", "GP_PTO", "GP_PTO_offline", "SBO_AIPPMS"]
                colors = [:black, :orange, :green, :red, :orchid1, RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073)]
            end

            plt = Plots.plot()
            for n = 1:length(dir_names)
                name = dir_names[n]
                drill_hist = [findall(x->x.==1.0e-9, load(path_name * "/" * name * "/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"])  for i in 1:num_trials]
                plt = plot_drill_stats(drill_hist, trial_names[n], colors[n])
            end
            push!(plts_drill_stats, plt)
        end
    end
    plot(plts_rmse..., legend=false, size=(1200,1200))
    # plot(plts_rmse..., legend=false, size=(1200,800))
    savefig(save_path * "/All_RMSE.pdf")

    # plot(plts_trace..., legend=false, size=(1200,800))
    plot(plts_trace..., legend=false, size=(1200,1200))
    savefig(save_path * "/All_Trace.pdf")

    plot(plts_EI..., legend=false, size=(1200,800))
    # plot(plts_trace..., legend=false, size=(1200,1200), ylabel="Tr(Σ)", xlabel="Trajectory Step")
    savefig(save_path * "/All_EI.pdf")

    plot(plts_drill_stats..., legend=false, size=(1200,400), layout=(1,3), margin=5mm)
    savefig(save_path * "/All_Drill_Stats.pdf")
end




if !load_gp_mcts
    JLD2.save(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_gp_mcts.jld", "rmse_hist_gp_mcts", rmse_hist_gp_mcts)
    JLD2.save(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_gp_mcts.jld", "trace_hist_gp_mcts", trace_hist_gp_mcts)
    JLD2.save(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_gp_mcts.jld", "EI_hist_gp_mcts", EI_hist_gp_mcts)
end

if !load_ergodic
    JLD2.save(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_ergodic.jld", "rmse_hist_ergodic", rmse_hist_ergodic)
    JLD2.save(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_ergodic.jld", "trace_hist_ergodic", trace_hist_ergodic)
    JLD2.save(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_ergodic.jld", "EI_hist_ergodic", EI_hist_ergodic)
end

if !load_ergodic_gp
    JLD2.save(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_ergodicGP.jld", "rmse_hist_ergodicGP", rmse_hist_ergodicGP)
    JLD2.save(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_ergodicGP.jld", "trace_hist_ergodicGP", trace_hist_ergodicGP)
    JLD2.save(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_ergodicGP.jld", "EI_hist_ergodicGP", EI_hist_ergodicGP)
end

if !load_gp_pto
    JLD2.save(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_gp_pto.jld", "rmse_hist_gp_pto", rmse_hist_gp_pto)
    JLD2.save(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_gp_pto.jld", "trace_hist_gp_pto", trace_hist_gp_pto)
    JLD2.save(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_gp_pto.jld", "EI_hist_gp_pto", EI_hist_gp_pto)
end

if !load_gp_pto_offline
    JLD2.save(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_gp_pto_offline.jld", "rmse_hist_gp_pto_offline", rmse_hist_gp_pto_offline)
    JLD2.save(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_gp_pto_offline.jld", "trace_hist_gp_pto_offline", trace_hist_gp_pto_offline)
    JLD2.save(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_gp_pto_offline.jld", "EI_hist_gp_pto_offline", EI_hist_gp_pto_offline)
end

if !load_random
    JLD2.save(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_random.jld", "rmse_hist_random", rmse_hist_random)
    JLD2.save(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_random.jld", "trace_hist_random", trace_hist_random)
    JLD2.save(path_name * "/Random/total_budget_$(total_budget)/σ_spec_$(σ_spec)/EI_hist_random.jld", "EI_hist_random", EI_hist_random)
end



# theme(:default)

# σ = 1.0
# budget = 60.0
# σ_spec = σ
# total_budget = budget

# rmse_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_gp_mcts.jld")["rmse_hist_gp_mcts"]
# trace_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_gp_mcts.jld")["trace_hist_gp_mcts"]
# EI_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_gp_mcts.jld")["EI_hist_gp_mcts"]

# rmse_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_ergodic.jld")["rmse_hist_ergodic"]
# trace_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_ergodic.jld")["trace_hist_ergodic"]
# EI_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_ergodic.jld")["EI_hist_ergodic"]

# rmse_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_ergodicGP.jld")["rmse_hist_ergodicGP"]
# trace_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_ergodicGP.jld")["trace_hist_ergodicGP"]
# EI_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_ergodicGP.jld")["EI_hist_ergodicGP"]

# rmse_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_gp_pto.jld")["rmse_hist_gp_pto"]
# trace_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_gp_pto.jld")["trace_hist_gp_pto"]
# EI_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_gp_pto.jld")["EI_hist_gp_pto"]

# rmse_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_gp_pto_offline.jld")["rmse_hist_gp_pto_offline"]
# trace_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_gp_pto_offline.jld")["trace_hist_gp_pto_offline"]
# EI_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_gp_pto_offline.jld")["EI_hist_gp_pto_offline"]

# rmse_hist_random = load(path_name * "/Random/total_budget_$(budget)/σ_spec_$(σ)/rmse_hist_random.jld")["rmse_hist_random"]
# trace_hist_random = load(path_name * "/Random/total_budget_$(budget)/σ_spec_$(σ)/trace_hist_random.jld")["trace_hist_random"]
# EI_hist_random = load(path_name * "/Random/total_budget_$(budget)/σ_spec_$(σ)/EI_hist_random.jld")["EI_hist_random"]


# hists = [EI_hist_gp_mcts, EI_hist_ergodic, EI_hist_ergodicGP, EI_hist_gp_pto, EI_hist_gp_pto_offline, EI_hist_random]
# colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :green, :red, :orchid1, :orange]

# min_length = []
# for hist in hists
#     append!(min_length, minimum([length(hist[i]) for i in 1:length(hist)]))
# end

# plt = Plots.plot()
# for i in 1:length(hists)
#     plt = plot_objective_trajectory_history_together(hists[i], trial_names[i], min_length[i], colors[i], "expected_improvement")
# end

# push!(plts_EI, plt)

# ##########################################################################################
# if objective == "variance" && σ == 1.0 && budget == 100.0
#     trial_names = ["Ergodic", "MCTS-DPW", "Random", "Ergodic-GP", "GP-PTO", "GP-PTO Offline"]
#     dir_names = ["Ergodic", "SBO_AIPPMS", "Random", "ErgodicGP", "GP_PTO", "GP_PTO_offline"]
#     colors = [:black, RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :orange, :green, :red, :orchid1]
# elseif objective == "variance" && σ == 0.5 && budget == 30.0
#     trial_names = ["Ergodic", "Ergodic-GP", "GP-PTO", "GP-PTO Offline", "Random", "MCTS-DPW"]
#     dir_names = ["Ergodic", "ErgodicGP", "GP_PTO", "GP_PTO_offline", "Random", "SBO_AIPPMS"]
#     colors = [:black, :green, :red, :orchid1, :orange, RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073)]
# else
#     trial_names = ["Ergodic", "Random", "Ergodic-GP", "GP-PTO", "GP-PTO Offline", "MCTS-DPW"]
#     dir_names = ["Ergodic", "Random", "ErgodicGP", "GP_PTO", "GP_PTO_offline", "SBO_AIPPMS"]
#     colors = [:black, :orange, :green, :red, :orchid1, RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073)]
# end

# plot(plts_EI..., legend=false, size=(1200,1200))



# μμ = []
# σ = []
# hist = hists[1]
# for i in 1:min_length[1]
#     mn = []
#     for j in 1:length(hist)
#         append!(mn, hist[j][i])
#     end
#     append!(μμ, mean(mn))
#     append!(σ, sqrt(var(mn)))
# end
# plot(collect(1:min_length[1]), μμ ./ maximum(μμ) .+ 1e-3, ribbon = σ ./ maximum(μμ) .+ 1e-3, yscale=:log10, legend=false, fillalpha=0.2, size=(400,400), xlim=(1,total_budget))#, ylim = (0.15, 1.0))#, ylim=(2000,8300))
