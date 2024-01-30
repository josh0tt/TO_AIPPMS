# results.jl
using Distributed
using SharedArrays
using Plots
using DelimitedFiles
using StatsBase
using JLD2
using KernelFunctions
using LinearAlgebra
using Colors

include("CustomGP.jl")
include("parameters.jl")

load_gp_mcts = false
load_ergodic = false
load_ergodic_gp = false
load_gp_pto = false
load_gp_pto_offline = false

theme(:dao)

function plot_rmse_comparison(true_map, xd, gp_X, gp_y, gp_ν, trial_name, trial_num)
    plot_scale = 0:0.01:1
    true_interp_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))
    measured_interp_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))

    for i in 1:size(true_interp_map)[1]
        for j in 1:size(true_interp_map)[2]
            idx_i = Int(floor(i/10) + 1)
            idx_j = Int(floor(j/10) + 1)
            true_interp_map[i,j] = true_map[idx_i, idx_j]
        end
    end

    x = [xd[i][1] for i in 1:length(xd)]
    y = [xd[i][2] for i in 1:length(xd)]
    map_size = size(true_interp_map)

    @show trial_name

    if trial_name == "SBO_AIPPMS"
        @show all(xd .== gp_X) 
        for i = 1:length(xd)
            # pos = LinearIndices(size(true_map))[Int(x[i]*10 + 1), Int(y[i]*10 + 1)]
            pos = LinearIndices(map_size)[Int(x[i]*100 + 1), Int(y[i]*100 + 1)]

            # x_samp = [y[i], map_size[1] - x[i]]
                
            # pos_x = Int(round(x_samp[1]*100 + 1, digits=0))
            # if pos_x > map_size[2]
            #     pos_x = map_size[2]
            # elseif pos_x < 1
            #     pos_x = 1
            # end
    
            # pos_y = map_size[1] - Int(round(x_samp[2]*100 + 1, digits=0))
            # if pos_y > map_size[1]
            #     pos_y = map_size[1]
            # elseif pos_y < 1
            #     pos_y = 1
            # end
    
            # pos = LinearIndices(map_size)[pos_y, pos_x]
            measured_interp_map[pos] = gp_y[i]
    
            if gp_ν[i] == 1.0e-9
                # @show gp_y[i] - true_map[pos]
                @show gp_y[i] - true_interp_map[pos]

            end
        end

    else
         
        for i = 1:length(xd)
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
            measured_interp_map[pos] = gp_y[i]

            if gp_ν[i] == 1.0e-9
                @show gp_y[i] - true_interp_map[pos]
            end
        end
    end

end


# GP MCTS
if !load_gp_mcts
    # rmse_hist_gp_mcts = []
    # trace_hist_gp_mcts = []
    rmse_hist_gp_mcts = SharedArray{Float64}(N, num_trials)
    trace_hist_gp_mcts = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_gp_mcts.jld")["rmse_hist_gp_mcts"]
    trace_hist_gp_mcts = load(path_name * "/SBO_AIPPMS/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_gp_mcts.jld")["trace_hist_gp_mcts"]
    #num_trials = size(executed_traj_gp_mcts)[2]
    @show num_trials
end

# Ergodic
if !load_ergodic
    rmse_hist_ergodic = SharedArray{Float64}(N, num_trials)
    trace_hist_ergodic = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_ergodic.jld")["rmse_hist_ergodic"]
    trace_hist_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_ergodic.jld")["trace_hist_ergodic"]
end

# Ergodic GP
if !load_ergodic_gp
    rmse_hist_ergodicGP = SharedArray{Float64}(N, num_trials)
    trace_hist_ergodicGP = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_ergodicGP.jld")["rmse_hist_ergodicGP"]
    trace_hist_ergodicGP = load(path_name * "/ErgodicGP/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_ergodicGP.jld")["trace_hist_ergodicGP"]
end

# GP PTO 
if !load_gp_pto
    rmse_hist_gp_pto = SharedArray{Float64}(N, num_trials)
    trace_hist_gp_pto = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_gp_pto.jld")["rmse_hist_gp_pto"]
    trace_hist_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_gp_pto.jld")["trace_hist_gp_pto"]
end

# GP PTO Offline
if !load_gp_pto_offline
    rmse_hist_gp_pto_offline = SharedArray{Float64}(N, num_trials)
    trace_hist_gp_pto_offline = SharedArray{Float64}(N, num_trials)
else
    rmse_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/rmse_hist_gp_pto_offline.jld")["rmse_hist_gp_pto_offline"]
    trace_hist_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/trace_hist_gp_pto_offline.jld")["trace_hist_gp_pto_offline"]
end



for i = 1:num_trials
    @show i

    true_map = load(path_name * "/true_maps/true_map$(i).jld")["true_map"]


    ##########################
    # GP MCTS
    ##########################
    global rmse_hist_gp_mcts
    global trace_hist_gp_mcts

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

        xd = executed_traj_gp_mcts#[1:traj_length]
        drill_idxs = findall(x->x==1.0, executed_drills_gp_mcts)

        plot_rmse_comparison(true_map, xd, gp_X_gp_mcts, gp_y_gp_mcts, gp_ν_gp_mcts, "SBO_AIPPMS", i)

    end

    ##########################
    # Ergodic
    ##########################
    global rmse_hist_ergodic
    global trace_hist_ergodic

    if !load_ergodic
        executed_traj_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
        executed_drills__ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
        gp_X_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
        gp_y_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
        gp_ν_ergodic = load(path_name * "/Ergodic/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]
        
        xd = executed_traj_ergodic
        drill_idxs = executed_drills__ergodic

        plot_rmse_comparison(true_map, xd, gp_X_ergodic, gp_y_ergodic, gp_ν_ergodic, "Ergodic", i)
    end

    ##########################
    # Ergodic GP
    ##########################
    global rmse_hist_ergodicGP
    global trace_hist_ergodicGP

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

        plot_rmse_comparison(true_map, xd, gp_X_ergodicGP, gp_y_ergodicGP, gp_ν_ergodicGP, "ErgodicGP", i)

    end

    ##########################
    # GP PTO
    ##########################
    global rmse_hist_gp_pto
    global trace_hist_gp_pto

    if !load_gp_pto
        executed_traj_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
        executed_drills_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
        gp_X_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
        gp_y_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
        gp_ν_gp_pto = load(path_name * "/GP_PTO/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]
        
        xd = executed_traj_gp_pto
        drill_idxs = executed_drills_gp_pto

        plot_rmse_comparison(true_map, xd, gp_X_gp_pto, gp_y_gp_pto, gp_ν_gp_pto, "GP_PTO", i)

    end


    ##########################
    # GP PTO Offline
    ##########################
    global rmse_hist_gp_pto_offline
    global trace_hist_gp_pto_offline

    if !load_gp_pto_offline
        executed_traj_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_traj$(i).jld")["executed_traj"]
        executed_drills_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/executed/executed_drill_locations$(i).jld")["executed_drill_locations"]
        gp_X_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_X$(i).jld")["final_gp_X"]
        gp_y_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_y$(i).jld")["final_gp_y"]
        gp_ν_gp_pto_offline = load(path_name * "/GP_PTO_offline/total_budget_$(total_budget)/σ_spec_$(σ_spec)/gp_hist/final_gp_ν$(i).jld")["final_gp_ν"]
        
        xd = executed_traj_gp_pto_offline
        drill_idxs = executed_drills_gp_pto_offline

        plot_rmse_comparison(true_map, xd, gp_X_gp_pto_offline, gp_y_gp_pto_offline, gp_ν_gp_pto_offline, "GP_PTO_offline", i)
    end

end