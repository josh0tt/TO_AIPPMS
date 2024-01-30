include("ErgodicControl.jl")
# using .ErgodicControl
using Plots
ENV["GKSwstype"]="nul" 

using KernelFunctions
using LinearAlgebra
using JLD2
include("../../CustomGP.jl")
include("../../parameters.jl")
include("plot_ergodic.jl")

# Ergodic GP

function run_ergodic_mpc(K, em, replan_rate, N, h, sample_cost, phi)

    i = 1 # counter to indicate when it's time to replan
    t = 1

    tm = TrajectoryManager(x0, xf, h, N, ConstantInitializer([0.0,0.0]));
    # tm = TrajectoryManager(x0, xf, h, N, CornerInitializer());
    tm.barrier_cost=1000.0
    tm.sample_cost = sample_cost
    xd, ud = pto_trajectory(em, tm, verbose=false, logging=true);
    # xd, ud = smc_trajectory(em, tm, verbose=false);
    executed_traj = []

    phi_storage = zeros(bins_x+1, bins_y+1, 1+convert(Int, floor(N/replan_rate)))
    xd_storage = Matrix{Vector{Float64}}(undef, N+1, 1+convert(Int, floor(N/replan_rate)))
    while t < N
        if i >= replan_rate
            xd_storage[1:length(xd), convert(Int, floor(t/replan_rate))] = xd
            phi_storage[:,:, convert(Int, floor(t/replan_rate))] = phi
            
            ck = decompose(em, xd[1:replan_rate])
            phi_k = reconstruct(em, ck)
            Ta = replan_rate
            Tb = N-t
            phi = ((Ta+Tb)/Tb)*(em.phi .- Ta/(Ta+Tb).*phi_k)
            
            em = ErgodicManagerR2(em.domain, phi, K, [bins_x+1, bins_y+1])


            x0 = xd[replan_rate+1]
            append!(executed_traj, [x0])
            
            xd = nothing
            ud = nothing
            tm = nothing

            tm = TrajectoryManager(x0, xf, h, Tb, ConstantInitializer([0.0,0.0]));
            # tm = TrajectoryManager(x0, xf, h, Tb, CornerInitializer());
            tm.barrier_cost=1000.0
            tm.sample_cost = sample_cost
            xd, ud = pto_trajectory(em, tm, verbose=false, logging=true);
    #         xd, ud = smc_trajectory(em, tm, verbose=false);
            
            i = 1
        else
            i += 1
        end
        t += 1
    end

    xd_storage[1:length(xd), convert(Int, floor(t/replan_rate))] = xd
    phi_storage[:,:, convert(Int, floor(t/replan_rate))] = em.phi

    # deconstruct one last time to show the final result 
    ck = decompose(em, xd)
    phi_k = reconstruct(em, ck)
    em.phi = em.phi .- phi_k
    xd_storage[1, 1+convert(Int, floor(t/replan_rate))] = xd[end]
    phi_storage[:,:, 1+convert(Int, floor(t/replan_rate))] = em.phi
    
    append!(executed_traj, xd)

    anim = @animate for i ∈ 1:size(phi_storage)[3]
        @show i
        xd = xd_storage[1:(N-(i-1)*replan_rate),i]
        em.phi = phi_storage[:,:,i]
        plot_erg(em, xd)
    end
    Plots.gif(anim, path_name * "/temp3.gif", fps = 20)


    # # plot ever-increasing pieces of the trajectory
    # for i ∈ 1:size(phi_storage)[3] # remove this -1
    #     xd = xd_storage[1:(N-(i-1)*replan_rate),i]
    #     em.phi = phi_storage[:,:,i]
    #     plot_erg(em, xd)
    #     push!(frames, gcf())
    #     close()
    # end

    # final result
    em.phi = phi_storage[:,:,end]

    JLD2.save(path_name * "/executed_traj.jld", "executed_traj", executed_traj)

end

d = Domain(domain_min, domain_max, domain_bins)
run_ergodic_mpc(K_fc, ErgodicManagerR2(d, phi, K_fc, [bins_x+1, bins_y+1]), replan_rate, N, h, sample_cost, phi)
