# function belief_reward(pomdp::RoverPOMDP, b::RoverBelief, a::Symbol, bp::RoverBelief)
#     r = 0.0
#
#     if isterminal(pomdp, b)
#         return 0
#     else
#         if b.location_belief.X == []
#             μ_init, ν_init = query_no_data(b.location_belief)
#         else
#             μ_init, ν_init, S_init = query(b.location_belief)
#         end
#
#         if bp.location_belief.X == []
#             μ_post, ν_post, = query_no_data(bp.location_belief)
#         else
#             μ_post, ν_post, S_post = query(bp.location_belief)
#         end
#
#         variance_reduction = (sum(ν_init) - sum(ν_post))
#         # TODO: pass lamda in as a parameter to pomdp struct
#         r += 1*variance_reduction
#
#         if a == :drill
#             if round(μ_init[b.pos], digits=1) in b.drill_samples
#                 r += pomdp.repeat_sample_penalty
#             else
#                 r +=  pomdp.new_sample_reward
#             end
#         else
#             r +=  0
#         end
#     end
#
#     return r
# end

function belief_reward(pomdp::RoverPOMDP, b::RoverBelief, a::Symbol, bp::RoverBelief)
    r = 0.0

    if isterminal(pomdp, b)
        return 0
    else
        if b.location_belief.X == []
            μ_init, ν_init, S_init, EI_init = query_no_data(b.location_belief)
        else
            μ_init, ν_init, S_init, EI_init = query(b.location_belief)
        end

        if bp.location_belief.X == []
            μ_post, ν_post, S_post, EI_post = query_no_data(bp.location_belief)
        else
            μ_post, ν_post, S_post, EI_post = query(bp.location_belief)
        end


        # NOTE: commented this out because this only applies to Rover Exploration Problem
        # for even comparison with other TO_AIPPMS methods we are only using variance reduction 
        # if a == :drill
        #     mu = μ_init[b.pos]
        #     σ = sqrt(ν_init[b.pos])
        #     z_score = 1.4 #1.96 # 95% confidence is too conservative for drilling
        #     if any([mu-z_score*σ <= d <= mu+z_score*σ for d in b.drill_samples])
        #     # if round(μ_init[b.pos], digits=1) in b.drill_samples
        #         r += pomdp.repeat_sample_penalty
        #     # this is for when we have already drilled in an area so the confidence region shrinks
        #     elseif round(μ_init[b.pos], digits=1) in b.drill_samples
        #         r += pomdp.repeat_sample_penalty

        #     #XXX: hack since julia does not return true for -0.0 in set with 0.0
        #     elseif round(μ_init[b.pos], digits=1)==0.0 && 0.0 in b.drill_samples
        #         r += pomdp.repeat_sample_penalty

        #     else
        #         r +=  pomdp.new_sample_reward
        #     end
        # else
        #     # entropy reduction
        #     variance_reduction = (sum(ν_init) - sum(ν_post))
        #     # mutual information
        #     # variance_reduction = 0.5*logdet(S_init) - 0.5*logdet(S_post)
        #     # TODO: pass lamda in as a parameter to pomdp struct
        #     r += 0.1*variance_reduction
        # end

        # added this, remove it if you switch back to Rover Exploration Problem
        variance_reduction = (sum(ν_init) - sum(ν_post))
        expected_improvement_reduction = (sum(EI_init) - sum(EI_post))
        if objective == "expected_improvement"
            r += expected_improvement_reduction
        else
            r += variance_reduction
        end
        
    end

    return r
end

function POMDPs.reward(pomdp::RoverPOMDP, s::RoverState, a::Symbol, sp::RoverState)
    # if isterminal(pomdp, s)
    #     return 0
    # end

    if a == :drill
        if s.location_states[s.pos] in s.drill_samples
            return pomdp.repeat_sample_penalty
        else
            return pomdp.new_sample_reward
        end
    else
        return 0
    end
    # if isgoal(pomdp, s)
    #     ν_init = diag(GP.KXqXq) .+ eps() # eps prevents numerical issues
    #     ν_posterior = query(s.gp)[2]
    #     return (sum(ν_init) - sum(ν_posterior))
    # else
    #     return 0
    # end

    #return -s.tot_vQ
end

# function POMDPs.reward(mdp::SSTGridWorld, s::SSTState, a::Symbol, sp::SSTState)
#     x = s.pos[1]
#     y = s.pos[2]
#
#     if isgoal(mdp, s)
#         r = 0
#     elseif a == :drill
#         r = -mdp.depth_map[y,x]
#     else
#         r = -1
#     end
#     return r
# end

# function POMDPs.reward(pomdp::SSTPOMDP, s::SSTDistribution, action::Symbol, sp::SSTDistribution)
#     if isterminal(pomdp, s)
#         return 0
#     end
#     if isgoal(pomdp, s)
#         ν_init = diag(GP.KXqXq) .+ eps() # eps prevents numerical issues
#         ν_posterior = query(s.gp)[2]
#         return (sum(ν_init) - sum(ν_posterior))
#     else
#         return -0.1
#     end
#
#     #return -s.tot_vQ
# end

# function POMDPs.reward(pomdp::SSTPOMDP, s::SSTDistribution, action::Symbol, sp::SSTDistribution)
#     if isgoal(pomdp, s)
#         r = 0
#     elseif action == :drill
#         if s.gp.X == []
#             ν_prior = query_no_data(s.gp)[2]
#         else
#             ν_prior = query(s.gp)[2]
#         end
#         ν_posterior = query(sp.gp)[2]
#         r = (sum(ν_prior) - sum(ν_posterior))
#     else
#         r = -0.1
#     end
# end

# function POMDPs.reward(pomdp::SSTPOMDP, s::SSTDistribution)
#     return -s.tot_vQ
# end
#
# function POMDPs.reward(pomdp::SSTPOMDP, s::SSTDistribution,  action::Symbol)
#     return -s.tot_vQ
# end

# function POMDPs.reward(pomdp::SSTPOMDP, s::SSTDistribution, action::Symbol, sp::SSTDistribution)
#     # if s.gp.X == []
#     #     ν_prior = query_no_data(s.gp)[2]
#     # else
#     #     ν_prior = query(s.gp)[2]
#     # end
#     #
#     # if sp.gp.X == []
#     #     ν_posterior = query_no_data(sp.gp)[2]
#     # else
#     #     ν_posterior = query(sp.gp)[2]
#     # end
#
#     # return (sum(ν_prior) - sum(ν_posterior))
#     return (s.tot_vQ - sp.tot_vQ)
#
# end

# # THIS ACCEPTs OBSERVATION AS WELL SINCE belief_old AND belief_new WILL BE THE SAME BECAUSE ONLY THE TRANSITION WILL HAVE BEEN CALLED
# # FOR THE DIFFERENCE AMONGST SP AND S
# function POMDPs.reward(pomdp::SSTPOMDP, belief_old::SSTDistribution, action::Symbol, belief_new::SSTDistribution, observation::Tuple{Vector{Float64}, Float64, Symbol})
#     x_grid = belief_old.gp.data.x
#
#     x = GPPPInput(:f3, ColVecs(reshape(observation[1],(2,1))))
#     y = observation[2]
#     σ²_n = action == :drill ? pomdp.σ²_drill : pomdp.σ²_spec
#     f_posterior = belief_old.gp
#     f_posterior_x = f_posterior(x, σ²_n)
#     f_posterior = posterior(f_posterior_x, [y])
#
#     new_posterior = f_posterior
#
#     return (sum(Stheno.std.(marginals(belief_old.gp(x_grid, 1e-9)))) - sum(Stheno.std.(marginals(new_posterior(x_grid, 1e-9)))))
# end
#
# function POMDPs.reward(pomdp::SSTPOMDP, belief::SSTDistribution, action::Symbol)
#     observation = POMDPs.observation(pomdp, action, belief)
#     x_grid = belief.gp.data.x
#
#     x = GPPPInput(:f3, ColVecs(reshape(observation[1],(2,1))))
#     y = observation[2]
#     σ²_n = action == :drill ? pomdp.σ²_drill : pomdp.σ²_spec
#     f_posterior = belief.gp
#     f_posterior_x = f_posterior(x, σ²_n)
#     f_posterior = posterior(f_posterior_x, [y])
#
#     new_posterior = f_posterior
#
#     return (sum(Stheno.std.(marginals(belief.gp(x_grid, 1e-9)))) - sum(Stheno.std.(marginals(new_posterior(x_grid, 1e-9)))))
# end



# function POMDPs.reward(mdp::SSTGridWorld, s::SSTState, a::Symbol, sp::SSTState)
#     x = s.pos[1]#s.pos[1] # x = j
#     y = s.pos[2]#mdp.size[1] - s.pos[2] + 1
#
#     r = 0
#     #n_drills = sum([s.drill[i]!=[] for i=1:length(s.drill)]) # number of drills so far
#     n_drills = sum([s.drill[i]!=GWPos(0,0) for i=1:length(s.drill)]) # number of drills so far
#
#     #TODO: check reward units
#
#     if isgoal(mdp, s)
#         if n_drills == 3
#             if a != :drill
#                 r += 1
#                 #TODO: switch back to checking drill spacing
#                 r += check_drill_spacing(mdp, s)
#             else
#                 r += -1
#             end
#             return r
#         elseif ((a == :drill) & (n_drills == 2))
#             r += 1
#             #TODO: switch back to checking drill spacing
#             r += check_drill_spacing(mdp, s)
#             # dont return yet, let it drill reward in the if a == :drill
#         else
#             r += -1
#             return r
#         end
#     end
#
#     if isterminal(mdp, s)
#         return 0
#     end
#
#     # R drill
#     if a == :drill
#         if n_drills == 3
#             r += -1
#         else
#             spec_meas = rand(Normal(mdp.depth_map[y,x],mdp.uncertainty_map[y,x]))
#             p_spec_meas = pdf(Normal(mdp.depth_map[y,x],mdp.uncertainty_map[y,x]), spec_meas)
#
#             #TODO: switch back to using spectrometer measurement
#             r += 100*(mdp.uncertainty_map[y,x]*(1-mdp.depth_map[y,x])) #+ maximum([0, (1 - spec_meas)*p_spec_meas]))#10*(mdp.uncertainty_map[x,y]*(1-mdp.depth_map[x,y]) + maximum([0, (1 - spec_meas)*p_spec_meas]))
#         end
#     end
#
#     if ((a!= :drill) ||  (a!= :wait)) & (s == sp) # we ran into the wall
#         r += -1
#     end
#
#
#     return r
# end
#
# function check_drill_spacing(mdp::SSTGridWorld, s::SSTState)
#     #n_drills = sum([s.drill[i]!=[] for i=1:length(s.drill)]) # number of drills so far
#     n_drills = sum([s.drill[i]!=GWPos(0,0) for i=1:length(s.drill)]) # number of drills so far
#     r = 0
#     for i = 1:n_drills
#         for j = 1:n_drills
#             if drill_distance(s.drill[i], s.drill[j]) < 1
#                 r += -1
#             end
#         end
#     end
#     return r
# end
#
# function drill_distance(drill1, drill2)
#     x1 = drill1[1]
#     y1 = drill1[2]
#     x2 = drill2[1]
#     y2 = drill2[2]
#
#     return sqrt((x2-x1)^2 + (y2-y1)^2)
# end
#
# function build_rewards(size::Tuple{Int, Int}, zone_map_rewards::Matrix{Float64}, uncertainty_map::Array{Float64, 2}, traversability_map::Array{Float64, 2}, terminal_pos::Tuple{Int, Int})
#     rewards = Dict()
#     nx = size[1]
#     ny = size[2]
#     for x in 1:nx, y in 1:ny
#         i = abs(y - size[1]) + 1
#         j = x
#
#         #rewards[GWPos(x, y)] = -1000 .* log(uncertainty_map[x, y])*uncertainty_map[x, y]
#         # FOR INFORMATION DENSITY
#         rewards[GWPos(x, y)] = uncertainty_map[x, y]
#         rewards[GWPos(x, y)] += zone_map_rewards[x,y]
#         # if use_thompson_sampling
#         #     rewards[GWPos(x, y)] = rand(Normal(zone_map_rewards[x,y], uncertainty_map[x, y]), 1)[1] #uncertainty_map[x, y]
#         # else
#         #     rewards[GWPos(x, y)] = uncertainty_map[x, y]
#         #     rewards[GWPos(x, y)] += zone_map_rewards[x,y]
#         # end
#
#         if x == terminal_pos[1] && y == terminal_pos[2]
#             rewards[GWPos(x, y)] += 1*nx*nx*nx*nx
#         end
#         # if zone_map[i,j] == RGB(1.0N0f8, 1.0N0f8, 0.0N0f8) #yellow DEEP
#         #     rewards[GWPos(x, y)] += 1
#         # end
#         # if zone_map[i,j] == RGB(00.0N0f8, 0.502N0f8, 0.0N0f8) #green SHALLOW
#         #     rewards[GWPos(x, y)] += 2
#         # end
#         # if zone_map[i,j] == RGB(0.545N0f8, 0.0N0f8, 0.0N0f8) #red SURFACE
#         #     rewards[GWPos(x, y)] += 3
#         # end
#     end
#     return rewards
# end


# Rewards

# POMDPs.reward(mdp::SSTGridWorld, s::AbstractVector{Int}) = get(mdp.rewards, s, 0.0)
# POMDPs.reward(mdp::SSTGridWorld, s::AbstractVector{Int}, a::Symbol) = reward(mdp, s)
#POMDPs.reward(mdp::SSTGridWorld, s::SSTState, a::Symbol, sp::SSTState) = reward(mdp, s, a)
