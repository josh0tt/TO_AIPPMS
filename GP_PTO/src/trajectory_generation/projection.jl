######################################################################
# projection.jl
#
# Handles projections and descents
######################################################################

# returns xdn and udn, the feasible projected trajectory
@everywhere function project(gpm::GaussianProcessManager, tm::TrajectoryManager, K::VMF, xd::VVF, ud::VVF, zd::VVF, vd::VVF, step_size::Float64)
	xdn = [xd[1]]

	udn = Array{Vector{Float64}}(undef, 0)

	# perform descent
	alpha = Vector{Vector{Float64}}(undef, length(xd))

	for n = 0:(length(xd)-1)#tm.N
		alpha[n+1] = xd[n+1] + step_size * zd[n+1]
	end

	# perform the projection
	for n = 1:(length(xd)-1)
		push!(udn, ud[n] + step_size*vd[n] + K[n]*(alpha[n] - xdn[n]))
		push!(xdn, integrate(tm, xdn[n], udn[n]) )
	end
	return xdn, udn
end

@everywhere function project(gpm::GaussianProcessManager, tm::TrajectoryManager, sad::MF, sample_actions::VF, step_size::Float64)
	sample_actions = sample_actions - 0.001*[sad[i] for i in 1:length(sad)] # minus because descending
	for i in 1:length(sample_actions)
		if sample_actions[i] > 1.0
			sample_actions[i] = 1.0
		elseif sample_actions[i] < 0.0
			sample_actions[i] = 0.0
		end
	end

	return sample_actions
end

# A projection for LTI systems
@everywhere function project2(gpm::GaussianProcessManager, tm::TrajectoryManager, K::VMF, xd::VVF, ud::VVF, zd::VVF, vd::VVF, step_size::Float64)
	xdn = [xd[1]]
    udn = Array{VF}(undef, 0)#Array{VF}(0)

	xdn = VVF(0)
	udn = VVF(0)

	# perform the projection
	# Shouldn't need to even integrate...
	for n = 1:tm.N
		push!(udn, ud[n] + step_size*vd[n])
		push!(xdn, xd[n] + step_size*zd[n])
	end
	push!(xdn, xd[tm.N+1] + step_size*zd[tm.N+1])
	return xdn, udn
end


# No projection and performed in place
@everywhere function descend!(xd::VVF, ud::VVF, zd::VVF, vd::VVF, step_size::Float64)
	N = length(ud)
	for ni = 1:N
		ud[ni] += step_size * vd[ni]
		xd[ni] += step_size * zd[ni]
	end
	xd[N+1] += step_size * zd[N+1]
end
