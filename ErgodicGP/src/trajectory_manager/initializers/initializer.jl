######################################################################
# initializer.jl
# different ways to initalize a trajectory
######################################################################
export ConstantInitializer
export RandomInitializer
export SampleInitializer
export CornerInitializer
export PrevTrajInitializer

include("constant_initializer.jl")
include("random_initializer.jl")
include("sample_initializer.jl")
include("corner_initializer.jl")
include("previoustraj_initializer.jl")


@everywhere function initialize(egpm::ErgodicGPManager, tm::TrajectoryManager)
	initialize(tm.initializer, egpm, tm)
end
@everywhere function initialize(egpm::ErgodicGPManager, vtm::Vector{TrajectoryManager})
	num_agents = length(vtm)
	xd0s = VVVF()
	ud0s = VVVF()
	for j = 1:num_agents
		xd0, ud0 = initialize(egpm, vtm[j])
		push!(xd0s, xd0)
		push!(ud0s, ud0)
	end
	return vvvf2vvf(xd0s, ud0s)
end



@everywhere mutable struct CornerConstantInitializer <: Initializer
	magnitude::Float64
end

# """
# `gi = GreedyInitializer()`

# Greedily goes to spot with maximum phi.
# Assumes phi decreases at a constant rate.
# """
@everywhere mutable struct GreedyInitializer <: Initializer end


# """
# `poi = PointInitializer(xd::Vector{Float64})`

# Moves to point `xd`.
# """
@everywhere mutable struct PointInitializer <: Initializer
	xd::Vector{Float64}
end

# """
# `di = DirectionInitializer(xd::Vector{Float64}, mag::Float64)`

# Moves in the direction of point `xd` with magnitude `mag`.
# """
@everywhere mutable struct DirectionInitializer <: Initializer
	xd::Vector{Float64}
	mag::Float64
end


@everywhere function initialize(cci::CornerConstantInitializer, egpm::ErgodicGPManager, tm::TrajectoryManager)
	xd = Array{Vector{Float64}}(undef, tm.N+1)#Array(Vector{Float64}, tm.N+1)
	ud = Array{Vector{Float64}}(undef, tm.N)#Array(Vector{Float64}, tm.N)
	xd[1] = deepcopy(tm.x0)
	ud[1] = deepcopy(ci.action)
	for i = 1:(tm.N-1)
		xd[i+1] = tm.A*xd[i] + tm.B*ud[i]
		ud[i+1] = deepcopy(ci.action)
	end
	xd[tm.N+1] = tm.A*xd[tm.N] + tm.B*ud[tm.N]

	return xd, ud
end


export greedy_trajectory
@everywhere function greedy_trajectory(egpm::ErgodicGPManager, tm::TrajectoryManager)
	return initialize(GreedyInitializer(), egpm, tm)
end
@everywhere function initialize(gi::GreedyInitializer, egpm::ErgodicGPManager, tm::TrajectoryManager)
	d_rate = sum(gpm.phi)/tm.N
	num_cells = egpm.bins*gpm.bins
	total_info = 0.0
	xd = Array{Vector{Float64}}(undef, tm.N+1)#Array(Vector{Float64}, tm.N+1)
	xd[1] = deepcopy(tm.x0)
	temp_phi = deepcopy(egpm.phi)
	size_tuple = (egpm.bins, egpm.bins)
	for n = 1:tm.N
		bi = argmax(temp_phi)
		xi, yi = Tuple(bi)#CartesianIndex(size_tuple)[bi]#ind2sub(size_tuple, bi)
		xd[n+1] = [(xi)*0.01, (yi)*0.01]
		temp_phi[bi] -= min(temp_phi[bi], d_rate)
	end
	ud = compute_controls(xd, tm.h)
	return xd,ud
end


# moves to a point with a constant control input
@everywhere function initialize(initializer::PointInitializer, egpm::ErgodicGPManager, tm::TrajectoryManager)

	# compute the direction and action we most go towards
	dx = initializer.xd[1] - tm.x0[1]
	dy = initializer.xd[2] - tm.x0[2]


	# if we are double integrator, only apply the input once
	ud = Array{Vector{Float64}}(undef, tm.N)#Array(Vector{Float64}, tm.N)
	if tm.dynamics.n > 2
		den = tm.h * tm.h * (tm.N-1)
		ud[1] = [dx/den, dy/den]
		for i = 1:(tm.N-1)
			ud[i+1] = zeros(2)
		end
	else  # single integrator
		den = tm.h * tm.N
		u = [dx / den, dy / den]
		for i = 1:tm.N
			ud[i] = deepcopy(u)
		end
	end

	xd = integrate(tm, ud)

	return xd, ud
end

# moves to a point with a constant control input
@everywhere function initialize(initializer::DirectionInitializer, egpm::ErgodicGPManager, tm::TrajectoryManager)
	xd = Array{Vector{Float64}}(undef, tm.N+1)#Array(Vector{Float64}, tm.N+1)
	ud = Array{Vector{Float64}}(undef, tm.N)#Array(Vector{Float64}, tm.N)

	# compute the direction and action we most go towards
	dx = initializer.xd[1] - tm.x0[1]
	dy = initializer.xd[2] - tm.x0[2]
	u = initializer.mag * [dx, dy] / sqrt(dx*dx + dy*dy)

	xd[1] = deepcopy(tm.x0)
	ud[1] = deepcopy(u)
	for i = 1:(tm.N-1)
		xd[i+1] = tm.A*xd[i] + tm.B*ud[i]
		ud[i+1] = deepcopy(u)
	end
	xd[tm.N+1] = tm.A*xd[tm.N] + tm.B*ud[tm.N]

	return xd, ud
end
