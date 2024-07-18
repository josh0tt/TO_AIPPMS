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

# """
# `xd, ud = initialize(em::ErgodicManager, tm::TrajectoryManager)`

# Runs `tm`'s initializer to return a trajectory.
# """
@everywhere function initialize(gpm::GaussianProcessManager, tm::TrajectoryManager)
	initialize(tm.initializer, gpm, tm)
end
@everywhere function initialize(gpm::GaussianProcessManager, vtm::Vector{TrajectoryManager})
	num_agents = length(vtm)
	xd0s = VVVF()
	ud0s = VVVF()
	for j = 1:num_agents
		xd0, ud0 = initialize(gpm, vtm[j])
		push!(xd0s, xd0)
		push!(ud0s, ud0)
	end
	return vvvf2vvf(xd0s, ud0s)
end



# """
# `cci = CornerConstantInitializer(action::Vector{Float64})`

# Just takes a constant action.
# """
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





@everywhere function initialize(cci::CornerConstantInitializer, gpm::GaussianProcessManager, tm::TrajectoryManager)
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
@everywhere function greedy_trajectory(gpm::GaussianProcessManager, tm::TrajectoryManager)
	return initialize(GreedyInitializer(), gpm, tm)
end
@everywhere function initialize(gi::GreedyInitializer, gpm::GaussianProcessManager, tm::TrajectoryManager)
	d_rate = sum(gpm.phi)/tm.N
	num_cells = gpm.bins*gpm.bins
	total_info = 0.0
	xd = Array{Vector{Float64}}(undef, tm.N+1)#Array(Vector{Float64}, tm.N+1)
	xd[1] = deepcopy(tm.x0)
	temp_phi = deepcopy(gpm.phi)
	size_tuple = (gpm.bins, gpm.bins)
	for n = 1:tm.N
		bi = argmax(temp_phi)
		xi, yi = Tuple(bi)#CartesianIndex(size_tuple)[bi]#ind2sub(size_tuple, bi)
		xd[n+1] = [(xi)*0.01, (yi)*0.01]
		#xd[n+1] = [(xi)*em.domain.maxes[1]/em.bins[1], (yi)*em.domain.maxes[2]/em.bins[1]]#[(xi-0.5)*em.domain.cell_size, (yi-0.5)*em.domain.cell_size]
		#xd[n+1] = [xd[n][1]+ 0.5*(xi*em.domain.maxes[1]/em.bins[1] -xd[n][1]), xd[n][2]+ 0.5*(yi*em.domain.maxes[2]/em.bins[1] -xd[n][2])]
		temp_phi[bi] -= min(temp_phi[bi], d_rate)
	end
	ud = compute_controls(xd, tm.h)
	return xd,ud
end


# moves to a point with a constant control input
@everywhere function initialize(initializer::PointInitializer, gpm::GaussianProcessManager, tm::TrajectoryManager)

	# compute the direction and action we most go towards
	dx = initializer.xd[1] - tm.x0[1]
	dy = initializer.xd[2] - tm.x0[2]
	#x_step = (initializer.xd[1] - tm.x0[1]) / (tm.N * tm.h)
	#y_step = (initializer.xd[2] - tm.x0[2]) / (tm.N * tm.h)
	#u = [x_step, y_step]

	# if we are double integrator, only apply the input once
	@show tm
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
@everywhere function initialize(initializer::DirectionInitializer, gpm::GaussianProcessManager, tm::TrajectoryManager)
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
