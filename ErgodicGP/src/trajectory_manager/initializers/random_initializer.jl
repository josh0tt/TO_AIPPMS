######################################################################
# random_initializer.jl
######################################################################

# """
# `ri = RandomInitializer()`

# Random selection of points.
# """
@everywhere mutable struct RandomInitializer <: Initializer end


@everywhere function initialize(ri::RandomInitializer, egpm::ErgodicGPManager, tm::TrajectoryManager)
	xd = [[tm.x0[1], tm.x0[2]] for i = 1:tm.N+1]
	points = Array(Vector{Float64}, tm.N)
	for i = 1:tm.N
		points[i] = [rand(), rand()]
	end

	tsp_nn!(xd, points)

	ud = compute_controls(xd, tm.h)

	return xd, ud
end
