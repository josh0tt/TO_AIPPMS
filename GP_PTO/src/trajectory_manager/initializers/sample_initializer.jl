######################################################################
# sample_initializer.jl
######################################################################

# """
# `si = SampleInitializer()`

# Samples points from a distribution.
# """
@everywhere mutable struct SampleInitializer <: Initializer end


@everywhere function initialize(si::SampleInitializer, gpm::GaussianProcessManager, tm::TrajectoryManager)
	xd = Array(Vector{Float64}, tm.N+1)
	points = Array(Vector{Float64}, tm.N)
	xd[1] = deepcopy(tm.x0)
	ud = Array(Vector{Float64}, tm.N)
	bin_size = (gpm.bins, gpm.bins)

	# first sample N points from e.phi
	weights = Weights(vec(gpm.phi))
	for n = 1:tm.N
		xi, yi = ind2sub(bin_size, sample(weights))
		points[n] = [gpm.cell_size*(xi-.5), gpm.cell_size*(yi-.5)]
	end

	# find a short path heuristically
	#tsp_rand!(xd, points)
	tsp_nn!(xd, points)

	# compute controls
	ud = compute_controls(xd, tm.h)

	return xd, ud
end
