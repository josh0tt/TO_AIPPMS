######################################################################
# trajectory_manager.jl
#
# handles the trajectory manager
######################################################################

using StatsBase: Weights, sample
using Random

abstract type Dynamics end
abstract type Initializer end
abstract type Descender end
abstract type IntegrationScheme end


mutable struct TrajectoryManager

	# needed for all trajectories
	N::Int
	h::Float64
	x0::Vector{Float64}
	xf::Vector{Float64}

	# Cost functions
	q::Float64
	R::Matrix{Float64}
	Qn::Matrix{Float64}
	Qf::Matrix{Float64}
	qsa::Float64
	Rn::Matrix{Float64}
	barrier_cost::Float64
	sample_cost::Int64 # cost for adding another sample (i.e. sample cost of 3 means that 1 additional sample is equivalent to 3 steps on the trajectory)

	initializer::Initializer
	descender::Descender
	dynamics::Dynamics
	int_scheme::IntegrationScheme

	rng::MersenneTwister

	function TrajectoryManager(x0::Vector{Float64}, xf::Vector{Float64}, h::Real, N::Int, i::Initializer=RandomInitializer())
		tm = new()

		# needed for all trajectories
		tm.N = N
		tm.h = h
		tm.x0 = deepcopy(x0)
		tm.xf = deepcopy(xf)

		# needed for ergodic trajectories
		tm.Qn = [1 0; 0 1]
		tm.Qf = [1 0; 0 1]#[1000 0; 0 1000]
		tm.qsa = 1.0 #[1 0 0; 0 1 0; 0 0 1]
		tm.q = 1.0
		tm.R = 0.1*[1 0; 0 1]#0.01 * [1 0; 0 1]
		tm.Rn = [1 0; 0 1]
		tm.barrier_cost = 0.
		tm.initializer = i
		tm.descender = ArmijoLineSearch()

		tm.sample_cost = 10 #21 #3 # should be 150? 

		# dynamics stuff
		tm.dynamics = LinearDynamics([1 0; 0 1], tm.h*[1 0; 0 1])
		tm.int_scheme = ForwardEuler()

		tm.rng = MersenneTwister(1234)

		return tm
	end
end

#typealias VTM Vector{TrajectoryManager}
const VTM = Vector{TrajectoryManager}



# computes controls from a trajectory
export compute_controls
function compute_controls(xd::VVF, h::Float64)
	N = length(xd) - 1
	ud = Array{Vector{Float64}}(undef, N)#Array(Vector{Float64}, N)
	for n = 1:N
		ud[n] = (xd[n+1] - xd[n]) / h
	end

	return ud
end


# creates a sample_trajectory
function sample_trajectory(em::ErgodicManager, tm::TrajectoryManager)
	return initialize(SampleInitializer(), em, tm)
end
