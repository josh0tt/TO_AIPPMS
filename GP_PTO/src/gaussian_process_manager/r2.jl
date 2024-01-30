######################################################################
# ergodic_manager/r2.jl
#
# handles stuff needed for ergodicity
######################################################################


export GaussianProcessManagerR2


@everywhere mutable struct GaussianProcessManagerR2 <: GaussianProcessManager
	domain::Domain				# spatial domain
	GP::GaussianProcess
	σ_drill::Float64
	σ_spec::Float64

	function GaussianProcessManagerR2(d::Domain, GP::GaussianProcess, σ_drill::Float64, σ_spec::Float64)
		gpm = new()
		gpm.domain = deepcopy(d)
		gpm.GP = GP
		gpm.σ_drill = σ_drill
		gpm.σ_spec = σ_spec

		return gpm
	end
end
