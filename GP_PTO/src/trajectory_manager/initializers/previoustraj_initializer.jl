######################################################################
# constant_initialier.jl
######################################################################
# """
# `ci = ConstantInitializer(action::Vector{Float64})`

# Just takes a constant action.
# """
@everywhere mutable struct PrevTrajInitializer <: Initializer
	xd::VVF
	ud::VVF
end


@everywhere function initialize(ci::PrevTrajInitializer,  gpm::GaussianProcessManager, tm::TrajectoryManager)
	return ci.xd, ci.ud
end
