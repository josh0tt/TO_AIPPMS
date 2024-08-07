######################################################################
# armijo_descender.jl
######################################################################
# """
# type `ArmijoLineSearch <: Descender`

# `ArmijoLineSearch(initial_step, c)`

# Defaults are:
# * `initial_step` = 10
# * `c` = 0.5
# * `max_iters` = 50
# """

@everywhere mutable struct ArmijoLineSearch <: Descender
    initial_step::Float64
    c::Float64		# just a constant between 0 and 1
    max_iters::Float64

    function ArmijoLineSearch(initial_step::Real, c::Real, mi::Real)
        return new(float(initial_step), float(c), float(mi))
    end
    function ArmijoLineSearch(initial_step::Real, c::Real)
        return new(float(initial_step), float(c), 50.)
    end

    ArmijoLineSearch() = ArmijoLineSearch(100, 1e-6, 50.) #0.5
end

@everywhere function get_step_size(als::ArmijoLineSearch, gpm::GaussianProcessManager, tm::TrajectoryManager, xd::VVF, ud::VVF, sample_actions::VF, zd::VVF, vd::VVF, ad::MF, bd::MF, sad::MF, K::Vector{MF}, i::Int)
    tau = 0.5
    step_size = als.initial_step

    # compute m = p' * grad f(x)
    m = directional_derivative(ad, bd, zd, vd)

    f_x = total_score(gpm, tm, xd, ud, sample_actions)[6] # total_score() returns a tuple of gps,xfs,cs,bs,ts

    xdn, udn = project(gpm, tm, K, xd, ud, zd, vd, step_size)

    ts_requested = total_score(gpm, tm, xdn, udn, sample_actions)[6]
    armijo_index = 0

    while (ts_requested > f_x + step_size*als.c*m) && (armijo_index < als.max_iters)
        step_size *= tau
        xdn, udn = project(gpm, tm, K, xd, ud, zd, vd, step_size)
        armijo_index += 1
        ts_requested = total_score(gpm, tm, xdn, udn, sample_actions)[6]
    end
    return step_size
end


@everywhere function get_step_size2(als::ArmijoLineSearch, gpm::GaussianProcessManager, tm::TrajectoryManager, xd::VVF, ud::VVF, sample_actions::VF, zd::VVF, vd::VVF, ad::MF, bd::MF, sad::VF, K::Vector{MF}, i::Int)
    tau = 0.5
    step_size = als.initial_step

    m = directional_derivative(ad, bd, zd, vd)

    f_x = total_score(gpm, tm, xd, ud)

    xdn, udn = project2(gpm, tm, K, xd, ud, zd, vd, step_size)
    ts_requested = total_score(gpm, tm, xdn, udn)
    armijo_index = 0
    while (total_score(gpm, tm, xdn, udn) > f_x + step_size*als.c*m) && (armijo_index < als.max_iters)
        ts_requested = total_score(gpm, tm, xdn, udn)
        step_size *= tau
        xdn, udn = project2(gpm, tm, K, xd, ud, zd, vd, step_size)
        armijo_index += 1
    end
    return step_size
end
