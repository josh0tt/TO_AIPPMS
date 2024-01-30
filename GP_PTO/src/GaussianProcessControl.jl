# __precompile__()
#
# module GaussianProcessControl
using Distributed
import Base.normalize!
using SpecialFunctions
using Zygote

# export functions I've made
export GaussianProcessManager
export control_score, total_score
export control_effort
export TrajectoryManager, dynamics!, sample_trajectory
export assign_step, mean_step

export
    Initializer,
    initialize,
    CornerConstantInitializer,
    GreedyInitializer,
    PointInitializer,
    DirectionInitializer

export
    x_min, y_min, z_min,
    x_max, y_max, z_max,
    x_size, y_size, z_size,
    x_cells, y_cells, z_cells

# to make some things easier
@everywhere const T2F = NTuple{2, Float64}    # x, y
@everywhere const MF = Matrix{Float64}
@everywhere const VMF = Vector{MF}
@everywhere const VF = Vector{Float64}
@everywhere const VVF = Vector{VF}
@everywhere const VVVF = Vector{VVF}

# math-type stuff I might need
include("math.jl")
include("lq.jl")
include("lqr.jl")

include("gaussian_process_manager/domain.jl")
include("gaussian_process_manager/gaussian_process_manager.jl")
include("gaussian_process_manager/r2.jl")

# trajectory manager stuff
include("trajectory_manager/trajectory_manager.jl")
include("trajectory_manager/dynamics.jl")
include("trajectory_manager/initializers/initializer.jl")
include("trajectory_manager/descender.jl")

# trajectory generation
include("trajectory_generation/trajectory_generation.jl")

# GP
include("../../CustomGP.jl")

#end # module
