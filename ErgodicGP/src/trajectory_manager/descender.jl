######################################################################
# descender.jl
# provides descent engine
######################################################################
export Descender

export InverseStep
export InverseRootStep
export ArmijoLineSearch
export ConstantStep

@everywhere struct InverseStep <: Descender
    alpha::Float64
end

@everywhere function get_step_size(ir::InverseStep, egpm::ErgodicGPManager, tm::TrajectoryManager, xd::VVF, ud::VVF, zd::VVF, vd::VVF, ad::MF, bd::MF, K::Vector{MF}, i::Int)
    return ir.alpha / i
end


@everywhere struct InverseRootStep <: Descender
    alpha::Float64
end

#get_step_size(ir::InverseRootStep, i::Int) = ir.alpha / sqrt(i)
@everywhere function get_step_size(ir::InverseRootStep, egpm::ErgodicGPManager, tm::TrajectoryManager, xd::VVF, ud::VVF, zd::VVF, vd::VVF, ad::MF, bd::MF, K::Vector{MF}, i::Int)
    return ir.alpha / sqrt(i)
end


@everywhere struct ConstantStep <: Descender
    alpha::Float64
end

@everywhere function get_step_size(cs::ConstantStep, egpm::ErgodicGPManager, tm::TrajectoryManager, xd::VVF, ud::VVF, zd::VVF, vd::VVF, ad::MF, bd::MF, K::Vector{MF}, i::Int)
    return cs.alpha
end


include("armijo_line_search.jl")
