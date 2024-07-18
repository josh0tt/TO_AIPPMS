######################################################################
# trajectory_generation.jl
######################################################################

# helper files
include("gradients.jl")
include("scoring.jl")
include("projection.jl")
include("printing.jl")

# actual methods
include("pto.jl")

@everywhere function check_convergence(gps::Float64, gps_crit::Float64, i::Int, max_iters::Int, dd::Float64, dd_crit::Float64, verbose::Bool, gps_count::Int)
    not_finished = true
    if gps < gps_crit
        not_finished = false
        if verbose
            println("reached GP criterion...")
        end
    end
    if i > max_iters
        not_finished = false
        if verbose
            println("max iterations reached...")
        end
    end
    if abs(dd) < dd_crit
        not_finished = false
        if verbose
            println("reached directional derivative criterion...")
        end
    end
    if gps_count > 50
        not_finished = false
        if verbose
            println("We've been stuck for 50 iterations...")
        end
    end
    return not_finished
end

# called if logging, not meant for general use
@everywhere function save(outfile::IOStream, xd::VVF)
    n = length(xd[1])
    for xi in xd
        for i = 1:(n-1)
            wi = xi[i]
            write(outfile,"$(xi[i]),")
        end
        write(outfile,"$(xi[n])\n")
    end
end

@everywhere function save(outfile::IOStream, sample_actions::VF)
    for a in sample_actions
        write(outfile,"$(a)\n")
    end
end

# called if logging, not meant for general use
@everywhere function save(outfile::IOStream, ts::Float64)
    write(outfile,"$(ts),")
end
