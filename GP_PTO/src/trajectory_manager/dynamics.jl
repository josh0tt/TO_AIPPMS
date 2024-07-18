######################################################################
# dynamics.jl
# Each trajectory manager will have a type of dynamics
######################################################################
using LinearAlgebra

export Dynamics, LinearDynamics, DubinsDynamics, linearize
export IntegrationScheme, ForwardEuler, SymplecticEuler
export forward_euler
export GroupDynamics
export vvf2vvvf

@everywhere mutable struct LinearDynamics <: Dynamics
    n::Int
    m::Int
    A::Matrix{Float64}
    B::Matrix{Float64}

    function LinearDynamics(A, B)
        n,m = size(B)
        return new(n,m,deepcopy(A),deepcopy(B))
    end
end

@everywhere mutable struct GroupDynamics <: Dynamics
	n::Int
	m::Int
	num_agents::Int
	array

	function GroupDynamics(array)
		num_agents = length(array)
		n = m = 0
		for j = 1:num_agents
			n += array[j].n
			m += array[j].m
		end
		return new(n, m, num_agents, array)
	end
end

@everywhere mutable struct DubinsDynamics <: Dynamics
	n::Int
	m::Int

	v0::Float64		# constant speed
	r::Float64		# minimum turn radius

	# Constructors
	DubinsDynamics() = DubinsDynamics(1.0, 1.0)
	DubinsDynamics(v0::Real, r::Real) = new(3, 1, v0, r)
end

# Not really types,
export SingleIntegrator, DoubleIntegrator
@everywhere function SingleIntegrator(n::Int, h::Float64)
	A = diagm(ones(n,))
	B = h*diagm(ones(n,))
	return LinearDynamics(A, B)
end
@everywhere function DoubleIntegrator(n::Int, h::Float64)
	A = diagm(ones(2*n,))
	for i = 1:n
		A[i, n+i] = h
	end

	B = zeros(2*n, n)
	for i = 1:n
		B[n+i, i] = h
	end

	return LinearDynamics(A, B)
end


# """
# `dynamics!(tm::TrajectoryManager, d::Dynamics)`

# Not only sets `tm.dynamics = d`, but also sets reward matrices `tm.Qn`, `tm.R`, and `tm.Rn` to default matrices of the correct sizes. These defaults are:

# `tm.Qn = eye(d.n)`

# `tm.R = 0.01 * eye(d.m)`

# `tm.Rn = eye(d.m)`

# """
@everywhere function dynamics!(tm::TrajectoryManager, d::Dynamics)
	tm.dynamics = d
	tm.Qn = diagm(ones(d.n,))
	tm.R = 0.01 * diagm(ones(d.m,))
	tm.Rn = diagm(ones(d.m,))
end


######################################################################
# linearization
######################################################################

# linearizes about a trajectory
@everywhere function linearize(d::Dynamics, x::VVF, u::VVF, h::Float64)
	N = length(u)
	A = VMF()
	B = VMF()
	for n = 1:N
		An, Bn = linearize(d, x[n], u[n], h)
		push!(A, An)
		push!(B, Bn)
	end
	return A,B
end


@everywhere function linearize(ld::LinearDynamics, x::VF, u::VF, h::Float64)
	return ld.A, ld.B
end

@everywhere function linearize(ld::DubinsDynamics, x::VF, u::VF, h::Float64)
	A = diagm(ones(3,))
	A[1,3] = -h * sin(x[3]) * ld.v0
	A[2,3] = h * cos(x[3]) * ld.v0

	B = zeros(3,1)
	B[3] = h/ld.r

	return A, B
end

@everywhere function gsplit(gd::GroupDynamics, x::VF)
	xarr = VVF()
	xind = 1
	for j = 1:gd.num_agents
		d = gd.array[j]
		push!(xarr, x[xind + d.n - 1])
		xind += d.n
	end
	return xarr
end

@everywhere function gsplit(gd::GroupDynamics, x::VF, u::VF)
	xarr = VVF()
	uarr = VVF()
	xind = uind = 1
	for j = 1:gd.num_agents
		d = gd.array[j]
		push!(xarr, x[xind:(xind + d.n - 1)])
		push!(uarr, u[uind:(uind + d.m - 1)])
		xind += d.n
		uind += d.m
	end
	return xarr, uarr
end

@everywhere function linearize(gd::GroupDynamics, x::VF, u::VF, h::Float64)

	xarr, uarr = gsplit(gd, x, u)

	Aarr = VMF()
	Barr = VMF()
	for j = 1:gd.num_agents
		A, B = linearize(gd.array[j], xarr[j], uarr[j], h)
		push!(Aarr, A)
		push!(Barr, B)
	end

	dims = 2*ones(Int, gd.num_agents)
	dims[1] = 1

	return cat(dims, Aarr...), cat(dims, Barr...)
end


######################################################################
# integration
######################################################################
@everywhere mutable struct ForwardEuler <: IntegrationScheme end
@everywhere mutable struct SymplecticEuler <: IntegrationScheme end

@everywhere function integrate(tm::TrajectoryManager, x::VF, u::VF)
	integrate(tm.int_scheme, tm.dynamics, x, u, tm.h)
end

@everywhere function integrate(::ForwardEuler, d::Dynamics, x::VF, u::VF, h::Float64)
	forward_euler(d, x, u, h)
end

@everywhere function integrate(::SymplecticEuler, d::Dynamics, x::VF, u::VF, h::Float64)
	symplectic_euler(d, x, u, h)
end

@everywhere function integrate(tm::TrajectoryManager, ud::VVF)
	xd = Array{Vector{Float64}}(undef, tm.N+1)#Array(Vector{Float64}, tm.N+1)

	xd[1] = deepcopy(tm.x0)
	for i = 1:tm.N
		xd[i+1] = integrate(tm.int_scheme, tm.dynamics, xd[i], ud[i], tm.h)
	end

	return xd
end


######################################################################
# forward_euler
######################################################################
@everywhere function forward_euler(tm::TrajectoryManager, x::VF, u::VF)
	forward_euler(tm.dynamics, x, u, tm.h)
end

@everywhere function forward_euler(ld::LinearDynamics, x::VF, u::VF, h::Float64)
	umax = ud_max
	if sqrt(u[1]^2 + u[2]^2) > umax
		# rescale to have magnitude equal to umax
		u = (u./sqrt(u[1]^2 + u[2]^2))*umax
	end

	return ld.A*x + ld.B*u
end
@everywhere function forward_euler(gd::GroupDynamics, x::VF, u::VF, h::Float64)
	n_ind = m_ind = 1
	xp = VVF()
	xarr, uarr = gsplit(gd, x, u)
	for j = 1:gd.num_agents
		push!(xp, forward_euler(gd.array[j], xarr[j], uarr[j], h))
	end
	return vcat(xp...)
end

@everywhere function forward_euler(dd::DubinsDynamics, x::VF, u::VF, h::Float64)
	xp = deepcopy(x)
	xp[1] += cos(x[3]) * dd.v0 * h
	xp[2] += sin(x[3]) * dd.v0 * h
	u_val = u[1]
	if u_val > 1.
		u_val = 1.
	end
	if u_val < -1.
		u_val = -1.
	end
	xp[3] += u_val / dd.r * h
	return xp
end


######################################################################
# symplectic_euler
######################################################################
@everywhere function symplectic_euler(d::Dynamics, x::VF, u::VF, h::Float64)
	#A,B = linearize(d, x, u, h)
	A = zeros(4,4)
	A[1,3] = 1
	A[2,4] = 1
	A[3,1] = -1
	A[4,2] = -1
	B = zeros(4,2)
	B[3,1] = 1
	B[4,2] = 1

	#n2 = round(Int, d.n/2)
	n2 = 2

	A11 = A[1:n2, 1:n2]
	A12 = A[1:n2, n2+1:d.n]
	A21 = A[n2+1:d.n, 1:n2]
	A22 = A[n2+1:d.n, n2+1:d.n]

	B1 = B[1:n2,:]
	B2 = B[n2+1:d.n,:]

	# block matrices
	Ad22 = inv(diagm(ones(n2,))-h*A22)
	Ad11 = diagm(ones(n2,)) + h*A11 + h*h*A12*Ad22*A21
	Ad12 = h*A12*Ad22
	Ad21 = h*Ad22*A21

	Bd1 = h*h*A12*Ad22*B2 + h*B1
	Bd2 = h*Ad22*B2

	Ase = [Ad11 Ad12; Ad21 Ad22]
	Bse = [Bd1; Bd2]

	xp = Ase*x + Bse*u
	return xp
end


# split one vector per state up
@everywhere function vvf2vvvf(xd::VVF, ud::VVF, vtm::Vector{TrajectoryManager})
	N = length(xd) - 1
	num_agents = length(vtm)

	x_start = u_start = 1
	xds = VVVF(num_agents)
	uds = VVVF(num_agents)
	for j = 1:num_agents
		x_end = x_start + vtm[j].dynamics.n - 1
		u_end = u_start + vtm[j].dynamics.m - 1
		xds[j] = VVF()
		uds[j] = VVF()
		for n = 1:N
			push!(xds[j], xd[n][x_start:x_end])
			push!(uds[j], ud[n][u_start:u_end])
		end
		push!(xds[j], xd[N+1][x_start:x_end])
		x_start = x_end + 1
		u_start = u_end + 1
	end
	return xds, uds
end

@everywhere function vvf2vvvf(xd::VVF, vtm::Vector{TrajectoryManager})
	N = length(xd) - 1
	num_agents = length(vtm)

	x_start = 1
	xds = VVVF(num_agents)
	for j = 1:num_agents
		x_end = x_start + vtm[j].dynamics.n - 1
		xds[j] = VVF()
		for n = 1:N+1
			push!(xds[j], xd[n][x_start:x_end])
		end
		x_start = x_end + 1
	end
	return xds
end

@everywhere function vvf2vvvf(xd::VVF, gd::GroupDynamics)
	N = length(xd) - 1

	x_start = 1
	xds = VVVF(gd.num_agents)
	for j = 1:gd.num_agents
		x_end = x_start + gd.array[j].n - 1
		xds[j] = VVF()
		for n = 1:(N+1)
			push!(xds[j], xd[n][x_start:x_end])
		end
		x_start = x_end + 1
	end
	return xds
end
