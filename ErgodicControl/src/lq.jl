######################################################################
# lq.jl
#
# Solves LQ descent direction problems
######################################################################
export LQ, apply_LQ_gains 

function LQ(A::MF, B::MF, a::MF, b::MF, Q::MF, R::MF, N::Int)
	# Also needed for LQR
	P = Array(Matrix{Float64}, N+1)
	G = Array(Matrix{Float64}, N)
	K = Array(Matrix{Float64}, N)
	P[N+1] = Q

	# Solely for LQ
	r = Array(Vector{Float64}, N+1)
	#r[N+1] = zeros(size(B,1))
	r[N+1] = 0.5*a[:,N+1]
	println("RARARAR")
	C = Array(Vector{Float64}, N)

	# Sweep from the back
	for n = (N-1):-1:0
		G[n+1] = R + (B' * P[n+1+1] * B)
		K[n+1] = inv(G[n+1]) * B' * P[n+1+1] * A
		P[n+1] = Q + (A' * P[n+1+1] * A) - (K[n+1]' * G[n+1] * K[n+1])
		r[n+1] = (A'-K[n+1]'*B')r[n+1+1] + .5*a[:,n+1] - .5*K[n+1]'*b[:,n+1]
		C[n+1] = inv(G[n+1]) * (B'*r[n+1+1] + .5*b[:,n+1])
	end

	return K, C
end


# This appears to be the one currrently in use
function LQ(A::VMF, B::VMF, a::MF, b::MF, Q::MF, R::MF, N::Int)
	# Also needed for LQR
	#Array{Vector{Float64}}(undef, tm.N+1)

	P = Array{Matrix{Float64}}(undef, N+1)
	G = Array{Matrix{Float64}}(undef, N)
	K = Array{Matrix{Float64}}(undef, N)
	P[N+1] = Q

	# Solely for LQ
	r = Array{Vector{Float64}}(undef, N+1)
	r[N+1] = 0.5*a[:,N+1]
	C = Array{Vector{Float64}}(undef, N)
	
	for n = N:-1:1
		G[n] = R + (B[n]' * P[n+1] * B[n])
		Ginv = inv(G[n])
		K[n] = Ginv * B[n]' * P[n+1] * A[n]
		Kp = K[n]'
		P[n] = Q + (A[n]' * P[n+1] * A[n]) - (Kp * G[n] * K[n])
		r[n] = (A[n]'-Kp*B[n]')r[n+1] + .5*a[:,n] - .5*Kp*b[:,n]
		C[n] = Ginv * (B[n]'*r[n+1] + .5*b[:,n])
	end

	return K, C
end

# Experimental...
# used in linear pto
function LQ(K::VMF, Ginv::VMF, A::VMF, B::VMF, a::MF, b::MF, Q::MF, R::MF, N::Int)

	# Solely for LQ
    r = Array{Vector{Float64}}(undef, N+1)
	r[N+1] = 0.5*a[:,N+1]
    C = Array{Vector{Float64}}(undef, N)

	for n = N:-1:1
		Kp = K[n]'
		r[n] = (A[n]'-Kp*B[n]')r[n+1] + .5*a[:,n] - .5*Kp*b[:,n]
		C[n] = Ginv[n] * (B[n]'*r[n+1] + .5*b[:,n])
	end

	return C
end

# used in linear pto
function LQ2(A::VMF, B::VMF, Q::MF, R::MF, N::Int)
	# Also needed for LQR
    P = Array{Matrix{Float64}}(N+1)
    G = Array{Matrix{Float64}}(N)
    Ginv = Array{Matrix{Float64}}(N)
    K = Array{Matrix{Float64}}(N)
	P[N+1] = Q

	for n = N:-1:1
		G[n] = R + (B[n]' * P[n+1] * B[n])
		Ginv[n] = inv(G[n])
		K[n] = Ginv[n] * B[n]' * P[n+1] * A[n]
		Kp = K[n]'
		P[n] = Q + (A[n]' * P[n+1] * A[n]) - (Kp * G[n] * K[n])
	end

	return K, Ginv
end


function apply_LQ_gains(A::MF, B::MF, K::Vector{MF}, C::VVF)
	N = length(K)

	z = [zeros(size(B,1))]
    v = Array{Vector{Float64}}(0)
	for n = 1:N
		push!(v, -K[n]*z[n] - C[n])
		push!(z, A*z[n] + B*v[n])
	end
	return z, v
end

function apply_LQ_gains(A::VMF, B::VMF, K::VMF, C::VVF)
	N = length(K)

	z = [zeros(size(B[1],1))]
	v = Vector{Float64}[]
	for n = 1:N
		push!(v, -K[n]*z[n] - C[n])
		push!(z, A[n]*z[n] + B[n]*v[n])
	end
	return z, v
end

