######################################################################
# ergodic_manager/r2.jl
#
# handles stuff needed for ergodicity
######################################################################


export ErgodicGPManagerR2


@everywhere mutable struct ErgodicGPManagerR2 <: ErgodicGPManager
	domain::Domain				# spatial domain
	GP::GaussianProcess
	σ_drill::Float64
	σ_spec::Float64

	K::Int						# number of Fourier coefficients
	hk::Matrix{Float64}
	phi::Matrix{Float64}		# spatial distribution
	phik::Matrix{Float64}		# distribution's Fourier coefficients

	# constant regardless of phi (depend on k1,k2)
	Lambda::Matrix{Float64}
	#bins::Vector{Int}#Int

	# to speed up computation
	kpixl::Matrix{Float64}
	kpiyl::Matrix{Float64}


	function ErgodicGPManagerR2(d::Domain, GP::GaussianProcess, σ_drill::Float64, σ_spec::Float64, phi::MF, K::Int=5)
		egpm = new()
		egpm.domain = deepcopy(d)
		egpm.GP = GP
		egpm.σ_drill = σ_drill
		egpm.σ_spec = σ_spec

		egpm.K = K
		egpm.hk = zeros(K+1,K+1)
		egpm.phi = deepcopy(phi)
		egpm.phik = zeros(K+1,K+1)
		egpm.Lambda = zeros(K+1,K+1)
		egpm.kpixl = zeros(K+1, d.cells[1])
		egpm.kpiyl = zeros(K+1, d.cells[2])

		Lambda!(egpm)
		kpixl!(egpm)
		hk!(egpm)
		decompose!(egpm)

		return egpm
	end
end

# fills each entry Lambda[k1,k2] in the Lambda matrix
@everywhere function Lambda!(egpm::ErgodicGPManagerR2)
    for k1 = 0:egpm.K, k2 = 0:egpm.K
        den = (1.0 + k1*k1 + k2*k2) ^ 1.5
        egpm.Lambda[k1+1, k2+1] = 1.0 / den
    end
end

@everywhere function kpixl!(egpm::ErgodicGPManagerR2)
	Lx = egpm.domain.lengths[1]
	xmin = x_min(egpm)
	for xi = 1:x_cells(egpm)
		x = xmin + (xi-0.5)*x_size(egpm)
		for k = 0:egpm.K
			egpm.kpixl[k+1,xi] = cos(k*pi*(x-xmin) / Lx)
		end
	end

	Ly = egpm.domain.lengths[2]
	ymin = y_min(egpm)
	for yi = 1:y_cells(egpm)
		y = ymin + (yi-0.5)*y_size(egpm)
		for k = 0:egpm.K
			egpm.kpiyl[k+1,yi] = cos(k*pi*(y-ymin) / Ly)
		end
	end
end

# generates the hk coefficients for the ergodic manager
# these coefficients only need to be computed once
@everywhere function hk!(egpm::ErgodicGPManagerR2)
    for k1 = 0:egpm.K, k2 = 0:egpm.K
        egpm.hk[k1+1,k2+1] = hk_ij(egpm, k1, k2)
    end
end

# computes the coefficients for a specific value of k1 and k2
# called by hk!
@everywhere function hk_ij(egpm::ErgodicGPManagerR2, k1::Int, k2::Int)
	val = 0.0
	for xi = 1:x_cells(egpm)
		cx = egpm.kpixl[k1+1,xi]
		cx2 = cx * cx
		for yi = 1:y_cells(egpm)
			cy = egpm.kpiyl[k2+1,yi]
			val += cx2 * cy * cy * egpm.domain.cell_size
		end
	end

	return sqrt(val)
end



######################################################################
# Computing Fourier coefficients
######################################################################

@everywhere function decompose!(egpm::ErgodicGPManagerR2, d::Matrix{Float64})
    for k1 = 0:egpm.K, k2 = 0:egpm.K
        egpm.phik[k1+1,k2+1] = phi_ij(egpm, k1, k2, d)
    end
    egpm.phi = d
end


# iterate over the state space
@everywhere function phi_ij(egpm::ErgodicGPManagerR2, k1::Int, k2::Int, d::Matrix{Float64})
	val = 0.0
	for xi = 1:x_cells(egpm)
		cx = egpm.kpixl[k1+1,xi]
		for yi = 1:y_cells(egpm)
			cy = egpm.kpiyl[k2+1,yi]
			val += d[xi,yi] * cx * cy * egpm.domain.cell_size
		end
	end
	return val / egpm.hk[k1+1,k2+1]
end

@everywhere decompose!(egpm::ErgodicGPManagerR2) = decompose!(egpm, egpm.phi)

@everywhere function decompose!(egpm::ErgodicGPManagerR2, d::Matrix{Float64})
    for k1 = 0:egpm.K, k2 = 0:egpm.K
        egpm.phik[k1+1,k2+1] = phi_ij(egpm, k1, k2, d)
    end
    egpm.phi = d
end

# """
# `decompose(egpm, traj::VVF)`

# Decomposes a set of positions into a set of `ck` Fourier coefficients.

# """
@everywhere function decompose(egpm::ErgodicGPManagerR2, traj::VVF)

	# trajectory is really of length N+1
	N = length(traj)-1

	# create matrix to hold trajectory's Fourier coefficients
	ck = zeros(egpm.K+1, egpm.K+1)

	# lengths of each dimension
	Lx = egpm.domain.lengths[1]
	Ly = egpm.domain.lengths[2]

	# minimum values in each dimension
	xmin = x_min(egpm)
	ymin = y_min(egpm)

	for k1 = 0:egpm.K
		kpiL1 = k1 * pi / Lx
		for k2 = 0:egpm.K
			kpiL2 = k2 * pi / Ly
			hk = egpm.hk[k1+1, k2+1]
			fk_sum = 0.0
			# now loop over time
			for n = 0:N
				xn = traj[n + 1]
				c1 = cos(kpiL1 * (xn[1]-xmin))
				c2 = cos(kpiL2 * (xn[2]-ymin))
				fk_sum += c1*c2
			end
			ck[k1+1, k2+1] = fk_sum / (hk * (N+1))
		end
	end
	return ck
end


# reconstructs from Fourier coefficients in ck
@everywhere function reconstruct(egpm::ErgodicGPManagerR2, ck::Matrix{Float64})
	# iterate over all bins
	v = egpm.domain.cell_size
	vals = zeros(x_cells(egpm), y_cells(egpm))

	for xi = 1:x_cells(egpm)
		x = x_min(egpm) + (xi-0.5)*x_size(egpm)
		for yi = 1:y_cells(egpm)
			y = y_min(egpm) + (yi-0.5)*y_size(egpm)
			for k1 = 0:egpm.K
				cx = egpm.kpixl[k1+1,xi]
				for k2 = 0:egpm.K
					cy = egpm.kpixl[k2+1,yi]
					vals[xi,yi] += ck[k1+1,k2+1]*cx*cy/egpm.hk[k1+1,k2+1]
				end
			end
		end
	end
	return vals
end
