######################################################################
# examples.jl
#
# This used to be in constructors of ergodic managers
# It added a lot of clutter and I didn't like it there
######################################################################
@everywhere function ErgodicManagerR2(example_name::String; K::Int=5, bins::Int=100)
	L = 1.0
	d = Domain([0.,0.], [L, L], [bins,bins])

	if example_name == "single gaussian"
		mu = [L/2.0, L/2.0]
		Sigma = 0.03 * [1 0; 0 1]
		phi = gaussian(d, mu, Sigma)
		return ErgodicManagerR2(d, phi, K, bins)
	elseif example_name == "double gaussian"
		mu1 = [0.3, 0.7]
		Sigma1 = 0.025 * [1 0; 0 1]
		mu2 = [0.7, 0.3]
		Sigma2 = 0.025 * [1 0; 0 1]
		#phi = gaussian(d, [mu1,mu2], [Sigma1, Sigma2], [.5,.5])
		phi = gaussian(d, [mu1,mu2], [Sigma1, Sigma2])
		return ErgodicManagerR2(d, phi, K, bins)

	elseif example_name == "double gaussian weights"
		mu1 = [0.3, 0.7]
		Sigma1 = 0.025 * [1 0; 0 1]
		mu2 = [0.7, 0.3]
		Sigma2 = 0.025 * [1 0; 0 1]
		weights =[0.8, 1]
		#phi = gaussian(d, [mu1,mu2], [Sigma1, Sigma2], [.5,.5])
		phi = gaussian(d, [mu1,mu2], [Sigma1, Sigma2], weights)
		return ErgodicManagerR2(d, phi, K, bins)
	else
		error("example name not recognized")
	end
end

@everywhere function ErgodicManagerSE2(example_name::String; K::Int=5,bins::Int=50)
	L = 1.0
	d = Domain([0.,0.,-pi], [L,L,pi], [bins,bins,bins])
	println("h2")
	println("h3")

	if example_name == "single gaussian"
		mu = [L/2.0, L/2.0, 0]
		Sigma = 0.03 * [1 0 0; 0 1 0; 0 0 1]
		phi = gaussian(d, mu, Sigma)
		return ErgodicManagerSE2(d, phi, K)

	elseif example_name == "double gaussian"
		# How I should do it...
		mu1 = [0.3, 0.7]
		Sigma1 = 0.025* [1 0; 0 1]
		mu2 = [0.7, 0.3]
		Sigma2 = 0.025* [1 0; 0 1]
		phi = gaussian(d, mu1, Sigma1, mu2, Sigma2)
		return ErgodicManagerR2(d, phi, K)

	else
		error("example name not recognized")
	end
	return em
end
