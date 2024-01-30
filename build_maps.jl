using JLD2
using Distributed
using Random
using Statistics
using Distributions
using Plots
using KernelFunctions
using JLD2
using LinearAlgebra
using AbstractGPs
include("CustomGP.jl")
include("parameters.jl")
include("SBO_AIPPMS/GP_BMDP_Rover/rover_pomdp.jl")


function get_neighbors(idx::Int, map_size::Tuple{Int, Int})
	pos = [CartesianIndices(map_size)[idx].I[1], CartesianIndices(map_size)[idx].I[2]]
	neighbors = [pos+[0,1], pos+[0,-1], pos+[1,0], pos+[-1,0]]
	bounds_neighbors = []
	for i in 1:length(neighbors)
		if inbounds(map_size, RoverPos(neighbors[i][1], neighbors[i][2]))
			append!(bounds_neighbors, [neighbors[i]])
		end
	end

	bounds_neighbors_idx = [LinearIndices(map_size)[bounds_neighbors[i][1], bounds_neighbors[i][2]] for i in 1:length(bounds_neighbors)]
	return bounds_neighbors_idx
end

function inbounds(map_size::Tuple{Int, Int}, pos::RoverPos)
    if map_size[1] >= pos[1] > 0 && map_size[2] >= pos[2] > 0
        # i = abs(s[2] - pomdp.map_size[1]) + 1
        # j = s[1]
        return true
    else
        return false
    end
end

function build_map(rng::RNG, number_of_sample_types::Int, map_size::Tuple{Int, Int}) where {RNG<:AbstractRNG}
	sample_types = collect(0:(1/number_of_sample_types):(1-1/number_of_sample_types))
	init_map = rand(rng, sample_types, map_size[1], map_size[2])
	new_map = zeros(map_size)

	p_neighbors = 0.95

	for i in 1:(map_size[1]*map_size[2])
		if i == 1
			continue
		else
			if rand(rng) < p_neighbors
				neighbor_values = init_map[get_neighbors(i, map_size)]
				new_map[i] = round(mean(neighbor_values),digits=1)
				#true_map[i] = true_map[i-1]
			else
				continue
			end
		end
	end

	return new_map
end

function build_rand_maps()
    i = 1
    idx = 1
    seed = 1234
    while idx <= num_trials
        @show i
        rng = MersenneTwister(seed+i)

        true_map = build_map(rng, number_of_sample_types, map_size_sboaippms)
        JLD2.save(path_name * "/true_maps/true_map$(idx).jld", "true_map", true_map)
        i += 1
        idx += 1
    end
end

function build_large_maps()
	i = 1
	idx = 1
	seed = 1234
	while idx <= num_trials
		@show i
		rng = MersenneTwister(seed+i)

		true_map = build_map(rng, number_of_sample_types, map_size_sboaippms)

		##########################################################################
		# GP
		##########################################################################
		L = length_scale*3
		gp = AbstractGPs.GP(with_lengthscale(SqExponentialKernel(), L))
		# build GP from true map and use posterior mean as the true map to get smoother true map

		for i in 1:length(true_map)
			# X = [X_query[i]]
			X = [[CartesianIndices(query_size)[i].I[1], CartesianIndices(query_size)[i].I[2]]] ./ 10 #./100

			y = [true_map[i]]
			gp = AbstractGPs.posterior(gp(X, 0.1), y)
		end

		X_plot_query = [[i,j] for i = range(0, 1, length=(640)), j = range(0, 1, length=(640))]

		X_plot_query = reshape(X_plot_query, size(X_plot_query)[1]*size(X_plot_query)[2])

		plot_map = reshape(mean(gp(X_plot_query)), (round(Int, sqrt(length(X_plot_query))),round(Int, sqrt(length(X_plot_query)))))
		true_map = plot_map

		JLD2.save(path_name * "/true_maps/true_map$(idx).jld", "true_map", true_map)
		heatmap(true_map)
		savefig(path_name * "/true_maps/true_map$(idx).png")

		i += 1
		idx += 1
	end
end

function build_gp_maps()
    i = 1
    idx = 1
    seed = 1234
    while idx <= num_trials
        @show i
        rng = MersenneTwister(seed+i)

        true_map = build_map(rng, number_of_sample_types, map_size_sboaippms)

		##########################################################################
		# GP
		##########################################################################
		L = length_scale*3
		gp = AbstractGPs.GP(with_lengthscale(SqExponentialKernel(), L))
		# build GP from true map and use posterior mean as the true map to get smoother true map

		for i in 1:length(true_map)
			# X = [X_query[i]]
			X = [[CartesianIndices(query_size)[i].I[1], CartesianIndices(query_size)[i].I[2]]] ./ 10 #./100

			y = [true_map[i]]
			gp = AbstractGPs.posterior(gp(X, 0.1), y)
		end

		X_plot_query = [[i,j] for i = range(0, 1, length=(bins_x+1)), j = range(0, 1, length=(bins_y+1))]

		X_plot_query = reshape(X_plot_query, size(X_plot_query)[1]*size(X_plot_query)[2])

		plot_map = reshape(mean(gp(X_plot_query)), (round(Int, sqrt(length(X_plot_query))),round(Int, sqrt(length(X_plot_query)))))
		true_map = plot_map

        JLD2.save(path_name * "/true_maps/true_map$(idx).jld", "true_map", true_map)
		heatmap(true_map)
		savefig(path_name * "/true_maps/true_map$(idx).png")

        i += 1
        idx += 1
    end
end

# build_rand_maps()
# build_gp_maps()
# build_large_maps()
