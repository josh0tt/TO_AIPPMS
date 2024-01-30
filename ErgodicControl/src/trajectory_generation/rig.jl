######################################################################
# rig.jl
#
# Used to be rig3.jl, but it worked the best so it got updated
# like rig.jl, but avoiding the near function
######################################################################

export rig_trajectory, Node, tree2traj

#typealias Loc Tuple{Float64, Float64}
const Loc = Tuple{Float64, Float64}

mutable struct Node
    x::Loc			# location
    c::Int			# cost (really closer to depth)
    ig::Float64		# info gathered

    id::Int			# id of the node
    pnid::Int		# id of the parent node

    closed::Bool
    cx::Int
    cy::Int
end

get_pnid(n::Node) = n.pnid

# returns the distance between location x1 and location x2
function get_dist(x1::Loc, x2::Loc)
    dx = x1[1] - x2[1]
    dy = x1[2] - x2[2]
    return sqrt(dx*dx + dy*dy)
end
function get_dist2(x1::Loc, x2::Loc)
    dx = x1[1] - x2[1]
    dy = x1[2] - x2[2]
    return dx*dx + dy*dy
end

# returns node nearest to x_samp
function nearest(x_samp::Loc, V::Vector{Node})
	min_d = Inf #0.0 Note: this was originally set to 0.0 which didn't make sense because then it would always remain minimum distance
	nearest_node = V[1]
	for n in V
		if !n.closed
			d = get_dist(x_samp, n.x)
			if d < min_d
				min_d = d
				nearest_node = n
			end
		end
	end
	return nearest_node
end

function steer(x_start::Loc, x_goal::Loc, td::Float64)
    dx = x_goal[1] - x_start[1]
    dy = x_goal[2] - x_start[2]
    d = sqrt(dx*dx + dy*dy)
	if x_start[1] + td*dx/d <= 0
		x1 = 0.0
	elseif x_start[1] + td*dx/d >= 1
		x1 = 0.99
	else
		x1 = x_start[1] + td*dx/d
	end

	if x_start[2] + td*dy/d <= 0
		x2 = 0.0
	elseif x_start[2] + td*dy/d >= 1
		x2 = 0.99
	else
		x2 = x_start[2] + td*dy/d
	end

	return x1, x2
end


# Returns information gathered at a new cell, given previous trajectory.
#
# arguments:
#   pn is a parent node
#   cx, cy are indices (1-indexed) for eid cell of new location
#   eid is the information matrix (environment)
function get_info(pn::Node, cx::Int, cy::Int, eid::Matrix{Float64}, V::Vector{Node}, dr::Float64)

	# number of times we've been in this cell
	cell_count = 1.0

	# go all the way up the parent nodes
	# cn = current_node
	cn = pn
	while get_pnid(cn) != 0
		if cn.cx == cx && cn.cy == cy
			cell_count += 1.0
		end
		cn = V[cn.pnid]
	end
	# must do one last time for root node
	if cn.cx == cx && cn.cy == cy
		cell_count += 1.0
	end

	# using cell_count, determine info in this cell
	cell_info = max(eid[cx, cy] - (cell_count-1.0)*dr, 0.0)

	# gather dr info, unless cell_info is smaller. Then gather that.
	ig = min(dr, cell_info)

	return ig
end


# determines the index of the cell this value is in
# the domain size is always 1 x 1
# suppose there are 10 cells per side (10 x 10 grid)
# return values for each dimension will be within 1:10
function get_cell(x::Loc, cell_length::Float64)
    x1 = round(Int, x[1] / cell_length, RoundDown) + 1
    x2 = round(Int, x[2] / cell_length, RoundDown) + 1

    return x1, x2
end

function prune(n::Node)
    return false
end

# eid is expected information density
function rig_trajectory(eid::Matrix{Float64}, x0::Vector{Float64}, N::Int; num_points::Int=20)
	# create the start node
	cell_length = 1.0 / size(eid, 1)
	x0_tuple = tuple(x0...)#(x0...)
	cx, cy = get_cell(x0_tuple, cell_length)
	n_start = Node(x0_tuple, 0, 0.0, 1, 0, false, cx, cy)

	td = 0.1		# travel distance per step
	R = 2.0*td		# neighbor radius
	R2 = R*R
	#R = td		# neighbor radius
	dr = sum(eid) / N

	# create V and E
	V = [n_start]		# nodes (aka vertices)
	Vlen = 1

	for i = 1:num_points
		x_samp = (rand(), rand())
		#x_samp = (0.8*rand()+.1, 0.8*rand()+.1)

		n_nearest = nearest(x_samp, V)
		x_feas = steer(n_nearest.x, x_samp, td)

		#for n_near in V
		for vid = Vlen:-1:1
			n_near = V[vid]
			if !n_near.closed
				if get_dist2(x_feas, n_near.x) < R2
					x_new = steer(n_near.x, x_feas, td)

					# technically need an "if no collision here" but
					#  I have no obtacles so it will prob be ok

					# calculate new information and cost
					cx, cy = get_cell(x_new, cell_length)
					ig_new = get_info(n_near, cx, cy, eid, V, dr)
					c_new = n_near.c + 1

					# Find if the node is closed (trajectory is too long)
					cbool = false
					if c_new > N
						cbool = true
					end

					# create the node and decide if it should be pruned or added
					n_new = Node(x_new,c_new,ig_new, Vlen+1, n_near.id, cbool,cx,cy)
					if prune(n_new)
						println("not implemented yet")
					else
						push!(V, n_new)
						Vlen += 1
					end
				end
			end
		end

	end
	return V
end

# converts from tree of nodes into trajectory
function tree2traj(V::Vector{Node})

	# determine the best node in the tree
	max_ig = 0.0
	best_node = V[1]
	for node in V
		if node.ig > max_ig
			max_ig = node.ig
			best_node = node
			#println("max_ig = ", max_ig)
		end
	end

	# now follow this node up to the parent
	cn = best_node
	xdr = Array{Vector{Float64}}(undef, 0)#VVF(tm.N)#Array(Vector{Float64}, 0)
	while get_pnid(cn) != 0
		push!(xdr, [cn.x[1], cn.x[2]])
		cn = V[cn.pnid]
	end
	push!(xdr, [cn.x[1], cn.x[2]])

	xd = xdr[end:-1:1]
	return xd
end
