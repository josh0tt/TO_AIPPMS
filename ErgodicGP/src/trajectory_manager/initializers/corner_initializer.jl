######################################################################
# corner_initializer.jl
######################################################################

# """
# `ci = CornerInitializer()`

# Takes a trajectory to the farthest corner.
# """
@everywhere mutable struct CornerInitializer <: Initializer end


# Assumes we our domain is 2d
# corner 1 = 0,0
# corner 2 = 1,0
# corner 3 = 0,1
# corner 4 = 1,1
@everywhere function initialize(ci::CornerInitializer, egpm::ErgodicGPManager, tm::TrajectoryManager)

	# dimensionality of state space
	x0 = tm.x0
	n = length(x0)

	# first corner: 0,0
	dx1 = x_min(egpm) - x0[1]
	dy1 = y_min(egpm) - x0[2]
	dc1 = sqrt(dx1*dx1 + dy1*dy1)

	# second corner: 1,0
	dx2 = x_max(egpm) - x0[1]
	dy2 = y_min(egpm) - x0[2]
	dc2 = sqrt(dx2*dx2 + dy2*dy2)

	# third corner: 0,.5
	dx3 = x_min(egpm) - x0[1]
	dy3 = y_max(egpm) - x0[2]
	dc3 = sqrt(dx3*dx3 + dy3*dy3)

	# fourth corner: 1,.5
	dx4 = x_max(egpm) - x0[1]
	dy4 = y_max(egpm) - x0[2]
	dc4 = sqrt(dx4*dx4 + dy4*dy4)

	bi = argmax([dc1,dc2,dc3,dc4])

	bc = [x_min(egpm), y_min(egpm)]
	if bi == 2
		bc = [x_max(egpm), y_min(egpm)]
	elseif bi == 3
		#bc = [0.,.5]
		bc = [x_min(egpm),y_max(egpm)]
	elseif bi == 4
		#bc = [1.,.5]
		bc = [x_max(egpm), y_max(egpm)]
	end

	# NOTE: this will always choose top right corner (goal location )
	bc = xf #[x_max(egpm), y_max(egpm)]
	# could now do point initializer to the selected corner
	return initialize(PointInitializer(bc), egpm, tm)
end
