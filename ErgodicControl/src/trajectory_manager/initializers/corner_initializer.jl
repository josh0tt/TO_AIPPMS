######################################################################
# corner_initializer.jl
######################################################################

"""
`ci = CornerInitializer()`

Takes a trajectory to the farthest corner.
"""
mutable struct CornerInitializer <: Initializer end


# Assumes we our domain is 2d
# corner 1 = 0,0
# corner 2 = 1,0
# corner 3 = 0,1
# corner 4 = 1,1
function initialize(ci::CornerInitializer, em::ErgodicManager, tm::TrajectoryManager)

	# dimensionality of state space
	x0 = tm.x0
	n = length(x0)

	# first corner: 0,0
	dx1 = x_min(em) - x0[1]
	dy1 = y_min(em) - x0[2]
	dc1 = sqrt(dx1*dx1 + dy1*dy1)

	# second corner: 1,0
	dx2 = x_max(em) - x0[1]
	dy2 = y_min(em) - x0[2]
	dc2 = sqrt(dx2*dx2 + dy2*dy2)

	# third corner: 0,.5
	dx3 = x_min(em) - x0[1]
	dy3 = y_max(em) - x0[2]
	dc3 = sqrt(dx3*dx3 + dy3*dy3)

	# fourth corner: 1,.5
	dx4 = x_max(em) - x0[1]
	dy4 = y_max(em) - x0[2]
	dc4 = sqrt(dx4*dx4 + dy4*dy4)

	bi = argmax([dc1,dc2,dc3,dc4])

	bc = [x_min(em), y_min(em)]
	if bi == 2
		bc = [x_max(em), y_min(em)]
	elseif bi == 3
		#bc = [0.,.5]
		bc = [x_min(em),y_max(em)]
	elseif bi == 4
		#bc = [1.,.5]
		bc = [x_max(em), y_max(em)]
	end


	# NOTE: this will always choose top right corner (goal location )
	bc = [x_max(em), y_max(em)]#xf #[x_max(em), y_max(em)]

	# could now do point initializer to the selected corner
	return initialize(PointInitializer(bc), em, tm)
end
