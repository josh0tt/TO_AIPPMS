######################################################################
# scoring.jl
######################################################################
@everywhere function control_score(ud::VVF, R::Matrix{Float64}, h::Float64)
	cs = 0.0
	num_u = length(ud[1])
	for ui in ud
		for j = 1:num_u
		   cs += R[j,j] * ui[j] * ui[j]
		end
	end
	return 0.5 * h * cs
end

@everywhere function control_score(tm::TrajectoryManager, ud::VVF)
	return control_score(ud, tm.R, tm.h)
end


control_score(ud::VVF) = control_score(ud, eye(2), 1.0)
@everywhere function control_score(ud::VVF, N::Int)
	cs = 0.0
	for n = 1:N
		num_u = length(ud[n])
		for j = 1:num_u
			cs += ud[n][j] * ud[n][j]
		end
	end
	return cs
end

@everywhere function control_effort(ud::VVF, N::Int=length(ud))
	cs = 0.0
	for n = 1:N
		udx = ud[n][1]
		udy = ud[n][1]
		cs += sqrt(udx*udx + udy*udy)
	end
	return cs
end


@everywhere function barrier_score(gpm::GaussianProcessManager, xd::VVF, c::Float64)
	if c == 0.0; return 0.0; end

	bs = 0.0
	xmax = x_max(gpm)
	ymax = y_max(gpm)
	xmin = x_min(gpm)
	ymin = y_min(gpm)

	for xi in xd
		if (xi[1] > xmax)
			dx = xi[1] - xmax
			bs += c * dx * dx
		elseif (xi[1] < xmin)
			dx = xi[1] - xmin
			bs += c * dx * dx
		end
		if (xi[2] > ymax)
			dy = xi[2] - ymax
			bs += c * dy * dy
		elseif (xi[2] < ymin)
			dy = xi[2] - ymin
			bs += c * dy * dy
		end
	end
	return bs
end

@everywhere function barrier_score(gpm::GaussianProcessManager, sample_actions::VF, c::Float64)
	if c == 0.0; return 0.0; end

	bs = 0.0
	for sa in sample_actions
		if sa > 1.0
			dx = sa - 1.0
			bs += c * dx * dx
		elseif sa < 0.0
			dx = -sa
			bs += c * dx * dx
		end
	end
	return bs
end

@everywhere function endpt_score(tm::TrajectoryManager, xd::VVF)
	xfs = (xd[end] - tm.xf)'*tm.Qf*(xd[end] - tm.xf)
end

@everywhere function query_sequence(gpm::GaussianProcessManager, X, GP, tm::TrajectoryManager) # no sample selection
	x_hist = X[1:(tm.N+1)]
	y_hist = X[(tm.N+2):end]

	post_GP = deepcopy(GP)
	for i in 1:length(x_hist)
		x_samp = [[x_hist[i],y_hist[i]]]
		y_samp = GP.m([x_hist[i],y_hist[i]])
		# NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
		# for normal dist whereas our GP setup uses σ²_n
		post_GP = posterior(post_GP, x_samp , [y_samp], [gpm.σ_spec^2])
	end

	μₚ, νₚ, S, EI = query(post_GP)
	
	obj = objective == "expected_improvement" ? sum(EI) : sum(νₚ)

	return obj #(μₚ, νₚ, S)
end

@everywhere function find_nearest_traj_drill_pts(x_hist, y_hist, sample_actions_xy, action_idx)
	dist_to_drill_pt = [norm(sample_actions_xy[action_idx] - [x_hist[i], y_hist[i]]) for i in 1:length(x_hist)]
	min_dist_idx = argmin(dist_to_drill_pt)

	return min_dist_idx
end

@everywhere function query_sequence(gpm::GaussianProcessManager, X, sample_actions, GP, tm::TrajectoryManager, xd::VVF) # with sample selection
	N = Int(length(X)/2) - 1
	x_hist = X[1:(N+1)]
	y_hist = X[(N+2):end]

	post_GP = deepcopy(GP)

	for i in 1:length(x_hist)
		x_samp = [[x_hist[i],y_hist[i]]]
		y_samp = GP.m([x_hist[i],y_hist[i]])
		# NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
		# for normal dist whereas our GP setup uses σ²_n
		post_GP = posterior(post_GP, x_samp , [y_samp], [gpm.σ_spec^2])
		
	end

	# NOTE: the order that the samples are added doesn't matter
	for i in 1:length(sample_actions)
		sample_action_xy = convert_perc2xy(gpm, tm, sample_actions[i], xd)
		x_samp = [[sample_action_xy[1],sample_action_xy[2]]]
		y_samp = GP.m([sample_action_xy[1],sample_action_xy[2]])
		post_GP = posterior(post_GP, x_samp , [y_samp], [gpm.σ_drill]) # dont square this to prevent singularity
	end

	μₚ, νₚ, S, EI = query(post_GP)
	
	obj = objective == "expected_improvement" ? EI : νₚ

	return obj
end

@everywhere function total_score(gpm::GaussianProcessManager, tm::TrajectoryManager, xd::VVF, ud::VVF, sample_actions::VF)
	x = [xd[i][1] for i in 1:length(xd)]
	y = [xd[i][2] for i in 1:length(xd)]

	# NOTE: this is needed because sample_actions might not fall exactly on the trajectory so we get the closest index
	# with the two lines below to use in computing the variance for the actual score (i.e. this converts from requested to actual)
	sample_actions_xy = [convert_perc2xy(gpm, tm, sample_actions[i], xd) for i in 1:length(sample_actions)]
	drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]

	# NOTE: the GP score is evaluated using the requested sample locations NOT the true sample locations
	# we then penalize for being far from the requested locations in gradients.jl
	νₚ_actual = query_sequence(gpm, [x; y], [drill_idx[i]/length(xd) for i in 1:length(drill_idx)], gpm.GP, tm, xd)
	#gps_requested = tm.q * query_sequence(gpm, [x; y], sample_actions, gpm.GP, tm, xd)
	gps_actual = tm.q * sum(νₚ_actual)
	xfs = endpt_score(tm, xd)
	# NOTE: I'm removing the offset penalty from sample selections since they are constrained to be along the traj anyways
	xsa = 0.0
	cs = control_score(ud, tm.R, tm.h)
	bs = barrier_score(gpm, xd, tm.barrier_cost) + barrier_score(gpm, sample_actions, tm.barrier_cost)
	#ts_requested = gps_requested + xfs + xsa + cs + bs
	ts_actual    = gps_actual    + xfs + xsa + cs + bs

	# return gps_requested, gps_actual, xfs, xsa, cs, bs, ts_requested, ts_actual, νₚ_actual
	return gps_actual, xfs, xsa, cs, bs, ts_actual, νₚ_actual # NOTE: the index matters here for drawing outputs from total_score() so be careful adding additional score items. Only add them to the end of the output e.g. after vp_actual

end


@everywhere function optim_total_score(gpm::GaussianProcessManager, tm::TrajectoryManager, xd::VVF, ud::VVF, sample_actions::VF)
	x_hist = [xd[i][1] for i in 1:length(xd)]
	y_hist = [xd[i][2] for i in 1:length(xd)]
	sa = sample_actions

	# NOTE: the GP score is evaluated using the requested sample locations NOT the true sample locations
	# we then penalize for being far from the requested locations in gradients.jl
	gps_actual = tm.q * sum(query_sequence(gpm, [x_hist; y_hist], sa, gpm.GP, tm, xd))
	xfs = xfs = ([x_hist[end],y_hist[end]] - tm.xf)'*tm.Qf*([x_hist[end],y_hist[end]] - tm.xf) #endpt_score(tm, xd)
	ts_actual = gps_actual + xfs #+ xsa #+ cs + bs
	
	return ts_actual
end