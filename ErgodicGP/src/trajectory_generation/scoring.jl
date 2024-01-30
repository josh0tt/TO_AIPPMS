######################################################################
# scoring.jl
######################################################################
# """
# `ergodic_score(em, traj::VVF)`

# First breaks down the trajectory into components ck.
# """
@everywhere function ergodic_score(egpm::ErgodicGPManager, traj::VVF)
	ck = decompose(egpm, traj)
	return ergodic_score(egpm, ck)
end

@everywhere function ergodic_score(egpm::ErgodicGPManager, ck::Matrix{Float64})
	val = 0.0
	for (i, L) in enumerate(egpm.Lambda)
		d = egpm.phik[i] - ck[i]
		val += L * d * d
	end
	return val
end

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


@everywhere function barrier_score(egpm::ErgodicGPManager, xd::VVF, c::Float64)
	if c == 0.0; return 0.0; end

	bs = 0.0
	xmax = x_max(egpm)
	ymax = y_max(egpm)
	xmin = x_min(egpm)
	ymin = y_min(egpm)

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

@everywhere function barrier_score(egpm::ErgodicGPManager, sample_actions::VF, c::Float64)
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

@everywhere function query_sequence(egpm::ErgodicGPManager, X, GP, tm::TrajectoryManager) # no sample selection
	x_hist = X[1:(tm.N+1)]
	y_hist = X[(tm.N+2):end]

	post_GP = deepcopy(GP)
	for i in 1:length(x_hist)
		x_samp = [[x_hist[i],y_hist[i]]]
		y_samp = GP.m([x_hist[i],y_hist[i]])
		# NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
		# for normal dist whereas our GP setup uses σ²_n
		post_GP = posterior(post_GP, x_samp , [y_samp], [egpm.σ_spec^2])
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

@everywhere function query_sequence(egpm::ErgodicGPManager, X, sample_actions::VF, GP::GaussianProcess, tm::TrajectoryManager, xd::VVF) # with sample selection
	x_hist = [xd[i][1] for i in 1:length(xd)]
	y_hist = [xd[i][2] for i in 1:length(xd)]

	post_GP = GP

	for i in 1:length(x_hist)
		x_samp = [[x_hist[i],y_hist[i]]]
		y_samp = GP.m([x_hist[i],y_hist[i]])
		# NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
		# for normal dist whereas our GP setup uses σ²_n
		post_GP = posterior(post_GP, x_samp , [y_samp], [egpm.σ_spec^2])
	end

	# NOTE: the order that the samples are added doesn't matter
	for i in 1:length(sample_actions)
		sample_action_xy = convert_perc2xy(egpm, tm, sample_actions[i], xd)
		x_samp = [[sample_action_xy[1],sample_action_xy[2]]]
		y_samp = GP.m([sample_action_xy[1],sample_action_xy[2]])
		post_GP = posterior(post_GP, x_samp , [y_samp], [egpm.σ_drill]) # dont square this to prevent singularity
	end

	μₚ, νₚ, S, EI = query(post_GP)
	
	obj = objective == "expected_improvement" ? EI : νₚ

	return obj
end

@everywhere function total_score(egpm::ErgodicGPManager, tm::TrajectoryManager, xd::VVF, ud::VVF)
	es = tm.q * ergodic_score(egpm, xd)
	xfs = endpt_score(tm, xd)
	cs = control_score(ud, tm.R, tm.h)
	bs = barrier_score(egpm, xd, tm.barrier_cost)
	return es + xfs + cs + bs
end

@everywhere function optim_total_score(egpm::ErgodicGPManager, tm::TrajectoryManager, xd::VVF, ud::VVF, sample_actions::VF)
	x_hist = [xd[i][1] for i in 1:length(xd)]
	y_hist = [xd[i][2] for i in 1:length(xd)]
	sa = sample_actions

	gps_actual = tm.q * sum(query_sequence(egpm, [x_hist; y_hist], sa, egpm.GP, tm, xd))
	xfs = xfs = ([x_hist[end],y_hist[end]] - tm.xf)'*tm.Qf*([x_hist[end],y_hist[end]] - tm.xf) #endpt_score(tm, xd)
	ts_actual = gps_actual + xfs #+ xsa #+ cs + bs

	return ts_actual
end

@everywhere function gp_score(egpm::ErgodicGPManager, tm::TrajectoryManager, xd::VVF, ud::VVF, sample_actions::VF)
	x_hist = [xd[i][1] for i in 1:length(xd)]
	y_hist = [xd[i][2] for i in 1:length(xd)]
	sa = sample_actions


	gps_actual = tm.q * sum(query_sequence(egpm, [x_hist; y_hist], sa, egpm.GP, tm, xd))

	return gps_actual
end