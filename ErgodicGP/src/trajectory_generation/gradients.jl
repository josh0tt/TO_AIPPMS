######################################################################
# gradients.jl
# generation of gradients
######################################################################

# Only first two states matter for ergodic score and barrier penalty
# Assumes ad has been initialized with zeros; that is, ad[3:end, ni] = 0.0

@everywhere function gradients!(egpm::ErgodicGPManager, ad::MF, bd::MF, sad::MF, xd::VVF, ud::VVF, sample_actions::VF, tm::TrajectoryManager)
	ck = decompose(egpm, xd)

	ni =  1
	#ck = decompose(em, xd)
	for n = 0:(length(xd) - 2)#n = 0:(tm.N-1)

		# ergodic gradients
		an = compute_ans(egpm, xd, tm, n, ck)
		for i = 1:length(an)
			ad[i,ni] = an[i]
		end

		# quadratic boundary
		if tm.barrier_cost > 0.0
			xnx = xd[n+1][1]
			xny = xd[n+1][2]
			xmax = x_max(egpm)
			xmin = x_min(egpm)
			ymax = y_max(egpm)
			ymin = y_min(egpm)
			if (xnx > xmax)
				ad[1,ni] += tm.barrier_cost * 2.0 * (xnx - xmax)
			elseif xnx < xmin
				ad[1,ni] += tm.barrier_cost * 2.0 * (xnx - xmin)
			end
			if xny > ymax
				ad[2,ni] += tm.barrier_cost * 2.0 * (xny - ymax)
			elseif xny < ymin
				ad[2,ni] += tm.barrier_cost * 2.0 * (xny - ymin)
			end
		end

		# control gradients
		bd[:,ni] = tm.h * tm.R * ud[ni]

		ni += 1
	end

	n = length(xd)-1 #tm.N
	an = compute_ans(egpm, xd, tm, n, ck)
	for i = 1:length(an)
		ad[i,ni] = an[i]
	end
	if tm.barrier_cost > 0.0
		xnx = xd[n+1][1]
		xny = xd[n+1][2]
		xmax = x_max(egpm)
		xmin = x_min(egpm)
		ymax = y_max(egpm)
		ymin = y_min(egpm)
		if (xnx > xmax)
			ad[1,ni] += tm.barrier_cost * 2.0 * (xnx - xmax)
		elseif xnx < xmin
			ad[1,ni] += tm.barrier_cost * 2.0 * (xnx - xmin)
		end
		if xny > ymax
			ad[2,ni] += tm.barrier_cost * 2.0 * (xny - ymax)
		elseif xny < ymin
			ad[2,ni] += tm.barrier_cost * 2.0 * (xny - ymin)
		end
	end
	
	dJdxN = compute_final_penalty_grad(xd, tm)
	ad[1,end] += dJdxN[1]
	ad[2,end] += dJdxN[2]

end


# computes a_n, derivative of c_k wrt x_n
# returns tuple containing elements of an

# this was equation 36 from the tutorial
@everywhere function compute_ans(egpm::ErgodicGPManager, xd::VVF, tm::TrajectoryManager, n::Int, ck::Matrix{Float64})
	x = xd[n + 1][1]
	y = xd[n + 1][2]

	Lx = egpm.domain.lengths[1]
	Ly = egpm.domain.lengths[2]

	an_x = 0.0
	an_y = 0.0
	 
	xm = x_min(egpm)
	ym = y_min(egpm)

	for k1 = 0:egpm.K
		for k2 = 0:egpm.K
			hk = egpm.hk[k1+1,k2+1]

			dFk_dxn1 = -k1*pi*sin(k1*pi*(x-xm)/Lx)*cos(k2*pi*(y-ym)/Ly) / (hk*Lx)
			dFk_dxn2 = -k2*pi*cos(k1*pi*(x-xm)/Lx)*sin(k2*pi*(y-ym)/Ly) / (hk*Ly)

			c = egpm.Lambda[k1+1,k2+1] * (ck[k1+1,k2+1] - egpm.phik[k1+1,k2+1])
			an_x += c*dFk_dxn1
			an_y += c*dFk_dxn2
		end
	end
	an_x *= 2.0/(tm.N+1)
	an_y *= 2.0/(tm.N+1)
	return an_x, an_y
end

@everywhere function compute_final_penalty_grad(xd::VVF, tm::TrajectoryManager)
	dJdxN = 2*tm.Qf*(xd[end] - tm.xf)

	return dJdxN
end


@everywhere function find_nearest_traj_drill_pts(x_hist, y_hist, sample_actions_xy, action_idx)
	dist_to_drill_pt = [norm(sample_actions_xy[action_idx] - [x_hist[i], y_hist[i]]) for i in 1:length(x_hist)]
	min_dist_idx = argmin(dist_to_drill_pt)

	return min_dist_idx
end

@everywhere function convert_perc2xy(egpm::ErgodicGPManager, tm::TrajectoryManager, sample_action, xd::VVF)
	if sample_action >= 1.0
		return xd[end]
	elseif convert(Int, floor(length(xd)*sample_action)) <= 1
		return xd[1]
	end

	x_hist = [xd[i][1] for i in 1:length(xd)]
	y_hist = [xd[i][2] for i in 1:length(xd)]
	p = [i/length(xd) for i in 1:length(xd)]
	interp_linear_x = linear_interpolation(p, x_hist,extrapolation_bc=Line())
	interp_linear_y = linear_interpolation(p, y_hist,extrapolation_bc=Line())
	sa_x = interp_linear_x(sample_action)
	sa_y = interp_linear_y(sample_action)
	sample_action_xy = [sa_x, sa_y]

	return sample_action_xy
end

@everywhere function convert_xy2perc(egpm::ErgodicGPManager, tm::TrajectoryManager, sample_actions_xy::VVF, xd::VVF)
	x = [xd[i][1] for i in 1:length(xd)]
	y = [xd[i][2] for i in 1:length(xd)]
	drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]
	sample_actions = [drill_idx[i]/length(xd) for i in 1:length(drill_idx)]
	return sample_actions
end

@everywhere function compute_sample_selection_grad(egpm::ErgodicGPManager, xd::VVF, sample_actions::VF, tm::TrajectoryManager)
	x = [xd[i][1] for i in 1:length(xd)]
	y = [xd[i][2] for i in 1:length(xd)]

	# this is a moving target. When your trajectory gets near the desired drills, your drills want to move again
	sanz = Zygote.gradient(SA -> sum(query_sequence(egpm, [x; y], SA, egpm.GP, tm, xd)), sample_actions)
	san = sanz[1]
	return san
end

@everywhere function update_trajectory(egpm::ErgodicGPManager, tm::TrajectoryManager, xd::VVF, ud::VVF)
	return xd[1:(length(xd)-tm.sample_cost)], ud[1:(length(ud)-tm.sample_cost)] # note we -1 to ud[] because ud is of length 300 whereas xd is of length 301 (there is no control taken at the final location)
end

@everywhere function add_sample(egpm::ErgodicGPManager, tm::TrajectoryManager, xd::VVF, ud::VVF, sample_actions::VF, old_score::Float64)
	new_sample = rand(tm.rng, Distributions.Uniform(0.0,1.0))
	new_sample_actions = vcat(sample_actions, new_sample)
	new_sample_actions_xy = [convert_perc2xy(egpm, tm, new_sample_actions[i], xd) for i in 1:length(new_sample_actions)]
	# NOTE: we have to fix the current sample actions in x,y coordinates before updating the trajectory
	# otherwise the percentages will change when traj is shortened
	new_xd, new_ud = update_trajectory(egpm, tm, xd, ud)
	new_sample_actions = convert_xy2perc(egpm, tm, new_sample_actions_xy, new_xd)

	#gps_requested, gps_actual, xfs, xsa, cs, bs, ts_requested, ts_actual
	#new_score_actual = optim_total_score(egpm, tm, new_xd, new_ud, new_sample_actions)
	new_score = gp_score(egpm, tm, new_xd, new_ud, new_sample_actions)
	#_, _, _, _, _, new_score_actual, _ = total_score(egpm, tm, new_xd, new_ud, new_sample_actions)
	
	perc_improvement = ((new_score - old_score)/old_score) * 100

	#if new_score_actual + sample_addition_threshold < old_score # we want to MINIMIZE the score
	if perc_improvement < sample_addition_threshold
		return new_xd, new_ud, new_sample_actions
	else
		return xd, ud, sample_actions
	end
end