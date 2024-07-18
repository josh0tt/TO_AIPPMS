######################################################################
# gradients.jl
# generation of gradients
######################################################################
 
@everywhere function gradients!(gpm::GaussianProcessManager, ad::MF, bd::MF, sad::MF, xd::VVF, ud::VVF, sample_actions::VF, tm::TrajectoryManager)
	# GP gradients
	an = compute_ans(gpm, xd, sample_actions, tm)
	ad[1,:] = an[1] # an_x
	ad[2,:] = an[2]	# an_y

	dJdxN = compute_final_penalty_grad(xd, tm)
	ad[1,end] += dJdxN[1]
	ad[2,end] += dJdxN[2]

	ni =  1
	for n = 0:(length(xd) - 2)#(tm.N-1)
		# quadratic boundary
		if tm.barrier_cost > 0.0
			xnx = xd[n+1][1]
			xny = xd[n+1][2]
			xmax = x_max(gpm)
			xmin = x_min(gpm)
			ymax = y_max(gpm)
			ymin = y_min(gpm)
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
end


# computes a_n, derivative of c_k wrt x_n
# returns tuple containing elements of an

# this was equation 36 from the tutorial
@everywhere function compute_ans(gpm::GaussianProcessManager, xd::VVF, sample_actions::VF, tm::TrajectoryManager)
	x = [xd[i][1] for i in 1:length(xd)]
	y = [xd[i][2] for i in 1:length(xd)]
	an_stacked = Zygote.gradient(X -> sum(query_sequence(gpm, X, sample_actions, gpm.GP, tm, xd)), [x; y])

	an_x = an_stacked[1][1:length(xd)]
	an_y = an_stacked[1][length(xd)+1:end]

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

@everywhere function convert_perc2xy(gpm::GaussianProcessManager, tm::TrajectoryManager, sample_action, xd::VVF)
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

@everywhere function convert_xy2perc(gpm::GaussianProcessManager, tm::TrajectoryManager, sample_actions_xy::VVF, xd::VVF)
	x = [xd[i][1] for i in 1:length(xd)]
	y = [xd[i][2] for i in 1:length(xd)]
	drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]
	sample_actions = [drill_idx[i]/length(xd) for i in 1:length(drill_idx)]
	return sample_actions
end

@everywhere function compute_sample_selection_grad(gpm::GaussianProcessManager, xd::VVF, sample_actions::VF, tm::TrajectoryManager)
	x = [xd[i][1] for i in 1:length(xd)]
	y = [xd[i][2] for i in 1:length(xd)]

	# this is a moving target. When your trajectory gets near the desired drills, your drills want to move again
	sanz = Zygote.gradient(SA -> sum(query_sequence(gpm, [x; y], SA, gpm.GP, tm, xd)), sample_actions)
	san = sanz[1]

	return san
end

@everywhere function update_trajectory(gpm::GaussianProcessManager, tm::TrajectoryManager, xd::VVF, ud::VVF)
	return xd[1:(length(xd)-tm.sample_cost)], ud[1:(length(ud)-tm.sample_cost)] # note we -1 to ud[] because ud is of length 300 whereas xd is of length 301 (there is no control taken at the final location)
end

@everywhere function add_sample(gpm::GaussianProcessManager, tm::TrajectoryManager, xd::VVF, ud::VVF, sample_actions::VF, old_score::Float64)
	new_sample = rand(tm.rng, Distributions.Uniform(0.0,1.0))
	new_sample_actions = vcat(sample_actions, new_sample)
	new_sample_actions_xy = [convert_perc2xy(gpm, tm, new_sample_actions[i], xd) for i in 1:length(new_sample_actions)]
	# NOTE: we have to fix the current sample actions in x,y coordinates before updating the trajectory
	# otherwise the percentages will change when traj is shortened
	new_xd, new_ud = update_trajectory(gpm, tm, xd, ud)
	new_sample_actions = convert_xy2perc(gpm, tm, new_sample_actions_xy, new_xd)

	# gps_actual, xfs, xsa, cs, bs, ts_actual, νₚ_actual
	# _, _, _, _, _, new_score_actual, _ = total_score(gpm, tm, new_xd, new_ud, new_sample_actions)
	new_score_actual, _, _, _, _, _, _ = total_score(gpm, tm, new_xd, new_ud, new_sample_actions)

	# perc improvement should be negative if the score decreased (we want to minimize the score)
	perc_improvement = ((new_score_actual - old_score)/old_score) * 100

	if perc_improvement < sample_addition_threshold
		return new_xd, new_ud, new_sample_actions
	else
		return xd, ud, sample_actions
	end
end

