######################################################################
# pto_trajectory.jl
#######################################################################

export pto_trajectory

@everywhere function pto_trajectory(egpm::ErgodicGPManager, tm::TrajectoryManager; verbose::Bool=true, logging::Bool=false, max_iters::Int=100, gps_crit::Float64=0.0, dd_crit::Float64=1e-6)
	xd0, ud0 = initialize(tm.initializer, egpm, tm)
	sample_actions0 = [0.25; 0.5; 0.75]
	pto_trajectory(egpm, tm, xd0, ud0, sample_actions0; verbose=verbose, logging=logging, max_iters=max_iters, gps_crit=gps_crit, dd_crit = dd_crit)
end

@everywhere function pto_trajectory(egpm::ErgodicGPManager, tm::TrajectoryManager, sample_actions0::VF; verbose::Bool=true, logging::Bool=false, max_iters::Int=100, gps_crit::Float64=0.0, dd_crit::Float64=1e-6)
	xd0, ud0 = initialize(tm.initializer, egpm, tm)
	pto_trajectory(egpm, tm, xd0, ud0, sample_actions0; verbose=verbose, logging=logging, max_iters=max_iters, gps_crit=gps_crit, dd_crit = dd_crit)	
end

@everywhere function pto_trajectory(egpm::ErgodicGPManager, tm::TrajectoryManager, xd0::VVF, ud0::VVF, sample_actions0::VF; verbose::Bool=true, logging::Bool=false, max_iters::Int=100, gps_crit::Float64=0.0, dd_crit::Float64=1e-6)

	# let's not overwrite the initial trajectories
	xd = deepcopy(xd0)
	ud = deepcopy(ud0)
	sample_actions = deepcopy(sample_actions0)
	N = tm.N

	# matrices for gradients
	ad = zeros(tm.dynamics.n, length(xd))#zeros(tm.dynamics.n, N+1)
	bd = zeros(tm.dynamics.m, length(ud))#zeros(tm.dynamics.m, N)
	sad = zeros(1, length(sample_actions0)) # sample actions

	# prepare for logging if need be
	if logging
		outfile = open("temp.csv", "w")
		outfile_score_actual = open("temp_score_actual.csv", "w")
		outfile_sample_actions = open("temp_sample_actions.csv", "w")
		outfile_gps_actual = open("gps_actual.csv", "w")
		outfile_xfs = open("xfs.csv", "w")
		outfile_xsa = open("xsa.csv", "w")
		outfile_cs = open("cs.csv", "w")
		outfile_bs = open("bs.csv", "w")
		outfile_traj_length = open("traj_length.csv", "w")
		outfile_num_samples = open("num_samples.csv", "w")
	end

	if verbose; print_header(); end
	i = 1
	not_finished = true
	es=0.; xfs =0.; xsa=0.; cs = 0.; bs = 0.; ts_actual = 0.; dd = 0.; step_size = 0.
	gp_actual = 0.0
	es_prev = 0.0
	es_count = 1
	es_crit = gps_crit
	step_size_flag = false
	best_score_actual = 100000000.0
	best_xd = xd0
	best_ud = ud0
	best_sample_actions = sample_actions0
	sample_actions_optimized = false
	while not_finished

		ad = zeros(tm.dynamics.n, length(xd))
		bd = zeros(tm.dynamics.m, length(ud))
		sad = zeros(1, length(sample_actions)) # sample actions

		# determine gradients used in optimization
		gradients!(egpm, ad, bd, sad, xd, ud, sample_actions, tm)

		A, B = linearize(tm.dynamics, xd, ud, tm.h)
		K, C = LQ(A, B, ad, bd, tm.Qn, tm.Rn, length(xd)-1)#tm.N)
		zd, vd = apply_LQ_gains(A, B, K, C)

		# determine step size and descend
		step_size = get_step_size(tm.descender, egpm, tm, xd, ud, sample_actions, zd, vd, ad, bd, sad, K, i)

		# descend and project
		xd, ud = project(egpm, tm, K, xd, ud, zd, vd, step_size)

		x = [xd[i][1] for i in 1:length(xd)]
		y = [xd[i][2] for i in 1:length(xd)]

		if i % max_iters == 0 && length(sample_actions) > 0
			# NOTE: Just do one call to optimize drill points at the end (can still reason about adding drills along the way regardless of when they are optimized)
			f = X -> optim_total_score(egpm, tm, xd, ud, X)
			g = SA -> Zygote.gradient(X -> optim_total_score(egpm, tm, xd, ud, X), SA)[1]
			lower = zeros(length(sample_actions))
			upper = 1.1 .* ones(length(sample_actions))

			X_optim = try
				optimize(f, g, lower, upper, sample_actions, Fminbox(LBFGS()), inplace = false, Optim.Options(time_limit = optim_time_limit, iterations=optim_iterations, f_calls_limit=optim_f_calls_limit, g_calls_limit=optim_g_calls_limit))
			catch
				if verbose; println("🚨🚨🚨Optim.jl Exception Caught🚨🚨🚨"); end
			end
			optim_sample_actions = try 
				X_optim.minimizer 
			catch 
			end

			if optim_sample_actions != nothing
				sample_actions_optimized = true
				if verbose; println("Optimized✅"); end
				if verbose; @show X_optim; end 
				if verbose; @show optim_sample_actions; end 
				sample_actions = deepcopy(optim_sample_actions)
			end
		end

		# correct for sample actions that are placed in the same location
		if sample_actions != []
			sample_actions_xy = [convert_perc2xy(egpm, tm, sample_actions[i], xd) for i in 1:length(sample_actions)]
			drill_idx = [find_nearest_traj_drill_pts(x, y, sample_actions_xy, i) for i in 1:length(sample_actions_xy)]
			unique_idx = unique(i -> drill_idx[i], 1:length(drill_idx))
			num_sa_removed = length(drill_idx) - length(unique(drill_idx))
			for i = 1:num_sa_removed*sample_cost
				xd = vcat(xd, [xd[end]])
				ud = vcat(ud, [ud[end]])
			end
			sample_actions_xy = sample_actions_xy[unique_idx]
			sample_actions = convert_xy2perc(egpm, tm, sample_actions_xy, xd)
		end
		
		############################
		# Add Sample 10% of the time 
		############################
		if rand(tm.rng) > 0.9
			if verbose; println("👀Looking at adding a new sample👀"); end
			if verbose; println("Length of xd: ", length(xd)); end 
			if length(xd) > (tm.sample_cost+1) # the +1 is because ud is 1 less elemeent than xd
				xd, ud, sample_actions = add_sample(egpm, tm, xd, ud, sample_actions, gp_actual)
			end
		end
		tm.N = length(ud) # should you change this or just not use tm.N and use length(xd) in project??

		# compute statistics and report
		gp_actual = gp_score(egpm, tm, xd, ud, sample_actions)
		es = ergodic_score(egpm, best_xd)
		dd = directional_derivative(ad, bd, zd, vd)
		if verbose; step_report(i, es, xfs, xsa, cs, bs, ts_actual, dd, step_size, length(sample_actions)); end
		if logging; save(outfile, xd); end
		if logging; save(outfile_score_actual, [ts_actual]); end
		if logging; save(outfile_sample_actions, [convert_perc2xy(egpm, tm, sample_actions[i], xd) for i in 1:length(sample_actions)]); end
		if logging; save(outfile_gps_actual, [es]); end
		if logging; save(outfile_xfs, [xfs]); end
		if logging; save(outfile_xsa, [xsa]); end
		if logging; save(outfile_cs, [cs]); end
		if logging; save(outfile_bs, [bs]); end
		if logging; save(outfile_traj_length, [Float64(length(xd))]); end
		if logging; save(outfile_num_samples, [Float64(length(sample_actions))]); end

		# check convergence
		i += 1
		es_count = abs(es - es_prev) < 1e-7 ? es_count + 1 : 0
		es_prev = es
		not_finished = check_convergence(es, es_crit,i,max_iters,dd,dd_crit,verbose, es_count)

		# for storing the best we have seen so far, we should use the ACTUAL not the requested 
		if ts_actual < best_score_actual
			best_score_actual = ts_actual
			best_xd = xd
			best_ud = ud
			best_sample_actions = sample_actions
		end
	end

	# optimize samples at least once 
	if !sample_actions_optimized
		# NOTE: Just do one call to optimize drill points at the end (can still reason about adding drills along the way regardless of when they are optimized)
		f = X -> optim_total_score(egpm, tm, xd, ud, X)
		g = SA -> Zygote.gradient(X -> optim_total_score(egpm, tm, xd, ud, X), SA)[1]
		lower = zeros(length(sample_actions))
		upper = 1.1 .* ones(length(sample_actions))

		X_optim = try
			optimize(f, g, lower, upper, sample_actions, Fminbox(LBFGS()), inplace = false, Optim.Options(time_limit = optim_time_limit, iterations=optim_iterations, f_calls_limit=optim_f_calls_limit, g_calls_limit=optim_g_calls_limit))
		catch
			if verbose; println("🚨🚨🚨Optim.jl Exception Caught🚨🚨🚨"); end
		end
		optim_sample_actions = try 
			X_optim.minimizer 
		catch 
		end

		if optim_sample_actions != nothing
			sample_actions_optimized = true
			if verbose; println("Optimized✅"); end
			if verbose; @show X_optim; end 
			if verbose; @show optim_sample_actions; end 
			sample_actions = deepcopy(optim_sample_actions)
		end
	end

	# now that we are done, print a special finished report
	if verbose
		print_header()
	end

	if logging; close(outfile); end
	if logging; close(outfile_score_actual); end
	if logging; close(outfile_sample_actions); end
	if logging; close(outfile_gps_actual); end
	if logging; close(outfile_xfs); end
	if logging; close(outfile_xsa); end
	if logging; close(outfile_cs); end
	if logging; close(outfile_bs); end
	if logging; close(outfile_traj_length); end
	if logging; close(outfile_num_samples); end

	return xd, ud, sample_actions, best_score_actual, best_xd, best_ud, best_sample_actions#, νₚ_best
end