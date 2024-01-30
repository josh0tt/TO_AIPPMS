######################################################################
# printing.jl
######################################################################
@everywhere using Printf

@everywhere function print_header()
	print_dashes()
	println(" iter | GP act score | xf score | xsa score | control score | boundary score | total score (act) | direc deriv | step size | Sample Actions |")
	print_dashes()
end
@everywhere function print_dashes()
	println("---------------------------------------------------------------------------------------------------------------------------------------------------")
end

@everywhere function step_report(i::Int, gps_actual::Float64, xfs::Float64, xsa::Float64, cs::Float64, bs::Float64, ts_actual::Float64, dd::Float64, step_size::Float64, n_drills::Int)
	@printf " %-7i" i
	@printf " %-12.7f" gps_actual
	@printf " %-11.7f" xfs
	@printf " %-12.7f" xsa
	@printf " %-16.7f" cs
	@printf " %-17.7f" bs
	@printf " %-16.7f" ts_actual
	@printf " %-14.7f" dd
	@printf " %-14.7f" step_size
	@printf " %-14i" n_drills
	println()
end