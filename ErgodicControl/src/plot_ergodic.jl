# using PyPlot
# import PyPlot.plot
using Plots
ENV["GKSwstype"]="nul" 

function plot_erg(em, xd::VVF;
        alpha=1.0,
        cmap="Greys",
        show_score::Bool=true,
        ls::String="-",
        lw::Real=1.5,
        mew::Real=1,
        mfc::String="w",
        ms::Real=10,
        no_domain::Bool=false
    )

    # If it is in R3, let the trajectory know
    dims = 2
    if typeof(em) == ErgodicManagerR3
        dims = 3
    end

    # determines if ergodic score should be shown
    if show_score
        es = ergodic_score(em, xd)
        title_string = "es = $(round(es,digits=5))"
    end

    # plot domain
    plot(em, xd, alpha=alpha, cmap=cmap, no_domain=no_domain, title_string=title_string)
end

function plot(em, xd::VVF; alpha=1.0, cmap="Greys",no_domain=false,title_string="None")
    N = length(xd)
    xvals = zeros(N)
    yvals = zeros(N)
    zvals = zeros(N)
    for i = 1:N
        xvals[i] = xd[i][1]
        yvals[i] = xd[i][2]
    end

    dx = (x_max(em) - x_min(em))/(em.domain.cells[1]-1)
    dy = (y_max(em) - y_min(em))/(em.domain.cells[2]-1)

    contourf(x_min(em):dx:(x_max(em)), y_min(em):dy:(y_max(em)),em.phi', colorbar = true, c = cgrad(:inferno, rev = true),  xlims = (x_min(em), x_max(em)), ylims = (y_min(em), y_max(em)), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim=(0,4))
    # contourf(em.phi, colorbar = true, c = cgrad(:inferno, rev = true), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal)

    scatter!(xvals, yvals, title=title_string)

    # axis(a)
    # tick_params(direction="in")
end

######################################################################
# plot_trajectory.jl
#
# Contains files for plotting the trajectory
######################################################################

function plot_trajectory(xd::VVF;
        c::String="b",
        ls::String="-",
        lw::Real=1.5,
        mew::Real=1,
        mfc::String="w",
        ms::Real=10,
        dims::Int=2,
        title_string="None"
    )

    N = length(xd)
    xvals = zeros(N)
    yvals = zeros(N)
    zvals = zeros(N)
    for i = 1:N
        xvals[i] = xd[i][1]
        yvals[i] = xd[i][2]
    end

    m = "."
    #plot(xvals, yvals, c, linestyle=ls, marker=m, lw=lw, ms=ms, mfc=mfc)
    scatter!(xvals, yvals, title=title_string)

end
