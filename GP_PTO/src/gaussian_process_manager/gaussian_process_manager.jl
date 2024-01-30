######################################################################
# ergodic_manager.jl
######################################################################
@everywhere abstract type GaussianProcessManager end


@everywhere x_min(gpm::GaussianProcessManager) = gpm.domain.mins[1]
@everywhere y_min(gpm::GaussianProcessManager) = gpm.domain.mins[2]
@everywhere z_min(gpm::GaussianProcessManager) = gpm.domain.mins[3]

@everywhere x_max(gpm::GaussianProcessManager) = gpm.domain.maxes[1]
@everywhere y_max(gpm::GaussianProcessManager) = gpm.domain.maxes[2]
@everywhere z_max(gpm::GaussianProcessManager) = gpm.domain.maxes[3]

@everywhere x_size(gpm::GaussianProcessManager) = gpm.domain.cell_lengths[1]
@everywhere y_size(gpm::GaussianProcessManager) = gpm.domain.cell_lengths[2]
@everywhere z_size(gpm::GaussianProcessManager) = gpm.domain.cell_lengths[3]

@everywhere x_cells(gpm::GaussianProcessManager) = gpm.domain.cells[1]
@everywhere y_cells(gpm::GaussianProcessManager) = gpm.domain.cells[2]
@everywhere z_cells(gpm::GaussianProcessManager) = gpm.domain.cells[3]
