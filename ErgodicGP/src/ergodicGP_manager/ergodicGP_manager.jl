######################################################################
# ergodic_manager.jl
######################################################################
@everywhere abstract type ErgodicGPManager end


@everywhere x_min(egpm::ErgodicGPManager) = egpm.domain.mins[1]
@everywhere y_min(egpm::ErgodicGPManager) = egpm.domain.mins[2]
@everywhere z_min(egpm::ErgodicGPManager) = egpm.domain.mins[3]

@everywhere x_max(egpm::ErgodicGPManager) = egpm.domain.maxes[1]
@everywhere y_max(egpm::ErgodicGPManager) = egpm.domain.maxes[2]
@everywhere z_max(egpm::ErgodicGPManager) = egpm.domain.maxes[3]

@everywhere x_size(egpm::ErgodicGPManager) = egpm.domain.cell_lengths[1]
@everywhere y_size(egpm::ErgodicGPManager) = egpm.domain.cell_lengths[2]
@everywhere z_size(egpm::ErgodicGPManager) = egpm.domain.cell_lengths[3]

@everywhere x_cells(egpm::ErgodicGPManager) = egpm.domain.cells[1]
@everywhere y_cells(egpm::ErgodicGPManager) = egpm.domain.cells[2]
@everywhere z_cells(egpm::ErgodicGPManager) = egpm.domain.cells[3]
