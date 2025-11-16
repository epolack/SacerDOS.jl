module SacerDOS

using DFTK
# TODO: Threads in threads if DFTK @timing switched off by default.
# Needs #972.
disable_threading()
using LinearAlgebra

export SimParams, get_params, gaussian_parameters
include("Parameters.jl")

export diagonalize
include("diag.jl")

export eigvals_binning, eigvals_binning_3d, binning
include("binning.jl")

export Kinetic1D, Potential1D, Matrix, kinetic1d, potential1d
include("operators.jl")

export dos_χ, gaussian, dos_gaussian
include("dos.jl")
include("dos_distributed.jl")

using PrecompileTools

@setup_workload begin
  params = get_params(; case=:mfast)
  g_params = gaussian_parameters(params)
  orders = [0, 1, 2]
  n_bands_diag = 10
  terms = dos_gaussian_distributed(params.lattice, params.atoms, params.positions,
                                   params.Ecut, params.kgrid, g_params, params.xs;
                                   params.disregistries, params.δ, orders, n_bands_diag)
end

end
