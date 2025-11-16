using DFTK

"""
A struct to hold all parameters for a simulation.
"""
struct SimParams{T<:AbstractFloat}
  # Configuration
  case::Symbol
  Ecut::T
  kpoints::UnitRange{Int}
  ε::T
  δ::T
  gaussian_factor::T

  # Physics setup
  lattice::Matrix{T}
  atoms::Vector{ElementGaussian}
  positions::Function

  # Derived parameters
  kgrid::Vector{Int}
  disregistries::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int64}
  xs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int64}
  xlims::Tuple{Int, Int}
end

function gaussian_parameters(p::SimParams)
  (; α=1.0, p.δ, p.gaussian_factor, δx=step(p.xs))
end

function get_params(; case::Symbol=:fast, xlims::Tuple{T, T}=(-35.,35.),
                    gaussian_factor::T=0.08) where {T}
  @show xlims, case, gaussian_factor

  Ecut, kpoints, ε = if case == :full
    (1e4, 1:1000, 0.001)
  elseif case == :fast
    (1e2, 1:2, 0.5)
  elseif case == :mfast
    (1e3, 1:8, 0.2)
  elseif case == :clean
    # Nice zero-order, huge big shift on end of first-order.
    # A bit squiggly with low Gaussian factor but acceptable.
    (1e4, 1:20, 0.01)
  elseif case == :big
    (1e4, 1:100, 0.01)
  elseif case == :huge
    (1e4, 1:500, 0.002)
  elseif case == :huge_ecut
    (5e4, 1:500, 0.002)
  elseif case == :full
    (1e4, 1:1000, 0.001)
  else
    error("Unknown case: $case")
  end

  # Common parameters
  δ = 0.2
  xs = range(xlims...; step=δ/2)
  kgrid = [kpoints[end], 1, 1]
  disregistries = 0:ε:1-ε

  # Model definition
  unit_cell_length = 1
  lattice = T.(unit_cell_length .* [1.0 0 0; 0 0 0; 0 0 0])
  A = T[7.0, 5.0]
  σ = T(0.05) * ones(T, 2)
  atoms = [ElementGaussian(Ai, σi) for (Ai, σi) in zip(A, σ)]
  positions(shift) = [zeros(T, 3), [T(shift), 0.0, 0.0]]

  return SimParams{T}(case, Ecut, kpoints, ε, δ, gaussian_factor,
                      lattice, atoms, positions, kgrid, disregistries,
                      xs, xlims)
end
