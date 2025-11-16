using SacerDOS
using Distributed
using LinearAlgebra
using JLD2
using Dates

if nworkers() == 1
    addprocs(5)
end

@everywhere using SacerDOS

function run_params(case::Symbol;
                    n_bands_diag::Union{Nothing,Int}=30,
                    xlims::Tuple{T,T}=(5.,20.)) where {T}
    @show case
    @show n_bands_diag

    p = get_params(; case, xlims)
    g_params = gaussian_parameters(p)
    orders = [0, 1, 2]
    outdir = isdir("../../_data") ? "../../_data" : "."

    checkpoint_file = joinpath(outdir, "checkpoint_$(case)_nbands$(n_bands_diag).jld2")

    @info "Starting computation for case: '$case' with n_bands_diag=$n_bands_diag"
    @info "Using checkpoint file: '$checkpoint_file'"

    terms = @time SacerDOS.dos_gaussian_distributed(p.lattice, p.atoms, p.positions,
                                                     p.Ecut, p.kgrid, g_params, p.xs;
                                                     disregistries=p.disregistries,
                                                     orders, δ=p.δ, n_bands_diag,
                                                     checkpoint_path=checkpoint_file,
                                                     cleanup_checkpoint=false)

    @show norm.(terms.doses)
    tr0, tr1, tr2 = terms.doses

    date_str = Dates.format(now(), "yyyymmdd_HH_MM_SS")
    output_file = joinpath(outdir, "computed_$(case)_nbands$(n_bands_diag)_$(date_str).jld2")

    @info "Saving final results to $output_file"
    jldsave(output_file;
            tr0, tr1, tr2, p.δ, p.ε, p.kpoints, p.Ecut, p.xlims, p.xs, p.gaussian_factor, n_bands_diag)

    @info "Calculation complete."
end
