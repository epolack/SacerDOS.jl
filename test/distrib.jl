using Test
using DFTK
using Distributed
using LinearAlgebra

if nworkers() == 1
    addprocs(5)
end
@show nworkers()

@everywhere using SacerDOS

@testset "Distributed Checkpointing" begin
    using Distributed
    using LinearAlgebra
    using JLD2
    using SacerDOS

    if nworkers() == 1
        addprocs(5)
    end
    @everywhere using SacerDOS

    case = :mfast
    p = get_params(; case)
    g_params = gaussian_parameters(p)
    orders = [0, 1, 2]
    checkpoint_file = "dist_checkpoint_test.jld2"

    # Full computation.
    terms_reference = SacerDOS.dos_gaussian_distributed(p.lattice, p.atoms, p.positions,
                                                         p.Ecut, p.kgrid, g_params, p.xs;
                                                         disregistries=p.disregistries, orders,
                                                         δ=p.δ, checkpoint_path=nothing)

    # Simulate interrupted computation.
    disregistries_part1 = p.disregistries[1:3]
    all_positions_part1 = [p.positions(d) for d in disregistries_part1]

    map_fn = dp_tuple -> begin
        disregistry, positions = dp_tuple
        proxy_positions_fn = _ -> positions
        SacerDOS._dos_contribution_single_disregistry(
            disregistry, p.lattice, p.atoms, proxy_positions_fn, p.Ecut, p.kgrid, g_params, p.xs,
            orders, p.δ, nothing
        )
    end

    list_of_doses_part1 = pmap(map_fn, zip(disregistries_part1, all_positions_part1))
    accumulated_doses_partial = reduce(+, list_of_doses_part1)

    jldopen(checkpoint_file, "w") do file
        file["completed_indices"] = [1, 2, 3]
        file["accumulated_doses"] = accumulated_doses_partial
        file["disregistries"] = p.disregistries
    end
    @test isfile(checkpoint_file)

    # Resumed computation.
    terms_resumed = SacerDOS.dos_gaussian_distributed(p.lattice, p.atoms, p.positions,
                                                     p.Ecut, p.kgrid, g_params, p.xs;
                                                     disregistries=p.disregistries, orders,
                                                     δ=p.δ, checkpoint_path=checkpoint_file,
                                                     cleanup_checkpoint=true)

    @test terms_reference.dos ≈ terms_resumed.dos
    @test all(terms_reference.doses .≈ terms_resumed.doses)
    @test !isfile(checkpoint_file)

    if isfile(checkpoint_file) rm(checkpoint_file) end
end
