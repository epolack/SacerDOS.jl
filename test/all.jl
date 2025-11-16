using Test
using DFTK
using Distributed
using LinearAlgebra

if nworkers() == 1
    addprocs(5)
end
@show nworkers()

@everywhere using SacerDOS

function test_params(case)
    p = get_params(; case)
    g_params = gaussian_parameters(p)
    orders = [0, 1, 2]

    terms_serial = @time dos_gaussian(p.lattice, p.atoms, p.positions, p.Ecut, p.kgrid,
                                      g_params, p.xs; p.disregistries, orders, p.δ)

    terms_distributed = @time SacerDOS.dos_gaussian_distributed(p.lattice, p.atoms, p.positions,
                                                                p.Ecut, p.kgrid, g_params, p.xs;
                                                                p.disregistries, orders, p.δ)

    @test terms_serial.dos ≈ terms_distributed.dos
    @test all(terms_serial.doses .≈ terms_distributed.doses)
end
test_params(:mfast)
