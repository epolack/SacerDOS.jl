# Get all eigensolutions.
function _diagonalize(ham::Hamiltonian)
    kpoints = ham.basis.kpoints
    results = Vector{Any}(undef, length(kpoints))

    for ik in eachindex(kpoints)
        Afull = Hermitian(Array(ham[ik]))
        E = eigen(Afull)
        X = E.vectors
        λ = E.values
        results[ik] = (; λ, X)
    end
    (; λ=[real.(result.λ) for result in results],
       X=[result.X for result in results])
end

function diagonalize(bases; n_bands=false)
    data = map(bases) do basis
        ham   = Hamiltonian(basis)
        if n_bands isa Int
            diagonalize_all_kblocks(diag_full, ham, n_bands)
        else
            _diagonalize(ham)
        end
    end
    (; λ=getfield.(data, :λ), X=getfield.(data, :X))
end
