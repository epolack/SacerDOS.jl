struct Kinetic1D{T<:Real}
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
end
function Base.Matrix(op::Kinetic1D)
    basis = op.basis
    kpoint = op.kpoint

    n_G = length(G_vectors(basis, kpoint))
    TC = complex(eltype(basis))
    H = zeros(TC, n_G, n_G)
    for (i, Gpk) in enumerate(Gplusk_vectors_cart(basis, kpoint))
        H[i, i] = Gpk[1]
    end
    H
end
function kinetic1d(basis, ukX, kpt, i, j)
    ui = ukX[:, i]
    uj = ukX[:, j]
    Gk_vecs = [Gk[1] for Gk in G_vectors_cart(basis, kpt)]
    braket = sum(Gk_vecs .* conj(ui) .* uj)
    if i == j
        braket += (basis.model.recip_lattice * kpt.coordinate)[1, 1, 1]
    end
    braket
end

struct Potential1D{T<:Real, A<:AbstractArray{Complex{T}, 3}}
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    # Specific storage
    potential::A
end
function compute_v2_four(basis::PlaneWaveBasis{T}; positions=basis.model.positions,
                         q=zero(Vec3{T})) where {T}
    # pot_fourier is <e_G|V|e_G'> expanded in a basis of e_{G-G'}
    # Since V is a sum of radial functions located at atomic
    # positions, this involves a form factor (`local_potential_fourier`)
    # and a structure factor e^{-i G·r}
    model = basis.model
    Gqs_cart = [model.recip_lattice * (G + q) for G in G_vectors(basis)]
    # TODO Bring Gqs_cart on the CPU for compatibility with the pseudopotentials which
    #      are not isbits ... might be able to solve this by restructuring the loop
    disregistry_positions = [[i] for i in length(positions)÷2+1:length(positions)]

    # Pre-compute the form factors at unique values of |G| to speed up
    # the potential Fourier transform (by a lot). Using a hash map gives O(1)
    # lookup.
    form_factors = IdDict{Tuple{Int,T},T}()  # IdDict for Dual compatibility
    for G in Gqs_cart
        p = norm(G)
        for (igroup, group) in enumerate(disregistry_positions)
            if !haskey(form_factors, (igroup, p))
                element = model.atoms[first(group)]
                form_factors[(igroup, p)] = DFTK.local_potential_fourier(element, p)
            end
        end
    end

    Gqs = [G + q for G in G_vectors(basis)]  # TODO Again for GPU compatibility
    pot_fourier_1D = map(enumerate(Gqs)) do (iG, G)
        p = norm(Gqs_cart[iG])
        pot = sum(enumerate(disregistry_positions)) do (igroup, group)
            structure_factor = sum(r -> DFTK.cis2pi(-dot(G, r)), @view positions[group])
            form_factors[(igroup, p)] * structure_factor
        end
        pot / sqrt(model.unit_cell_volume)
    end
    # Explicitly reshape the 1D vector to the 3D grid size required by DFTK's FFT routines.
    return reshape(pot_fourier_1D, basis.fft_size)
end
function Potential1D(basis::PlaneWaveBasis{T}, kpoint::Kpoint{T}, ε::T, basis_fn::Function) where {T}
    basis = basis_fn(ε)
    V2_four = compute_v2_four(basis)
    prefac_m = map(G_vectors(basis)) do G
        -2π * im * G[1]
    end
    V2_four_m = zeros(complex(T), size(V2_four))
    for i in axes(V2_four, 1)
        V2_four_m[i, :, :] = V2_four[i, :, :] .* prefac_m[i, :, :]
    end
    potential = ifft(basis, V2_four_m)

    Potential1D(basis, kpoint, potential)
end
function Base.Matrix(op::Potential1D{T}) where {T}
    # V(G, G') = <eG|V|eG'> = 1/sqrt(Ω) <e_{G-G'}|V>
    pot_fourier = fft(op.basis, op.potential)
    n_G = length(G_vectors(op.basis, op.kpoint))
    H = zeros(complex(eltype(op.basis)), n_G, n_G)
    for (j, G′) in enumerate(G_vectors(op.basis, op.kpoint))
        for (i, G) in enumerate(G_vectors(op.basis, op.kpoint))
            # G_vectors(basis)[ind_ΔG] = G - G'
            ind_ΔG = DFTK.index_G_vectors(op.basis, G - G′)
            if isnothing(ind_ΔG)
                error("For full matrix construction, the FFT size must be " *
                      "large enough so that Hamiltonian applications are exact")
            end
            H[i, j] = pot_fourier[ind_ΔG] / sqrt(op.basis.model.unit_cell_volume)
        end
    end
    H
end
function compute_δv2(basis::PlaneWaveBasis{T}) where {T}
    V2_four = compute_v2_four(basis)
    prefac_m = map(G -> -2π * im * G[1], G_vectors(basis))
    # Reshape the 1D prefactor to the 3D grid size for consistent broadcasting
    prefac_m_3D = reshape(prefac_m, basis.fft_size)

    # Now this is a correct 3D broadcast operation
    V2_four_m = V2_four .* prefac_m_3D

    return ifft(basis, V2_four_m)[:, 1, 1]
end
function compute_δ²v2(basis::PlaneWaveBasis{T}) where {T}
    V2_four = compute_v2_four(basis)
    prefac_m = map(G -> -2π * im * G[1], G_vectors(basis))
    # Reshape the 1D prefactor to the 3D grid size for consistent broadcasting
    prefac_m_3D = reshape(prefac_m, basis.fft_size)

    # Now this is a correct 3D broadcast operation
    V2_four_m = V2_four .* prefac_m_3D .* prefac_m_3D

    return ifft(basis, V2_four_m)[:, 1, 1]
end
function potential1d(basis::PlaneWaveBasis{T}, ukX::AbstractArray, kpt::Kpoint{T}, potential::AbstractArray, i::Int, j::Int; ui::AbstractArray, uj::AbstractArray) where {T}
    ifft!(ui, basis, kpt, ukX[:, i])
    ifft!(uj, basis, kpt, ukX[:, j])
    sum(potential .* conj(ui) .* uj) ./ length(ui)
end
function potential1d_nos(basis::PlaneWaveBasis{T}, ukX::AbstractArray, kpt::Kpoint{T}, potential::AbstractArray, i::Int, j::Int) where {T}
    ui = ifft(basis, kpt, ukX[:, i])
    uj = ifft(basis, kpt, ukX[:, j])
    sum(potential .* conj(ui) .* uj) ./ length(ui)
end
