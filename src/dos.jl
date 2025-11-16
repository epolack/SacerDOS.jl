using JLD2
using LazyStack
using PaddedViews
using Printf

# Gaussian and its derivatives.

function gaussian(α, L, center; gaussian_factor=1.0, δx)
    # Heuristic to have width that make sense.
    # 0.1: sharp, close to χ; 1.0 smooth.
    σ = 0.5 / δx * L * gaussian_factor
    x -> α / (√(2π) * σ) * exp(-((x - center) / σ)^2 / 2)
end
function ∂gaussian(α, L, center; gaussian_factor=1.0, δx)
    σ = 0.5 / δx * L * gaussian_factor
    x -> -α / (√(2π) * σ^3) * (x - center) * exp(-((x - center) / σ)^2 / 2)
end
function ∂²gaussian(α, L, center; gaussian_factor=1.0, δx)
    σ = 0.5 / δx * L * gaussian_factor
    x -> α / (√(2π) * σ^5) * exp(-((x - center) / σ)^2 / 2) * ((x - center)^2 - σ^2)
end
function ∂³gaussian(α, L, center; gaussian_factor=1.0, δx)
    σ = 0.5 / δx * L * gaussian_factor
    x -> -α / (√(2π) * σ^7) * exp(-((x - center) / σ)^2 / 2) * (x - center) * ((x - center)^2 - 3σ^2)
end
function ∂⁴gaussian(α, L, center; gaussian_factor=1.0, δx)
    σ = 0.5 / δx * L * gaussian_factor
    x -> α / (√(2π) * σ^9) * exp(-((x - center) / σ)^2 / 2) * ((x - center)^4 - 6 * σ^2 * (x - center)^2 + 3σ^4)
end

# In-place fn_m functions.

function f2_2!(result, λi, λj, δ, i, j, gaussians, ∂gaussians, ∂²gaussians)
    λij = λj - λi
    if norm(λij) > δ
        @. result = 2 * (gaussians[j] - gaussians[i] - ∂gaussians[i] * λij) / λij^2
    else
        # Use derivative formula.
        copy!(result, ∂²gaussians[i])
    end
end

function f3_3!(result, r1, r2, λi, λj, λk, δ, i, j, k,
               gaussians, ∂gaussians, ∂²gaussians, ∂³gaussians)
    λjk = λk - λj
    λij = λj - λi
    if norm(λjk) > δ
        f2_2!(r1, λi, λk, δ, i, k, gaussians, ∂gaussians, ∂²gaussians)
        f2_2!(r2, λi, λj, δ, i, j, gaussians, ∂gaussians, ∂²gaussians)
        @. result = 3 * (r1 - r2) / λjk
    elseif norm(λij) > δ
        @. result = -12 * (gaussians[j] - gaussians[i] - 0.5 * (∂gaussians[j] + ∂gaussians[i]) * λij) / λij^3
    else
        copy!(result, ∂³gaussians[i])
    end
end

function f4_4!(result, r1, r11, r12, r2, r21, r22, λi, λj, λk, λl, δ, i, j, k, l,
               gaussians, ∂gaussians, ∂²gaussians, ∂³gaussians, ∂⁴gaussians)
    λkl = λl - λk
    if norm(λkl) > δ
        f3_3!(r1, r11, r12, λi, λj, λl, δ, i, j, l, gaussians, ∂gaussians, ∂²gaussians, ∂³gaussians)
        f3_3!(r2, r21, r22, λi, λj, λk, δ, i, j, k, gaussians, ∂gaussians, ∂²gaussians, ∂³gaussians)
        @. result = 4 * (r1 - r2) / λkl
        return nothing
    end
    λik = λk - λi
    λjk = λk - λj
    if norm(λjk) > δ && norm(λkl) < δ && norm(λik) > δ
        f2_2!(r1, λi, λk, δ, i, k, gaussians, ∂gaussians, ∂²gaussians)
        f2_2!(r2, λi, λj, δ, i, j, gaussians, ∂gaussians, ∂²gaussians)
        @. result = -48 * ((gaussians[k] - gaussians[i] - 0.5 * (∂gaussians[k] + ∂gaussians[i]) * (λk - λi)) / (λjk * (λk - λi)^3)) - 12 * ((r1 - r2) / (λk - λj)^2)
        return nothing
    end
    λjl = λl - λj
    if norm(λjl) > δ && norm(λik) < δ && norm(λkl) < δ
        f2_2!(r1, λi, λj, δ, i, j, gaussians, ∂gaussians, ∂²gaussians)
        @. result = 4 * ∂³gaussians[i] / (λi - λj) - 12 * (∂²gaussians[i] - r1) / (λi - λj)^2
        return nothing
    end
    λij = λj - λi
    if norm(λij) > δ && norm(λjk) < δ && norm(λkl) < δ
        @. result = 72 * (gaussians[j] - gaussians[i] - 0.5 * (∂gaussians[j] + ∂gaussians[i]) * λij) / λij^4 - 12 * (∂gaussians[j] - ∂gaussians[i] - ∂²gaussians[j] * λij) / λij^3
        return nothing
    end
    if norm(λij) < δ && norm(λjk) < δ && norm(λkl) < δ
        copy!(result, ∂⁴gaussians[i])
        return nothing
    end
end

function _precompute_operator_matrices(basis, kpt, ψkX, ∂v2, ∂²v2)
    n_bands = size(ψkX, 2)
    T = eltype(ψkX)
    K = zeros(T, n_bands, n_bands)
    X = zeros(T, n_bands, n_bands)
    X_sq = zeros(T, n_bands, n_bands)

    # Pre-allocate buffers for FFTs to reduce memory allocations inside the loop.
    ui = zeros(T, basis.fft_size)
    uj = zeros(T, basis.fft_size)

    for i in 1:n_bands
        for j in 1:n_bands
            K[i, j] = kinetic1d(basis, ψkX, kpt, i, j)
            X[i, j] = potential1d(basis, ψkX, kpt, ∂v2, i, j; ui, uj)
            X_sq[i, j] = potential1d(basis, ψkX, kpt, ∂²v2, i, j; ui, uj)
        end
    end
    (; K, X, X_sq)
end

function calculate_order2(λs, K, X, X_sq, gaussians_info, δ)
    n_bands = length(λs)
    dos_order2 = zeros(eltype(first(gaussians_info.gaussians)), length(first(gaussians_info.gaussians)))

    (; gaussians, ∂gaussians, ∂²gaussians, ∂³gaussians, ∂⁴gaussians) = gaussians_info

    # Temporary arrays for in-place fm_n function.
    tmp_f2_r = similar(dos_order2)
    tmp_f3_r1 = similar(dos_order2)
    tmp_f3_r2 = similar(dos_order2)
    tmp_f3_r11 = similar(dos_order2)
    tmp_f3_r12 = similar(dos_order2)
    tmp_f3_r21 = similar(dos_order2)
    tmp_f3_r22 = similar(dos_order2)
    tmp_f4_r = similar(dos_order2)
    tmp_f4_r1 = similar(dos_order2)
    tmp_f4_r11 = similar(dos_order2)
    tmp_f4_r12 = similar(dos_order2)
    tmp_f4_r2 = similar(dos_order2)
    tmp_f4_r21 = similar(dos_order2)
    tmp_f4_r22 = similar(dos_order2)

    for m in 1:n_bands
        λm = λs[m]
        dos_order2 .-= real(∂²gaussians[m] .* X_sq[m, m] ./ 8)

        for n in 1:n_bands
            λn = λs[n]

            f3_3!(tmp_f3_r1, tmp_f3_r11, tmp_f3_r12, λm, λm, λn, δ, m, m, n,
                  gaussians, ∂gaussians, ∂²gaussians, ∂³gaussians)
            f3_3!(tmp_f3_r2, tmp_f3_r21, tmp_f3_r22, λm, λn, λn, δ, m, n, n
                  , gaussians, ∂gaussians, ∂²gaussians, ∂³gaussians)

            dos_order2 .-= (1/(4*6)) .* (2 .* tmp_f3_r1 .- tmp_f3_r2) .* abs2(X[m, n])

            for p in 1:n_bands
                λp = λs[p]

                f3_3!(tmp_f3_r1, tmp_f3_r11, tmp_f3_r12, λm, λn, λp, δ, m, n, p,
                      gaussians, ∂gaussians, ∂²gaussians, ∂³gaussians)
                term3_expr = real(2 * K[n, p] * X_sq[m, n] * K[p, m] - K[m, n] * X_sq[n, p] * K[p, m])
                dos_order2 .-= (1/(4*6)) .* tmp_f3_r1 .* term3_expr

                for q in 1:n_bands
                    λq = λs[q]

                    f4_4!(tmp_f4_r, tmp_f4_r1, tmp_f4_r11, tmp_f4_r12, tmp_f4_r2, tmp_f4_r21,
                          tmp_f4_r22, λm, λn, λp, λq, δ, m, n, p, q,
                          gaussians, ∂gaussians, ∂²gaussians, ∂³gaussians, ∂⁴gaussians)

                    K_mn = K[m, n]; K_np = K[n, p]; K_pq = K[p, q]; K_qm = K[q, m]
                    X_mn = X[m, n]; X_np = X[n, p]; X_pq = X[p, q]; X_qm = X[q, m]

                    term4_expr = real(
                                (X_mn * K_np) * (K_pq * X_qm) +
                                (X_mn * K_pq) * (K_np * X_qm) +
                                (K_mn * X_pq) * (X_np * K_qm) +
                                (K_mn * X_np) * (X_pq * K_qm) -
                                2 * real( (K_mn * X_np) * (K_pq * X_qm) ) -
                                2 * real( (X_mn * K_pq) * (X_np * K_qm) ) +
                                2 * real( (X_mn * K_qm) * (K_np * X_pq - X_np * K_pq) )
                            )
                    dos_order2 .+= (1/(4*4*6)) .* tmp_f4_r .* term4_expr
                end
            end
        end
    end
    dos_order2
end

function _dos_contribution_single_disregistry(disregistry, lattice, atoms, positions_fn,
                                               Ecut, kgrid, params, xs, orders, δ, n_bands_diag)
    doses = [zeros(length(xs)) for _ in 0:maximum(orders)]

    gaussian_fns = (
        λ -> [gaussian(params.α, params.δ, ix; params.gaussian_factor, params.δx)(λ) for ix in xs],
        λ -> [∂gaussian(params.α, params.δ, ix; params.gaussian_factor, params.δx)(λ) for ix in xs],
        λ -> [∂²gaussian(params.α, params.δ, ix; params.gaussian_factor, params.δx)(λ) for ix in xs],
        λ -> [∂³gaussian(params.α, params.δ, ix; params.gaussian_factor, params.δx)(λ) for ix in xs],
        λ -> [∂⁴gaussian(params.α, params.δ, ix; params.gaussian_factor, params.δx)(λ) for ix in xs],
    )

    model = Model(lattice, atoms, positions_fn(disregistry);
                  terms=[Kinetic(), AtomicLocal()], n_electrons=length(atoms))
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    data = diagonalize([basis]; n_bands=n_bands_diag)

    ∂v2 = compute_δv2(basis)
    ∂²v2 = compute_δ²v2(basis)

    for ik in eachindex(basis.kpoints)
        kpt = basis.kpoints[ik]
        ψkX = data.X[1][ik]
        λs = data.λ[1][ik]
        n_bands = length(λs)

        gaussians_info = (;
            gaussians   = [gaussian_fns[1](λ) for λ in λs],
            ∂gaussians  = [gaussian_fns[2](λ) for λ in λs],
            ∂²gaussians = [gaussian_fns[3](λ) for λ in λs],
            ∂³gaussians = [gaussian_fns[4](λ) for λ in λs],
            ∂⁴gaussians = [gaussian_fns[5](λ) for λ in λs]
        )

        if 0 ∈ orders
            doses[1] .+= sum(gaussians_info.gaussians)
        end

        if any(o -> o > 0, orders)
            mats = _precompute_operator_matrices(basis, kpt, ψkX, ∂v2, ∂²v2)

            tmp_f2_order1 = similar(first(gaussians_info.gaussians))

            if 1 ∈ orders
                dos_order1_k = zeros(eltype(doses[2]), length(xs))
                for i in 1:n_bands
                    for j in 1:n_bands
                        term = imag(mats.K[i, j] * mats.X[j, i])
                        f2_2!(tmp_f2_order1, λs[i], λs[j], δ, i, j,
                              gaussians_info.gaussians, gaussians_info.∂gaussians, gaussians_info.∂²gaussians)
                        dos_order1_k .-= 0.5 .* term .* tmp_f2_order1
                    end
                end
                doses[2] .+= dos_order1_k
            end

            if 2 ∈ orders
                dos_order2_k = calculate_order2(λs, mats.K, mats.X, mats.X_sq, gaussians_info, δ)
                doses[3] .+= dos_order2_k
            end
        end
    end
    doses
end


# One disregistry at the time.
function dos_gaussian(lattice, atoms, positions_fn, Ecut, kgrid, params, xs;
                      disregistries, orders=[0], δ=zero(eltype(xs)), n_bands_diag=nothing,
                      checkpoint_path=nothing, cleanup_checkpoint=false)

    start_index = 1
    total_doses = [zeros(length(xs)) for _ in 0:maximum(orders)]

    if checkpoint_path !== nothing && isfile(checkpoint_path)
        @printf "Checkpoint file found at %s. Resuming calculation.\n" checkpoint_path
        jldopen(checkpoint_path, "r") do file
            start_index = file["last_completed_disregistry"] + 1
            total_doses = file["total_doses"]
        end
    end

    if start_index <= length(disregistries)
        for i in start_index:length(disregistries)
            disregistry = disregistries[i]

            @printf "Calculating for disregistry %d/%d (value: %.4f)\n" i length(disregistries) disregistry

            doses_contrib = _dos_contribution_single_disregistry(disregistry, lattice, atoms, positions_fn,
                                                                 Ecut, kgrid, params, xs, orders, δ, n_bands_diag)
            total_doses .+= doses_contrib

            # Save checkpoint after each step
            if checkpoint_path !== nothing
                jldopen(checkpoint_path, "w") do file
                    file["last_completed_disregistry"] = i
                    file["total_doses"] = total_doses
                    file["disregistries"] = disregistries  # for consistency checks
                end
            end
        end
    else
        @printf "Calculation already completed according to checkpoint.\n"
    end

    # Final normalization.
    n_disregistries = length(disregistries)
    n_kpoints_per_disregistry = prod(kgrid)
    normalization = 1 / (n_kpoints_per_disregistry * n_disregistries)

    final_doses = total_doses .* normalization

    total_dos = sum(final_doses)

    if cleanup_checkpoint && checkpoint_path !== nothing && isfile(checkpoint_path)
        rm(checkpoint_path)
        @printf "Calculation complete. Checkpoint file cleaned up.\n"
    end

    (; dos=total_dos, doses=final_doses)
end
