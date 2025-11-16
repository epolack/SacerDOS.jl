using LazyStack
using PaddedViews
using SparseArrays

@views function eigvals_binning(λs; digits::Int=2, δ=10.0^(-digits))
    # Minimum number of eigenvalues; to be consistent for the all grid.
    N_max = maximum([maximum(length.(λd)) for λd in λs])
    λs_restr = lazystack(PaddedView.(NaN, lazystack(λs), Ref((N_max,))))

    λs_min, λs_max = extrema(x->isnan(x) ? λs_restr[1] : x, λs_restr)

    scale = ceil(λs_min-δ; digits):δ:ceil(λs_max+δ; digits)
    bin = spzeros(Int, length(scale))
    for λ in λs_restr
        index = findfirst(x -> x ≥ λ, scale)
        bin[index] += 1
    end
    (; bin1d=bin, scale, λs=λs_restr, δ)
end

# Used to consume too much memory before views.
@views function eigvals_binning_3d(λs; digits::Int=2, δ=10.0^(-digits))
    #N_max = maximum([maximum(length.(λd)) for λd in λs])
    # Minimum number of eigenvalues; to be consistent for the all grid.
    N_max = minimum([minimum(length.(λd)) for λd in λs])
    λs_restr = lazystack(PaddedView.(NaN, lazystack(λs), Ref((N_max,))))

    λs_min, λs_max = extrema(x->isnan(x) ? λs_restr[1] : x, λs_restr)

    scale = ceil(λs_min-δ; digits):δ:ceil(λs_max+δ; digits)
    bin = spzeros(Int, length(scale)*size(λs_restr, 2)*size(λs_restr, 3))
    bin = reshape(bin, length(scale), size(λs_restr, 2), size(λs_restr, 3))
    bin = zeros(Int, length(scale),size(λs_restr, 2),size(λs_restr, 3))

    for ids in 1:size(λs_restr, 3)
        for idk in 1:size(λs_restr, 2)
            for λ in λs_restr[:, idk, ids]
                index = findfirst(x -> x ≥ λ, scale)
                bin[index, idk, ids] += 1
            end
        end
    end
    (; bin, scale, λs=λs_restr, δ, bin1d=vec(sum(bin; dims=(2,3))))
end

function binning(x, bins)
    ivalue = findfirst(e -> e ≥ x, bins.scale)
    return bins.bin1d[ivalue] ./ bins.δ
end
