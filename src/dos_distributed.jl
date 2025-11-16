using Distributed
using JLD2
using Printf
using ProgressMeter

# Distributed equivalent
function dos_gaussian_distributed(lattice, atoms, positions_fn, Ecut, kgrid, params, xs;
                                  disregistries, orders=[0], δ=zero(eltype(xs)),
                                  n_bands_diag=nothing, checkpoint_path=nothing,
                                  cleanup_checkpoint=false)

    n_dis = length(disregistries)
    completed_indices = Set{Int}()
    accumulated_doses = [zeros(length(xs)) for _ in 0:maximum(orders)]

    if checkpoint_path !== nothing && isfile(checkpoint_path)
        @printf "Checkpoint file found. Resuming...\n"
        jldopen(checkpoint_path, "r") do file
            completed_indices = file["completed_indices"]
            accumulated_doses = file["accumulated_doses"]
        end
        @printf "%d / %d jobs already completed.\n" length(completed_indices) n_dis
    end

    indices_to_run = [i for i in 1:n_dis if i ∉ completed_indices]
    n_jobs_to_run = length(indices_to_run)

    if n_jobs_to_run == 0
        @printf "All jobs already completed according to checkpoint.\n"
    else
        jobs_channel = RemoteChannel(() -> Channel{Tuple{Int,Any}}(n_jobs_to_run))
        results_channel = RemoteChannel(() -> Channel{Tuple{Int,Vector{Vector{Float64}}}}(n_jobs_to_run))

        function worker_task(jobs, results)
            while true
                job_index, disregistry = take!(jobs)
                if job_index == -1  # end of work
                    break
                end

                # Single-disregistry computation
                positions = positions_fn(disregistry)
                proxy_pos_fn = _ -> positions
                doses_contrib = _dos_contribution_single_disregistry(
                    disregistry, lattice, atoms, proxy_pos_fn, Ecut, kgrid, params, xs,
                    orders, δ, n_bands_diag
                )

                put!(results, (job_index, doses_contrib))
            end
        end

        for p in workers()
            remote_do(worker_task, p, jobs_channel, results_channel)
        end

        @printf "Dispatching %d jobs to %d workers...\n" n_jobs_to_run nworkers()
        for i in indices_to_run
            put!(jobs_channel, (i, disregistries[i]))
        end

        @showprogress "Jobs Completed" for _ in 1:n_jobs_to_run
            job_index, doses_contrib = take!(results_channel)

            accumulated_doses .+= doses_contrib
            push!(completed_indices, job_index)

            if checkpoint_path !== nothing
                jldopen(checkpoint_path, "w") do file
                    file["completed_indices"] = completed_indices
                    file["accumulated_doses"] = accumulated_doses
                    file["disregistries"] = disregistries # For consistency
                end
            end
        end

        for _ in workers()
            put!(jobs_channel, (-1, nothing))
        end
    end

    # Final normalization.
    n_disregistries = length(disregistries)
    n_kpoints_total = prod(kgrid) * n_disregistries
    normalization = 1 / n_kpoints_total
    final_doses = accumulated_doses .* normalization
    total_dos = sum(final_doses)

    if cleanup_checkpoint && checkpoint_path !== nothing && isfile(checkpoint_path)
        rm(checkpoint_path)
        @printf "Calculation complete. Checkpoint file cleaned up.\n"
    end

    (; dos=total_dos, doses=final_doses)
end
