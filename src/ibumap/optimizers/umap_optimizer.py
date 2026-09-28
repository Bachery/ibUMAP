from __future__ import annotations

from time import time

import numba
import numpy as np
from tqdm.auto import tqdm
from umap.layouts import clip, rdist
from umap.utils import tau_rand_int

from .ibumap_optimizer import (
    _clip_force_norm_inplace,
    _resolve_repulsion_clip_norm,
    _validate_clip_epoch_range_for_epochs,
)


def _make_rng_state_per_sample(head_embedding, rng_state):
    """Create umap-learn-compatible persistent RNG state for each head point."""
    return np.full(
        (head_embedding.shape[0], len(rng_state)),
        rng_state,
        dtype=np.int64,
    ) + head_embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)


def optimize_layout_euclidean_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    n_components,
    rng_state_per_sample,
    a,
    b,
    gamma,
    alpha,
    epoch_itr,
    epochs_per_sample,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    move_other,
    epsilon,
    whether_known_points,
    known_points_positions,
    known_points_reverse_index,
    soft_constraint=False,
    constraint_weight=0.1,
    attr_gauss=False,
    repl_gauss=False,
    gauss_sigma=1.0,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= epoch_itr:
            j = head[i]
            k = tail[i]
            j_known = whether_known_points[j]
            k_known = whether_known_points[k]
            if not soft_constraint and j_known and k_known:
                continue

            current = head_embedding[j]
            other = tail_embedding[k]
            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                if attr_gauss:
                    w_ij = np.exp(-dist_squared / (2 * gauss_sigma**2))
                    grad_coeff = (-1.0 / (1.0 - w_ij)) / (gauss_sigma**2)
                else:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(n_components):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                pull_d = 0.0
                if j_known:
                    if soft_constraint:
                        pidx = known_points_reverse_index[j]
                        pull_d = constraint_weight * (
                            known_points_positions[pidx][d] - current[d]
                        )
                        current[d] += (grad_d + pull_d) * alpha
                else:
                    current[d] += grad_d * alpha

                if move_other:
                    if k_known:
                        if soft_constraint:
                            pidx = known_points_reverse_index[k]
                            pull_d = constraint_weight * (
                                known_points_positions[pidx][d] - other[d]
                            )
                            other[d] += (-grad_d + pull_d) * alpha
                    else:
                        other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]
            n_neg_samples = int(
                (epoch_itr - epoch_of_next_negative_sample[i])
                / epochs_per_negative_sample[i]
            )
            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )

            if (not soft_constraint) and j_known:
                continue
            for _ in range(n_neg_samples):
                k = tau_rand_int(rng_state_per_sample[j]) % n_vertices
                if j == k:
                    continue
                other = tail_embedding[k]
                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    if repl_gauss:
                        w_ij = np.exp(-dist_squared / (2 * gauss_sigma**2))
                        grad_coeff = (w_ij / (1.0 - w_ij + epsilon)) / (gauss_sigma**2)
                    else:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (
                            (epsilon + dist_squared) * (a * pow(dist_squared, b) + 1)
                        )
                else:
                    grad_coeff = 0.0

                for d in range(n_components):
                    grad_d = clip(grad_coeff * (current[d] - other[d])) if grad_coeff > 0 else 0.0
                    if j_known:
                        pidx = known_points_reverse_index[j]
                        pull_d = constraint_weight * (
                            known_points_positions[pidx][d] - current[d]
                        )
                        current[d] += (grad_d + pull_d) * alpha
                    else:
                        current[d] += grad_d * alpha


def optimize_layout_euclidean_synchronous_single_epoch(
    head_embedding,
    tail_embedding,
    update_buffer,
    repulsion_buffer,
    head,
    tail,
    n_vertices,
    n_components,
    rng_state_per_sample,
    a,
    b,
    gamma,
    alpha,
    epoch_itr,
    epochs_per_sample,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    move_other,
    epsilon,
    whether_known_points,
    known_points_positions,
    known_points_reverse_index,
    soft_constraint=False,
    constraint_weight=0.1,
    attr_gauss=False,
    repl_gauss=False,
    gauss_sigma=1.0,
):
    # Diagnostic synchronous mode: read a fixed embedding for the whole epoch
    # and accumulate the same sampled UMAP updates before applying them together.
    for i in range(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= epoch_itr:
            j = head[i]
            k = tail[i]
            j_known = whether_known_points[j]
            k_known = whether_known_points[k]
            if not soft_constraint and j_known and k_known:
                continue

            current = head_embedding[j]
            other = tail_embedding[k]
            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                if attr_gauss:
                    w_ij = np.exp(-dist_squared / (2 * gauss_sigma**2))
                    grad_coeff = (-1.0 / (1.0 - w_ij)) / (gauss_sigma**2)
                else:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(n_components):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                pull_d = 0.0
                if j_known:
                    if soft_constraint:
                        pidx = known_points_reverse_index[j]
                        pull_d = constraint_weight * (
                            known_points_positions[pidx][d] - current[d]
                        )
                        update_buffer[j, d] += (grad_d + pull_d) * alpha
                else:
                    update_buffer[j, d] += grad_d * alpha

                if move_other:
                    if k_known:
                        if soft_constraint:
                            pidx = known_points_reverse_index[k]
                            pull_d = constraint_weight * (
                                known_points_positions[pidx][d] - other[d]
                            )
                            update_buffer[k, d] += (-grad_d + pull_d) * alpha
                    else:
                        update_buffer[k, d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]
            n_neg_samples = int(
                (epoch_itr - epoch_of_next_negative_sample[i])
                / epochs_per_negative_sample[i]
            )
            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )

            if (not soft_constraint) and j_known:
                continue
            for _ in range(n_neg_samples):
                k = tau_rand_int(rng_state_per_sample[j]) % n_vertices
                if j == k:
                    continue
                other = tail_embedding[k]
                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    if repl_gauss:
                        w_ij = np.exp(-dist_squared / (2 * gauss_sigma**2))
                        grad_coeff = (w_ij / (1.0 - w_ij + epsilon)) / (gauss_sigma**2)
                    else:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (
                            (epsilon + dist_squared) * (a * pow(dist_squared, b) + 1)
                        )
                else:
                    grad_coeff = 0.0

                for d in range(n_components):
                    grad_d = (
                        clip(grad_coeff * (current[d] - other[d]))
                        if grad_coeff > 0
                        else 0.0
                    )
                    if j_known:
                        pidx = known_points_reverse_index[j]
                        pull_d = constraint_weight * (
                            known_points_positions[pidx][d] - current[d]
                        )
                        update_buffer[j, d] += pull_d * alpha
                        repulsion_buffer[j, d] += grad_d * alpha
                    else:
                        repulsion_buffer[j, d] += grad_d * alpha


def umap_optimize_layout_euclidean(
    embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    n_components,
    rng_state,
    epochs_per_sample,
    epochs_per_negative_sample,
    parallel=False,
    verbose=False,
    tqdm_kwds=None,
    move_other=True,
    fft_params=None,
):
    t1 = time()
    initial_alpha = fft_params["umap_initial_alpha"]
    alpha = initial_alpha
    epoch_of_next_sample = epochs_per_sample.copy()
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    rng_state_per_sample = _make_rng_state_per_sample(embedding, rng_state)

    time_costs = {"opt_prep_time": time() - t1}

    optimize_fn = numba.njit(
        optimize_layout_euclidean_single_epoch,
        fastmath=True,
        parallel=parallel,
    )

    if tqdm_kwds is None:
        tqdm_kwds = {}
    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    t1 = time()
    for epoch_itr in tqdm(range(n_epochs), **tqdm_kwds):
        optimize_fn(
            embedding,
            embedding,
            head,
            tail,
            n_vertices,
            n_components,
            rng_state_per_sample,
            fft_params["umap_a"],
            fft_params["umap_b"],
            fft_params["umap_gamma"],
            alpha,
            epoch_itr,
            epochs_per_sample,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            move_other,
            fft_params["umap_epsilon"],
            fft_params["whether_known_points"],
            fft_params["known_points_positions"],
            fft_params["known_points_reverse_index"],
            fft_params["soft_constraint"],
            fft_params["constraint_weight"],
            fft_params["attr_gauss"],
            fft_params["repl_gauss"],
            fft_params["gauss_sigma"],
        )
        alpha = initial_alpha * (1.0 - (float(epoch_itr) / float(n_epochs)))

    time_costs["optimization_time"] = time() - t1
    time_costs["appl_time"] = 0.0
    return embedding, time_costs


def umap_optimize_layout_euclidean_synchronous(
    embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    n_components,
    rng_state,
    epochs_per_sample,
    epochs_per_negative_sample,
    verbose=False,
    tqdm_kwds=None,
    move_other=True,
    fft_params=None,
):
    """Run diagnostic CPU UMAP with one force-buffer application per epoch."""
    t1 = time()
    initial_alpha = fft_params["umap_initial_alpha"]
    alpha = initial_alpha
    epoch_of_next_sample = epochs_per_sample.copy()
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    rng_state_per_sample = _make_rng_state_per_sample(embedding, rng_state)
    update_buffer = np.zeros_like(embedding)
    repulsion_buffer = np.zeros_like(embedding)
    fft_params["repulsion_clip_epoch_range"] = (
        _validate_clip_epoch_range_for_epochs(
            fft_params.get("repulsion_clip_epoch_range"),
            n_epochs,
            "repulsion_clip_epoch_range",
        )
    )

    time_costs = {"opt_prep_time": time() - t1}
    optimize_fn = numba.njit(
        optimize_layout_euclidean_synchronous_single_epoch,
        fastmath=True,
    )

    if tqdm_kwds is None:
        tqdm_kwds = {}
    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    optimization_start = time()
    application_time = 0.0
    for epoch_itr in tqdm(range(n_epochs), **tqdm_kwds):
        update_buffer.fill(0.0)
        repulsion_buffer.fill(0.0)
        optimize_fn(
            embedding,
            embedding,
            update_buffer,
            repulsion_buffer,
            head,
            tail,
            n_vertices,
            n_components,
            rng_state_per_sample,
            fft_params["umap_a"],
            fft_params["umap_b"],
            fft_params["umap_gamma"],
            alpha,
            epoch_itr,
            epochs_per_sample,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            move_other,
            fft_params["umap_epsilon"],
            fft_params["whether_known_points"],
            fft_params["known_points_positions"],
            fft_params["known_points_reverse_index"],
            fft_params["soft_constraint"],
            fft_params["constraint_weight"],
            fft_params["attr_gauss"],
            fft_params["repl_gauss"],
            fft_params["gauss_sigma"],
        )
        repulsion_clip_norm = _resolve_repulsion_clip_norm(
            fft_params,
            alpha,
            epoch_itr,
        )
        if repulsion_clip_norm is not None:
            _clip_force_norm_inplace(
                repulsion_buffer,
                repulsion_clip_norm,
                False,
            )
        application_start = time()
        update_buffer += repulsion_buffer
        embedding += update_buffer
        application_time += time() - application_start
        alpha = initial_alpha * (1.0 - (float(epoch_itr) / float(n_epochs)))

    time_costs["optimization_time"] = time() - optimization_start
    time_costs["appl_time"] = application_time
    return embedding, time_costs
