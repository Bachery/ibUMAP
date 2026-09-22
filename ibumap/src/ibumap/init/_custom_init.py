from __future__ import annotations

def custom_init_jitter_scale(array_module, init_data):
    """Estimate umap-learn's duplicate-init jitter scale on the active device.

    Exact duplicate rows have zero nearest-neighbor distance. For singleton
    rows, lexicographically adjacent unique rows provide a linear-memory
    nearest-neighbor approximation that works with both NumPy and CuPy.
    """
    xp = array_module
    init_data = xp.asarray(init_data)
    if init_data.ndim != 2:
        raise ValueError("Unsupported init format")
    unique_rows, counts = xp.unique(
        init_data,
        axis=0,
        return_counts=True,
    )
    if unique_rows.shape[0] == init_data.shape[0]:
        return None
    if unique_rows.shape[0] == 1:
        return xp.asarray(0.0, dtype=init_data.dtype)

    adjacent = unique_rows[1:] - unique_rows[:-1]
    adjacent_distances = xp.sqrt(xp.sum(adjacent * adjacent, axis=1))
    nearest = xp.empty(unique_rows.shape[0], dtype=adjacent_distances.dtype)
    nearest[0] = adjacent_distances[0]
    nearest[-1] = adjacent_distances[-1]
    if unique_rows.shape[0] > 2:
        nearest[1:-1] = xp.minimum(
            adjacent_distances[:-1],
            adjacent_distances[1:],
        )

    # Rows in a duplicate group have an exact neighbor at distance zero.
    singleton_distances = xp.where(counts == 1, nearest, 0.0)
    return (
        xp.asarray(0.001, dtype=init_data.dtype)
        * xp.sum(singleton_distances)
        / init_data.shape[0]
    )
