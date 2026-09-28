"""Protocol tests of the safeguard failure-case experiment (small synthetic inputs, CPU)."""
from __future__ import annotations

import csv
import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = EXPERIMENT_ROOT / "scripts"
# Other experiments also have a scripts/_common.py; import this experiment's copy.
sys.modules.pop("_common", None)
sys.path.insert(0, str(SCRIPTS))

import _common  # noqa: E402


def load_script(name: str):
    """Import scripts/<name>.py against this experiment's _common, whatever other tests loaded."""
    previous = sys.modules.get("_common")
    sys.modules["_common"] = _common
    try:
        spec = importlib.util.spec_from_file_location(f"safeguards_{name}", SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("_common", None)
        else:
            sys.modules["_common"] = previous
    return module


def fixture(n_per_cluster: int = 40, k: int = 8, n_epochs: int = 15):
    """Three blobs, a symmetric kNN graph (canonical float32 CSR), a random initialization."""
    rng = np.random.default_rng(3)
    x = np.concatenate([c + rng.normal(size=(n_per_cluster, 5)) for c in rng.normal(scale=8, size=(3, 5))])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    rows = np.repeat(np.arange(len(x)), k)
    cols = np.argsort(d, axis=1)[:, :k].ravel()
    w = np.exp(-d[rows, cols] / d[rows, cols].mean())
    graph = sparse.csr_matrix((w, (rows, cols)), shape=(len(x), len(x)))
    graph = graph.maximum(graph.T).tocsr().astype(np.float32)
    graph.sum_duplicates()
    graph.sort_indices()
    init = rng.uniform(0, 10, size=(len(x), 2)).astype(np.float32)
    params = dict(_common.load_config()["params"], n_epochs=n_epochs, random_state=42)
    return graph, init, params


def test_bbox_metrics_find_far_points() -> None:
    y = np.random.default_rng(0).normal(size=(1000, 2))
    normal = _common.bbox_metrics(y)
    assert normal["far_count"] == 0 and normal["bbox_area_ratio"] < 2
    y[:5] += 100.0
    escaped = _common.bbox_metrics(y)
    assert escaped["far_count"] == 5 and escaped["far_pct"] == 0.5 and escaped["bbox_area_ratio"] > 100


def test_core_holds_ceil_fraction_of_points() -> None:
    y = np.zeros((101, 2))
    y[:, 0] = np.arange(101)
    mask, radius, core_radius = _common.core_mask(y, 0.99)
    assert mask.sum() == 100 and core_radius == np.sort(radius)[99]


def test_digests_depend_on_content_only(tmp_path) -> None:
    graph = sparse.random(40, 40, density=0.2, format="csr", random_state=0)
    same = sparse.csr_matrix(graph.toarray())
    assert _common.csr_digest(graph) == _common.csr_digest(same)
    changed = graph.copy()
    changed.data[0] += 1
    assert _common.csr_digest(graph) != _common.csr_digest(changed)
    init = np.arange(10, dtype=np.float32).reshape(5, 2)
    np.save(tmp_path / "init.npy", init)
    assert _common.array_digest(np.load(tmp_path / "init.npy")) == _common.array_digest(init)


def test_trace_hook_does_not_change_the_result() -> None:
    from common.algorithm_adapters import run_ibumap_optimize_from_prepared_graph

    run = load_script("01_run_embeddings")
    graph, init, params = fixture()
    for variant in ({}, {"repulsion_clip_norm": None}, {"attraction_degree_damping": False}):
        p = dict(params, **variant)
        plain, _ = run_ibumap_optimize_from_prepared_graph(None, graph.copy(), init.copy(), p)
        traced, rows, _ = run.run_traced(graph, init, p, 0.99, 2.0)
        assert np.array_equal(np.asarray(plain, dtype=np.float32), traced)
        assert [r["epoch"] for r in rows] == list(range(p["n_epochs"]))
        # The first update applies no attraction; with the clip its repulsion is bounded by 4 * alpha0.
        assert rows[0]["attr_norm_max"] == 0.0
        if variant == {}:
            assert rows[0]["repl_norm_max"] <= 4.0 * (1 + 1e-6)
        assert np.isclose(rows[-1]["after_bbox_area_ratio"], _common.bbox_metrics(traced)["bbox_area_ratio"])


def test_grid_rule_matches_the_library(tmp_path) -> None:
    run = load_script("01_run_embeddings")
    graph, init, params = fixture(n_epochs=8)
    diagnostics = tmp_path / "diagnostics.csv"
    _, rows, _ = run.run_traced(graph, init, dict(params, diagnostics_path=str(diagnostics)), 0.99, 2.0)
    with diagnostics.open(newline="") as handle:
        library = list(csv.DictReader(handle))[:len(rows)]
    assert len(library) == len(rows)
    for ours, theirs in zip(rows, library):
        assert ours["before_grid_boxes_per_dim"] == int(float(theirs["n_boxes_per_dim"]))
        assert np.isclose(ours["before_grid_box_width"], float(theirs["grid_cell_size"]), rtol=1e-12)


def test_paper_npz_is_deterministic(tmp_path) -> None:
    from common.paper_data import write_npz

    arrays = {"b__x": np.ones((3, 2), np.float32), "a__y": np.arange(6, dtype=np.float32).reshape(3, 2)}
    first = write_npz(tmp_path / "one.npz", arrays)
    assert first == write_npz(tmp_path / "two.npz", dict(reversed(list(arrays.items()))))
    with zipfile.ZipFile(tmp_path / "one.npz") as archive:
        assert [i.filename for i in archive.infolist()] == ["a__y.npy", "b__x.npy"]
    with np.load(tmp_path / "one.npz") as data:
        assert np.array_equal(data["a__y"], arrays["a__y"])


def test_configuration_matches_the_paper_protocol() -> None:
    cfg = _common.load_config()
    assert cfg["datasets"] == ["cifar10", "scdeed_cart"]
    assert cfg["params"]["repulsion_clip_norm"] == 4.0 and cfg["params"]["attraction_degree_damping"] is True
    assert cfg["variants"] == {"full": {}, "no_repulsion_clip": {"repulsion_clip_norm": None},
                               "no_attraction_damping": {"attraction_degree_damping": False}}
    mechanism = _common.load_config(EXPERIMENT_ROOT.parent / "mechanism" / "configs" / "experiment.yaml")
    for key in ("min_dist", "spread", "learning_rate", "repulsion_strength", "negative_sample_rate", "n_neighbors"):
        assert cfg["params"][key] == mechanism["common_params"][key], key
    assert cfg["params"]["n_epochs"] == mechanism["n_epochs"]


@pytest.mark.parametrize("name", ["02_summarize", "03_reference_bbox", "04_export_paper_data"])
def test_scripts_import(name: str) -> None:
    assert hasattr(load_script(name), "main")
