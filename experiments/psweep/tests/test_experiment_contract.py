from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import _common
from ibumap.fft_schedule import resolve_fft_schedule


def load_summary_module():
    path = SCRIPTS_ROOT / "04_summarize_results.py"
    spec = importlib.util.spec_from_file_location("fft_schedule_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plot_module():
    os.environ.setdefault("MPLBACKEND", "Agg")
    path = SCRIPTS_ROOT / "05_plot_results.py"
    spec = importlib.util.spec_from_file_location("fft_schedule_plots", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_dataset_and_algorithm_contract() -> None:
    configs = _common.load_configs()
    datasets = _common.dataset_entries(configs)
    algorithms = _common.algorithm_entries(configs)
    assert _common.validate_static_contract(configs) == []
    assert len(datasets) == 30
    assert max(int(entry["expected_rows"]) for entry in datasets) <= 200_000
    assert len({entry["family"] for entry in datasets}) == 6
    assert {entry["size_bin"] for entry in datasets} == {
        "micro",
        "small",
        "medium",
        "large",
        "xlarge",
    }
    assert {(entry["device"], entry["variant"]) for entry in algorithms} == {
        (device, variant)
        for device in ("cpu", "cuda")
        for variant in ("p1", "p2", "p3", "schedule")
    }
    assert configs["evaluation"]["max_samples"] is None


def test_schedule_is_persistent_and_fixed_variants_are_single_stage() -> None:
    configs = _common.load_configs()
    for entry in _common.algorithm_entries(configs):
        fft = _common.fft_config_for(entry)
        stages = resolve_fft_schedule(
            n_epochs=200,
            n_interpolation_points=fft.n_interpolation_points,
            combine_stages=fft.combine_stages,
            interpolation_schedule=fft.interpolation_schedule,
        )
        if entry["variant"] == "schedule":
            assert [
                (stage.start_epoch, stage.end_epoch, stage.n_interpolation_points)
                for stage in stages
            ] == [(0, 180, 1), (180, 190, 2), (190, 200, 3)]
        else:
            assert len(stages) == 1
            assert stages[0].n_interpolation_points == int(entry["variant"][1:])


def test_task_grid_pairs_repeat_seeds_across_all_variants() -> None:
    configs = _common.load_configs()
    tasks = _common.build_tasks(
        configs,
        _common.dataset_entries(configs),
        _common.algorithm_entries(configs),
    )
    assert len(tasks) == 30 * 8 * 3
    grouped: dict[tuple[str, int], set[int]] = {}
    for task in tasks:
        grouped.setdefault((task.dataset_id, task.repeat), set()).add(task.seed)
    assert all(len(seeds) == 1 for seeds in grouped.values())
    assert {next(iter(seeds)) for seeds in grouped.values()} == {42, 43, 44}


def test_memory_diagnostics_prefers_rss_and_nvml(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    rows = [
        {
            "time_since_start_s": 0.0,
            "cpu": {"rss_bytes": 100},
            "gpu": {
                "nvml_process_used_bytes": 200,
                "cupy_pool_total_bytes": 20,
                "cuda_total_bytes": 1000,
                "cuda_free_bytes": 700,
            },
        },
        {
            "time_since_start_s": 0.1,
            "cpu": {"rss_bytes": 180},
            "gpu": {
                "nvml_process_used_bytes": 360,
                "cupy_pool_total_bytes": 90,
                "cuda_total_bytes": 1000,
                "cuda_free_bytes": 500,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    cpu = _common.parse_memory_diagnostics(path, "cpu")
    cuda = _common.parse_memory_diagnostics(path, "cuda")
    assert cpu["primary_source"] == "cpu.rss_bytes"
    assert cpu["primary_peak_bytes"] == 180
    assert cpu["primary_peak_delta_bytes"] == 80
    assert cuda["primary_source"] == "gpu.nvml_process_used_bytes"
    assert cuda["primary_peak_bytes"] == 360
    assert cuda["primary_peak_delta_bytes"] == 160
    assert cuda["gpu_device_peak_bytes"] == 500


def test_stability_alignment_removes_rotation_scale_and_translation(tmp_path: Path) -> None:
    module = load_summary_module()
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(256, 2)).astype(np.float32)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    candidate = reference @ rotation * 3.5 + np.array([12.0, -4.0], dtype=np.float32)
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(reference_path, reference)
    np.save(candidate_path, candidate)
    result = module.stability_metrics(
        reference_path,
        candidate_path,
        max_points=0,
        pair_samples=2000,
        n_neighbors=15,
        seed=42,
        hashes_equal=False,
    )
    assert result["procrustes_rms"] < 1e-7
    assert result["pairwise_distance_spearman"] > 0.999999
    assert result["neighbor_overlap_at_15"] == 1.0


def test_plot_functions_accept_complete_synthetic_matrix(tmp_path: Path) -> None:
    module = load_plot_module()
    run_rows = []
    quality_rows = []
    stability_rows = []
    metrics = (
        "trustworthiness",
        "continuity",
        "neighborhood_preservation",
    )
    for device_index, device in enumerate(("cpu", "cuda")):
        for variant_index, variant in enumerate(("p1", "p2", "p3", "schedule")):
            base = 1.0 + device_index + variant_index / 10
            run_rows.append(
                {
                    "status": "ok",
                    "dataset": "synthetic",
                    "device": device,
                    "variant": variant,
                    "expected_rows": 1000 + variant_index * 100,
                    "optimization_wall_s": base,
                    "memory_primary_peak_bytes": base * 1024**3,
                }
            )
            for metric_index, metric in enumerate(metrics):
                quality_rows.append(
                    {
                        "status": "ok",
                        "device": device,
                        "variant": variant,
                        "metric": metric,
                        "value": 0.5 + metric_index / 20 + variant_index / 100,
                    }
                )
            stability_rows.append(
                {
                    "status": "ok",
                    "device": device,
                    "variant": variant,
                    "procrustes_rms": 0.01 + variant_index / 100,
                    "pairwise_distance_spearman": 0.99 - variant_index / 100,
                    "neighbor_overlap_at_15": 0.9 - variant_index / 100,
                }
            )
    runs = module.pd.DataFrame(run_rows)
    quality = module.pd.DataFrame(quality_rows)
    stability = module.pd.DataFrame(stability_rows)
    module.scaling_plot(runs, tmp_path, 72)
    module.distribution_plot(runs, tmp_path, 72)
    module.quality_plot(quality, tmp_path, 72)
    module.stability_plot(stability, tmp_path, 72)
    assert (tmp_path / "runtime_memory_scaling.png").exists()
    assert (tmp_path / "runtime_memory_distribution.png").exists()
    assert (tmp_path / "evaluation_quality.png").exists()
    assert (tmp_path / "embedding_stability.png").exists()
