from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse


from common import algorithm_adapters  # noqa: E402
from common.gpu_runtime import device_for_algorithm  # noqa: E402


class _TensorLike:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def detach(self) -> _TensorLike:
        return self

    def cpu(self) -> _TensorLike:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


def _install_fake_torchdr(
    monkeypatch: Any,
    constructor_calls: list[dict[str, Any]],
) -> None:
    module = types.ModuleType("torchdr")

    class UMAP:
        def __init__(self, **kwargs: Any) -> None:
            constructor_calls.append(kwargs)

        def fit_transform(self, X: Any) -> _TensorLike:
            values = np.arange(int(X.shape[0]) * 2, dtype=np.float64).reshape(-1, 2)
            return _TensorLike(np.asfortranarray(values))

    module.UMAP = UMAP
    monkeypatch.setitem(sys.modules, "torchdr", module)


def test_run_torchdr_umap_maps_shared_params_and_synchronizes_cuda(
    monkeypatch: Any,
) -> None:
    constructor_calls: list[dict[str, Any]] = []
    synchronize_calls: list[str] = []
    _install_fake_torchdr(monkeypatch, constructor_calls)
    monkeypatch.setattr(
        algorithm_adapters,
        "synchronize_torch_gpu",
        lambda device=None: synchronize_calls.append(str(device)),
    )

    X = np.zeros((5, 3), dtype=np.float32)
    embedding, extras = algorithm_adapters.run_torchdr_umap(
        X,
        {
            "n_neighbors": 12,
            "n_epochs": 200,
            "learning_rate": 0.5,
            "device": "gpu",
            "backend": "faiss",
            "random_state": 42,
            "repulsion_strength": 2.0,
        },
    )

    assert constructor_calls == [
        {
            "n_neighbors": 12,
            "device": "cuda",
            "backend": "faiss",
            "random_state": 42,
            "repulsion_strength": 2.0,
            "max_iter": 200,
            "lr": 0.5,
        }
    ]
    assert synchronize_calls == ["cuda", "cuda"]
    assert extras["used_params"] == constructor_calls[0]
    assert embedding.shape == (5, 2)
    assert embedding.dtype == np.float32
    assert embedding.flags.c_contiguous


def test_run_torchdr_umap_prefers_native_names_and_defaults_to_faiss_backend(
    monkeypatch: Any,
) -> None:
    constructor_calls: list[dict[str, Any]] = []
    _install_fake_torchdr(monkeypatch, constructor_calls)

    X = np.zeros((4, 3), dtype=np.float32)
    algorithm_adapters.run_torchdr_umap(
        X,
        {
            "n_epochs": 200,
            "max_iter": 300,
            "learning_rate": 0.5,
            "lr": 0.75,
        },
    )

    assert constructor_calls == [{"max_iter": 300, "lr": 0.75, "backend": "faiss"}]


@pytest.mark.parametrize(
    ("deterministic", "expected_fit_state"),
    [
        (True, (False, True, True, ":4096:8")),
        (False, (True, False, False, None)),
    ],
)
def test_run_torchdr_umap_configures_and_restores_torch_execution_mode(
    monkeypatch: Any,
    deterministic: bool,
    expected_fit_state: tuple[bool, bool, bool, str | None],
) -> None:
    constructor_calls: list[dict[str, Any]] = []
    fit_states: list[tuple[bool, bool, bool, str | None]] = []
    algorithms_enabled = not deterministic

    torch_module = types.ModuleType("torch")
    torch_module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(
            benchmark=deterministic,
            deterministic=not deterministic,
        )
    )

    def algorithms_are_enabled() -> bool:
        return algorithms_enabled

    def use_deterministic_algorithms(
        enabled: bool,
        *,
        warn_only: bool = False,
    ) -> None:
        del warn_only
        nonlocal algorithms_enabled
        algorithms_enabled = enabled

    torch_module.are_deterministic_algorithms_enabled = algorithms_are_enabled
    torch_module.is_deterministic_algorithms_warn_only_enabled = lambda: False
    torch_module.use_deterministic_algorithms = use_deterministic_algorithms
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    torchdr_module = types.ModuleType("torchdr")

    class UMAP:
        def __init__(self, **kwargs: Any) -> None:
            constructor_calls.append(kwargs)
            # Mirror torchDR 0.4's seeded constructor, which selects cuDNN's
            # fast settings before the adapter can configure fit execution.
            torch_module.backends.cudnn.benchmark = True
            torch_module.backends.cudnn.deterministic = False

        def fit_transform(self, X: Any) -> _TensorLike:
            del X
            fit_states.append(
                (
                    bool(torch_module.backends.cudnn.benchmark),
                    bool(torch_module.backends.cudnn.deterministic),
                    algorithms_are_enabled(),
                    os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                )
            )
            return _TensorLike(np.zeros((3, 2), dtype=np.float32))

    torchdr_module.UMAP = UMAP
    monkeypatch.setitem(sys.modules, "torchdr", torchdr_module)

    _, extras = algorithm_adapters.run_torchdr_umap(
        np.zeros((3, 4), dtype=np.float32),
        {"deterministic": deterministic},
    )

    assert constructor_calls == [{"backend": "faiss"}]
    assert fit_states == [expected_fit_state]
    assert torch_module.backends.cudnn.benchmark is deterministic
    assert torch_module.backends.cudnn.deterministic is (not deterministic)
    assert algorithms_are_enabled() is (not deterministic)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert extras["deterministic"] is deterministic
    assert extras["torch_runtime"]["deterministic"] is deterministic


def test_run_torchdr_umap_rejects_non_boolean_deterministic(
    monkeypatch: Any,
) -> None:
    constructor_calls: list[dict[str, Any]] = []
    _install_fake_torchdr(monkeypatch, constructor_calls)

    with pytest.raises(TypeError, match="deterministic"):
        algorithm_adapters.run_torchdr_umap(
            np.zeros((3, 4), dtype=np.float32),
            {"deterministic": "true"},
        )

    assert constructor_calls == []


def test_run_algorithm_entry_dispatches_torchdr_family(monkeypatch: Any) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    expected_embedding = np.zeros((3, 2), dtype=np.float32)

    def fake_run(X: Any, params: dict[str, Any]):
        calls.append((X, params))
        return expected_embedding, {"used_params": params}

    monkeypatch.setattr(algorithm_adapters, "run_torchdr_umap", fake_run)
    X = np.zeros((3, 4), dtype=np.float32)
    embedding, extras = algorithm_adapters.run_algorithm_entry(
        X,
        {
            "name": "torchdr_umap_cuda",
            "family": "torchdr_umap",
            "device": "cuda",
            "params": {"random_state": 7},
        },
        common_params={"n_epochs": 100},
    )

    assert calls == [
        (
            X,
            {
                "n_epochs": 100,
                "random_state": 7,
                "device": "cuda",
            },
        )
    ]
    assert embedding is expected_embedding
    assert extras["algorithm_name"] == "torchdr_umap_cuda"
    assert extras["family"] == "torchdr_umap"


def test_device_inference_recognizes_torchdr() -> None:
    assert device_for_algorithm("torchdr_umap_cuda") == "cuda"


def test_torchdr_graph_conversion_packs_uneven_csr_rows() -> None:
    graph = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 0.5, 0.0, 0.0],
                [0.5, 0.0, 0.2, 0.0],
                [0.0, 0.2, 0.0, 0.9],
                [0.0, 0.0, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
    )

    values, indices, metadata = algorithm_adapters._torchdr_graph_to_rowwise(
        graph,
        n_samples=4,
    )

    np.testing.assert_array_equal(
        values,
        np.asarray(
            [
                [0.5, 0.0],
                [0.5, 0.2],
                [0.2, 0.9],
                [0.9, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        indices,
        np.asarray(
            [
                [1, -1],
                [0, 2],
                [1, 3],
                [2, -1],
            ],
            dtype=np.int64,
        ),
    )
    assert metadata == {"n_samples": 4, "nnz": 6, "max_degree": 2}


def test_run_torchdr_optimization_only_injects_graph_and_fixed_init(
    monkeypatch: Any,
) -> None:
    factory_calls: list[dict[str, Any]] = []
    synchronize_calls: list[str] = []
    init = np.asarray(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )

    class Model:
        def fit_transform(self, X: np.ndarray) -> _TensorLike:
            assert X.shape == (3, 1)
            assert X.dtype == np.float32
            return _TensorLike(np.asfortranarray(init + 0.25))

    def fake_factory(
        graph_values: np.ndarray,
        graph_indices: np.ndarray,
        init_embedding: np.ndarray,
        model_kwargs: dict[str, Any],
    ) -> Model:
        factory_calls.append(
            {
                "graph_values": graph_values,
                "graph_indices": graph_indices,
                "init_embedding": init_embedding,
                "model_kwargs": model_kwargs,
            }
        )
        return Model()

    monkeypatch.setattr(
        algorithm_adapters,
        "_make_torchdr_optimization_model",
        fake_factory,
    )
    monkeypatch.setattr(
        algorithm_adapters,
        "synchronize_torch_gpu",
        lambda device=None: synchronize_calls.append(str(device)),
    )

    X = np.zeros((3, 5), dtype=np.float32)
    graph = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 0.8, 0.0],
                [0.8, 0.0, 0.4],
                [0.0, 0.4, 0.0],
            ],
            dtype=np.float32,
        )
    )
    embedding, extras = algorithm_adapters.run_torchdr_umap_optimize_from_graph(
        X,
        graph,
        init,
        {
            "n_epochs": 20,
            "learning_rate": 0.5,
            "repulsion_strength": 2.0,
            "backend": "faiss",
            "distributed": True,
        },
        device="gpu",
    )

    assert len(factory_calls) == 1
    call = factory_calls[0]
    np.testing.assert_array_equal(call["init_embedding"], init)
    assert call["model_kwargs"] == {
        "max_iter": 20,
        "lr": 0.5,
        "repulsion_strength": 2.0,
        "device": "cuda",
        "n_components": 2,
        "backend": None,
        "discard_NNs": False,
        "distributed": False,
    }
    assert call["graph_values"].shape == (3, 2)
    assert call["graph_indices"].shape == (3, 2)
    assert synchronize_calls == ["cuda", "cuda"]
    np.testing.assert_array_equal(embedding, init + 0.25)
    assert embedding.dtype == np.float32
    assert embedding.flags.c_contiguous
    assert extras["fixed_init"] is True
    assert extras["optimization_only"] is True
    assert extras["graph"] == {"n_samples": 3, "nnz": 4, "max_degree": 2}
    assert extras["time_costs"] == {}


def test_run_torchdr_optimization_only_applies_execution_mode(
    monkeypatch: Any,
) -> None:
    mode_calls: list[tuple[str, bool]] = []

    class Model:
        def fit_transform(self, X: np.ndarray) -> _TensorLike:
            assert mode_calls == [("enter", True), ("apply", True)]
            return _TensorLike(np.zeros((3, 2), dtype=np.float32))

    @contextmanager
    def fake_execution_mode(deterministic: bool):
        mode_calls.append(("enter", deterministic))

        def apply_mode() -> dict[str, Any]:
            mode_calls.append(("apply", deterministic))
            return {"deterministic": deterministic}

        yield apply_mode

    monkeypatch.setattr(
        algorithm_adapters,
        "_torch_execution_mode",
        fake_execution_mode,
    )
    monkeypatch.setattr(
        algorithm_adapters,
        "_make_torchdr_optimization_model",
        lambda *args, **kwargs: Model(),
    )
    graph = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 0.8, 0.0],
                [0.8, 0.0, 0.4],
                [0.0, 0.4, 0.0],
            ],
            dtype=np.float32,
        )
    )

    _, extras = algorithm_adapters.run_torchdr_umap_optimize_from_graph(
        np.zeros((3, 4), dtype=np.float32),
        graph,
        np.zeros((3, 2), dtype=np.float32),
        {"random_state": 42, "deterministic": True},
        device="cpu",
    )

    assert mode_calls == [("enter", True), ("apply", True)]
    assert extras["deterministic"] is True
    assert extras["torch_runtime"] == {"deterministic": True}


@pytest.mark.parametrize(
    "runner_name",
    [
        "run_ibumap_optimize_from_graph",
        "run_ibumap_optimize_from_prepared_graph",
    ],
)
@pytest.mark.parametrize(
    ("collect_effective_config", "expected_calls"),
    [
        (True, 1),
        (False, 0),
    ],
)
def test_run_ibumap_optimization_can_skip_effective_config_audit(
    monkeypatch: Any,
    runner_name: str,
    collect_effective_config: bool,
    expected_calls: int,
) -> None:
    module = types.ModuleType("ibumap")
    instances: list[Any] = []

    class IBUMAP:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.explain_calls = 0
            instances.append(self)

        def optimize_from_graph(self, *args: Any) -> np.ndarray:
            # CPU form (graph, init_embedding); CUDA form (X, graph, init_embedding).
            assert len(args) == (3 if self.kwargs["device"] == "cuda" else 2)
            return args[-1] + 0.25

        def optimize_from_prepared_graph(
            self,
            optimizer_graph: Any,
            init_embedding: np.ndarray,
        ) -> np.ndarray:
            del optimizer_graph
            return init_embedding + 0.25

        def get_time_costs(self) -> dict[str, float]:
            return {"optimization": 0.125}

        def explain_effective_config(self) -> dict[str, str]:
            self.explain_calls += 1
            return {"entrypoint": "optimization"}

    module.IBUMAP = IBUMAP
    monkeypatch.setitem(sys.modules, "ibumap", module)

    runner = getattr(algorithm_adapters, runner_name)
    X = np.zeros((3, 4), dtype=np.float32)
    init = np.zeros((3, 2), dtype=np.float32)
    graph = sparse.eye(3, dtype=np.float32, format="csr")
    embedding, extras = runner(
        X,
        graph,
        init,
        {"n_epochs": 2},
        collect_effective_config=collect_effective_config,
    )

    assert len(instances) == 1
    assert instances[0].explain_calls == expected_calls
    assert ("effective_config" in extras) is collect_effective_config
    assert extras["time_costs"] == {"optimization": 0.125}
    np.testing.assert_array_equal(embedding, init + 0.25)


def test_torchdr_optimization_only_cpu_smoke_when_installed() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchdr")

    n_samples = 8
    X = np.arange(n_samples * 3, dtype=np.float32).reshape(n_samples, 3)
    init = np.linspace(-1.0, 1.0, n_samples * 2, dtype=np.float32).reshape(
        n_samples,
        2,
    )
    rows = np.repeat(np.arange(n_samples), 2)
    columns = np.column_stack(
        [
            (np.arange(n_samples) - 1) % n_samples,
            (np.arange(n_samples) + 1) % n_samples,
        ]
    ).reshape(-1)
    graph = sparse.csr_matrix(
        (
            np.full(rows.shape[0], 0.8, dtype=np.float32),
            (rows, columns),
        ),
        shape=(n_samples, n_samples),
    )

    embedding, extras = algorithm_adapters.run_torchdr_umap_optimize_from_graph(
        X,
        graph,
        init,
        {
            "n_neighbors": 2,
            "negative_sample_rate": 1,
            "n_epochs": 2,
            "learning_rate": 0.1,
            "random_state": 0,
        },
        device="cpu",
    )

    assert embedding.shape == init.shape
    assert embedding.dtype == np.float32
    assert np.all(np.isfinite(embedding))
    assert extras["optimization_only"] is True


def test_torchdr_faiss_cuda_fast_and_reproducible_smoke_when_available() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchdr")
    pytest.importorskip("faiss")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    X = np.random.default_rng(20260723).normal(size=(48, 8)).astype(np.float32)
    common = {
        "n_neighbors": 5,
        "n_components": 2,
        "n_epochs": 3,
        "max_iter_affinity": 5,
        "learning_rate": 0.5,
        "negative_sample_rate": 1,
        "init": "normal",
        "device": "cuda",
        "backend": "faiss",
        "distributed": False,
        "compile": False,
        "process_duplicates": False,
    }

    fast, fast_extras = algorithm_adapters.run_torchdr_umap(
        X,
        {
            **common,
            "random_state": None,
            "deterministic": False,
        },
    )
    reproducible_1, reproducible_extras = algorithm_adapters.run_torchdr_umap(
        X,
        {
            **common,
            "random_state": 42,
            "deterministic": True,
        },
    )
    reproducible_2, _ = algorithm_adapters.run_torchdr_umap(
        X,
        {
            **common,
            "random_state": 42,
            "deterministic": True,
        },
    )

    for embedding in (fast, reproducible_1, reproducible_2):
        assert embedding.shape == (48, 2)
        assert embedding.dtype == np.float32
        assert np.all(np.isfinite(embedding))
    np.testing.assert_array_equal(reproducible_1, reproducible_2)
    assert fast_extras["used_params"]["backend"] == "faiss"
    assert fast_extras["torch_runtime"]["deterministic"] is False
    assert reproducible_extras["torch_runtime"]["deterministic"] is True
    assert reproducible_extras["torch_runtime"]["deterministic_algorithms"] is True
