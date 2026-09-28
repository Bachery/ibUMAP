import unittest
from unittest.mock import patch

import numpy as np

from ibumap import (
    AttractionScheduleConfig,
    CPUUMAPConfig,
    DegreeDampingConfig,
    DiagnosticsConfig,
    FFTConfig,
    GraphConfig,
    IBUMAPConfig,
    IBUMAPExperimentalConfig,
    InitializationConfig,
    NoiseConfig,
    NumericsConfig,
    RuntimeConfig,
    TFDPConfig,
    UMAPConfig,
    IBUMAP,
)


class ConfigRefactorTest(unittest.TestCase):
    def test_canonical_construction_examples(self):
        self.assertEqual(IBUMAP(algorithm="ibumap").config.runtime.algorithm, "ibumap")
        self.assertEqual(
            IBUMAP(runtime=RuntimeConfig(random_state=42)).config.runtime.random_state,
            42,
        )
        self.assertEqual(IBUMAP().workspace_policy, "performance")
        self.assertEqual(
            IBUMAP(workspace_policy="low_memory").config.runtime.workspace_policy,
            "minimal",
        )
        with self.assertRaisesRegex(ValueError, "workspace_policy"):
            IBUMAP(workspace_policy="compact")
        self.assertEqual(IBUMAP(p2m_mode="segmented").config.fft.p2m_mode, "segmented")
        self.assertEqual(
            IBUMAP(workspace_limit_bytes=1024).config.runtime.workspace_limit_bytes,
            1024,
        )
        with self.assertRaisesRegex(ValueError, "p2m_mode"):
            IBUMAP(deterministic=True, p2m_mode="atomic")
        self.assertEqual(
            IBUMAP(graph=GraphConfig(n_neighbors=30)).config.graph.n_neighbors,
            30,
        )
        self.assertEqual(
            IBUMAP(initialization=InitializationConfig(init="random")).config.initialization.init,
            "random",
        )
        self.assertFalse(
            IBUMAP().config.initialization.report_duplicate_ratio
        )
        self.assertTrue(
            IBUMAP(
                initialization=InitializationConfig(report_duplicate_ratio=True)
            ).config.initialization.report_duplicate_ratio
        )
        self.assertTrue(
            IBUMAP(
                report_duplicate_ratio=True
            ).config.initialization.report_duplicate_ratio
        )
        with self.assertRaisesRegex(ValueError, "report_duplicate_ratio"):
            InitializationConfig(report_duplicate_ratio="yes")
        compact_init = IBUMAP().config.initialization
        self.assertEqual(compact_init.spectral_scale_policy, "auto")
        self.assertEqual(
            compact_init.resolve_spectral_scale_policy("ibumap"),
            "fft_compact",
        )
        self.assertEqual(
            compact_init.resolve_spectral_scale_policy("umap"),
            "legacy_box10",
        )
        explicit_raw = IBUMAP(
            spectral_scale_policy="raw",
            spectral_max_span=0.25,
            spectral_jitter_relative=2e-4,
            spectral_jitter_max=5e-5,
        ).config.initialization
        self.assertEqual(explicit_raw.spectral_scale_policy, "raw")
        self.assertEqual(explicit_raw.spectral_max_span, 0.25)
        self.assertEqual(explicit_raw.spectral_jitter_relative, 2e-4)
        self.assertEqual(explicit_raw.spectral_jitter_max, 5e-5)
        self.assertEqual(
            explicit_raw.resolve_spectral_scale_policy("ibumap"),
            "raw",
        )
        with self.assertRaisesRegex(ValueError, "spectral_scale_policy"):
            InitializationConfig(spectral_scale_policy="box100")
        with self.assertRaisesRegex(ValueError, "spectral_max_span"):
            InitializationConfig(spectral_max_span=0.0)
        with self.assertRaisesRegex(ValueError, "spectral_jitter_relative"):
            InitializationConfig(spectral_jitter_relative=-1.0)
        with self.assertRaisesRegex(ValueError, "spectral_jitter_max"):
            InitializationConfig(spectral_jitter_max=np.inf)
        self.assertEqual(
            IBUMAP(fft=FFTConfig(kernel_clip=6.0)).config.fft.kernel_clip,
            6.0,
        )
        cache_model = IBUMAP(
            fft=FFTConfig(
                kernel_cache_policy="byte_lru",
                kernel_cache_limit_bytes=123456,
                kernel_cache_max_entries=9,
            )
        )
        self.assertEqual(cache_model.config.fft.kernel_cache_policy, "byte_lru")
        self.assertEqual(cache_model.config.fft.kernel_cache_limit_bytes, 123456)
        self.assertEqual(cache_model.config.fft.kernel_cache_max_entries, 9)
        flat_cache_model = IBUMAP(
            fft_kernel_cache_policy="legacy",
            fft_kernel_cache_limit_bytes=654321,
            fft_kernel_cache_max_entries=7,
        )
        self.assertEqual(flat_cache_model.config.fft.kernel_cache_policy, "legacy")
        self.assertEqual(flat_cache_model.config.fft.kernel_cache_limit_bytes, 654321)
        self.assertEqual(flat_cache_model.config.fft.kernel_cache_max_entries, 7)
        self.assertIsInstance(IBUMAP(ibumap=IBUMAPConfig()).config.ibumap, IBUMAPConfig)

        synchronous = IBUMAP(
            algorithm="umap",
            cpu_umap=CPUUMAPConfig(update_mode="synchronous"),
        )
        self.assertEqual(synchronous.config.cpu_umap.update_mode, "synchronous")

        shared_rng = IBUMAP(
            algorithm="umap",
            cpu_umap=CPUUMAPConfig(rng_lifecycle="shared_rng"),
        )
        self.assertEqual(
            shared_rng.config.cpu_umap.rng_lifecycle,
            "shared_rng",
        )

        tfdp = IBUMAP(
            algorithm="tfdp",
            tfdp=TFDPConfig(alpha=0.2),
            fft=FFTConfig(n_interpolation_points=2),
        )
        self.assertEqual(tfdp.config.tfdp.alpha, 0.2)
        self.assertEqual(tfdp.config.fft.n_interpolation_points, 2)

    def test_flat_inputs_map_to_grouped_config(self):
        model = IBUMAP(n_neighbors=30, min_dist=0.1, learning_rate=0.5)
        self.assertEqual(model.config.graph.n_neighbors, 30)
        self.assertEqual(model.config.umap.min_dist, 0.1)
        self.assertEqual(model.config.umap.learning_rate, 0.5)
        self.assertEqual(
            model._legacy_flat_parameters_used,
            ["learning_rate", "min_dist", "n_neighbors"],
        )

    def test_omission_sensitive_and_equal_duplicate_conflicts(self):
        model = IBUMAP(graph=GraphConfig(n_neighbors=30))
        self.assertEqual(model.config.graph.n_neighbors, 30)

        equal = IBUMAP(graph=GraphConfig(n_neighbors=30), n_neighbors=30)
        self.assertEqual(equal.config.graph.n_neighbors, 30)

        with self.assertRaisesRegex(ValueError, "n_neighbors"):
            IBUMAP(graph=GraphConfig(n_neighbors=30), n_neighbors=15)

        with self.assertRaisesRegex(ValueError, "ibfft_kernel_clip"):
            IBUMAP(fft=FFTConfig(kernel_clip=8.0), ibfft_kernel_clip=4.0)

        with self.assertRaisesRegex(ValueError, "rng_lifecycle"):
            IBUMAP(
                algorithm="umap",
                cpu_umap=CPUUMAPConfig(rng_lifecycle="independent"),
                rng_lifecycle="shared_rng",
            )

    def test_nested_legacy_container_conflicts(self):
        with self.assertRaisesRegex(ValueError, "epsilon"):
            IBUMAP(
                umap=UMAPConfig(epsilon=0.01),
                numerics_config=NumericsConfig(epsilon=0.02),
            )

        equal = IBUMAP(
            noise_config=NoiseConfig(mode="force", scale=0.1),
            noise_mode="force",
            noise_scale=0.1,
        )
        self.assertEqual(equal.config.ibumap.experimental.noise.scale, 0.1)

        with self.assertRaisesRegex(ValueError, "noise"):
            IBUMAP(
                noise_config=NoiseConfig(mode="force", scale=0.1),
                noise_scale=0.2,
            )

    def test_attraction_schedule_maps_flat_and_canonical_config(self):
        default_model = IBUMAP()
        default_schedule = (
            default_model.config.ibumap.experimental.attraction_schedule
        )
        self.assertEqual(default_schedule.kernel_mode, "auto")
        self.assertEqual(default_schedule.schedule_mode, "row_scan")
        self.assertEqual(default_schedule.warp_min_degree, 8)
        self.assertEqual(default_model.attraction_kernel_mode, "auto")

        model = IBUMAP(
            attraction_kernel_mode="warp_per_row",
            attraction_schedule_mode="active_edge_calendar",
            attraction_calendar_memory_limit_bytes=4096,
            attraction_warp_min_degree=5,
        )
        schedule = model.config.ibumap.experimental.attraction_schedule
        self.assertEqual(schedule.kernel_mode, "warp_per_row")
        self.assertEqual(schedule.schedule_mode, "active_edge_calendar")
        self.assertEqual(schedule.calendar_memory_limit_bytes, 4096)
        self.assertEqual(schedule.warp_min_degree, 5)
        self.assertEqual(model.attraction_kernel_mode, "warp_per_row")

        canonical = IBUMAP(
            ibumap=IBUMAPConfig(
                experimental=IBUMAPExperimentalConfig(
                    attraction_schedule=AttractionScheduleConfig(
                        kernel_mode="auto",
                        schedule_mode="row_scan",
                        warp_min_degree=3,
                    )
                )
            )
        )
        self.assertEqual(
            canonical.config.ibumap.experimental.attraction_schedule.kernel_mode,
            "auto",
        )

        with self.assertRaisesRegex(ValueError, "attraction_schedule"):
            IBUMAP(
                ibumap=IBUMAPConfig(
                    experimental=IBUMAPExperimentalConfig(
                        attraction_schedule=AttractionScheduleConfig(
                            kernel_mode="thread_per_row"
                        )
                    )
                ),
                attraction_kernel_mode="warp_per_row",
            )

    def test_config_inputs_are_copied(self):
        init = np.arange(12, dtype=np.float32).reshape(6, 2)
        metric_kwds = {"p": 2}
        initialization = InitializationConfig(init=init)
        graph = GraphConfig(metric_kwds=metric_kwds)
        model = IBUMAP(initialization=initialization, graph=graph)

        init[:] = -1
        initialization.init[:] = -2
        metric_kwds["p"] = 1
        graph.metric_kwds["p"] = 3

        np.testing.assert_array_equal(
            model.config.initialization.init,
            np.arange(12, dtype=np.float32).reshape(6, 2),
        )
        self.assertEqual(model.config.graph.metric_kwds, {"p": 2})

    def test_clear_workspace_drops_model_owned_buffers(self):
        model = IBUMAP()
        model._ensure_pipeline(require_graph_backend=False)
        model._pipeline.workspace.get_or_alloc("test", (4, 2))

        model.clear_workspace()

        self.assertEqual(model._pipeline.workspace.buffers, {})

    def test_custom_init_final_shape_is_validated_before_pipeline_work(self):
        model = IBUMAP(initialization=InitializationConfig(init=np.zeros((4, 2))))
        with self.assertRaisesRegex(ValueError, r"init shape must be \(5, 2\)"):
            model.fit(np.zeros((5, 3), dtype=np.float32))

    def test_legacy_umap_init_migrates_to_initialization_config(self):
        model = IBUMAP(umap=UMAPConfig(init="random"))
        self.assertEqual(model.config.initialization.init, "random")
        self.assertEqual(model.init, "random")

        with self.assertRaisesRegex(ValueError, "umap.init"):
            IBUMAP(
                umap=UMAPConfig(init="random"),
                initialization=InitializationConfig(init="spectral"),
            )

    def test_flat_initialization_params_map_to_grouped_config(self):
        model = IBUMAP(
            init="random",
            spectral_method="eigsh",
            spectral_tol=1e-5,
            spectral_maxiter=123,
            spectral_ncv=17,
        )

        cfg = model.config.initialization
        self.assertEqual(cfg.init, "random")
        self.assertEqual(cfg.spectral_method, "eigsh")
        self.assertEqual(cfg.spectral_tol, 1e-5)
        self.assertEqual(cfg.spectral_maxiter, 123)
        self.assertEqual(cfg.spectral_ncv, 17)

    def test_initialization_defaults_preserve_solver_policy(self):
        cfg = IBUMAP().config.initialization

        self.assertEqual(cfg.init, "spectral")
        self.assertIsNone(cfg.spectral_method)
        self.assertIsNone(cfg.spectral_tol)
        self.assertIsNone(cfg.spectral_maxiter)
        self.assertIsNone(cfg.spectral_ncv)

    def test_legacy_initialization_attributes_mutate_grouped_config(self):
        model = IBUMAP()

        model.init = "random"
        model.spectral_ncv = 64

        self.assertEqual(model.config.initialization.init, "random")
        self.assertEqual(model.config.initialization.spectral_ncv, 64)
        self.assertEqual(model.init, "random")
        self.assertEqual(model.spectral_ncv, 64)

    def test_optional_mechanism_none_semantics(self):
        model = IBUMAP(
            repulsion_clip_norm=None,
            total_update_clip_norm=None,
            ibfft_kernel_subsample_mode=None,
            ibfft_kernel_subsample_radius_cells=3,
            local_exact_repulsion=False,
            local_exact_k=11,
        )
        self.assertIsNone(model.config.umap.repulsion_clipping)
        self.assertIsNone(model.config.ibumap.experimental.total_update_clipping)
        self.assertIsNone(model.config.ibumap.experimental.kernel_subsampling)
        self.assertFalse(model.config.ibumap.experimental.local_exact_repulsion.enabled)
        self.assertEqual(model.config.ibumap.experimental.local_exact_repulsion.k, 11)

    def test_degree_damping_auto_resolution_and_runtime_switch(self):
        model = IBUMAP()
        self.assertIsNone(model.config.umap.degree_damping.enabled)
        self.assertTrue(model.attraction_degree_damping)

        model.set_runtime(algorithm="umap")
        self.assertEqual(model.config.runtime.algorithm, "umap")
        self.assertIsNone(model.config.umap.degree_damping.enabled)
        self.assertFalse(model.attraction_degree_damping)

        with self.assertRaisesRegex(ValueError, "degree damping"):
            IBUMAP(
                algorithm="umap",
                umap=UMAPConfig(
                    degree_damping=DegreeDampingConfig(enabled=True)
                ),
            )
        with self.assertRaisesRegex(NotImplementedError, "synchronous"):
            IBUMAP(
                algorithm="umap",
                umap=UMAPConfig(
                    degree_damping=DegreeDampingConfig(enabled=True)
                ),
                cpu_umap=CPUUMAPConfig(update_mode="synchronous"),
            )

    def test_runtime_update_is_transactional(self):
        model = IBUMAP(local_exact_repulsion=True)
        with self.assertRaisesRegex(NotImplementedError, "CPU only"):
            model.set_runtime(device="cuda")
        self.assertEqual(model.config.runtime.device, "cpu")

        umap = IBUMAP(algorithm="umap")
        with self.assertRaisesRegex(ValueError, "degree damping"):
            umap.attraction_degree_damping = True
        self.assertIsNone(umap.config.umap.degree_damping.enabled)

    def test_grouped_effective_report_is_safe_and_versioned(self):
        model = IBUMAP(
            initialization=InitializationConfig(init=np.zeros((4, 2), dtype=np.float32)),
            n_neighbors=15,
        )
        report = model.explain_effective_config()

        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["config"]["runtime"]["algorithm"], "ibumap")
        self.assertEqual(
            report["config"]["initialization"]["init"],
            {"type": "ndarray", "shape": [4, 2], "dtype": "float32"},
        )
        self.assertIsNone(
            report["config"]["umap"]["degree_damping"]["enabled"]
        )
        self.assertTrue(
            report["config"]["umap"]["degree_damping"]["resolved_enabled"]
        )
        self.assertEqual(report["legacy_flat_parameters_used"], ["n_neighbors"])
        self.assertIn("execution_path", report)
        self.assertIn("used_parameters", report)

    def test_memory_diagnostics_config_maps_to_flat_api(self):
        model = IBUMAP(
            diagnostics=DiagnosticsConfig(
                memory_path="memory.jsonl",
                memory_epoch_stride=5,
                memory_sample_interval_ms=25,
            )
        )

        self.assertEqual(model.diagnostics_memory_path, "memory.jsonl")
        self.assertEqual(model.diagnostics_memory_epoch_stride, 5)
        self.assertEqual(model.diagnostics_memory_sample_interval_ms, 25.0)
        report = model.explain_effective_config()
        self.assertTrue(report["diagnostics"]["enabled"])
        self.assertEqual(report["diagnostics"]["memory_path"], "memory.jsonl")
        self.assertEqual(report["diagnostics"]["memory_epoch_stride"], 5)
        self.assertEqual(report["diagnostics"]["memory_sample_interval_ms"], 25.0)

        with self.assertRaisesRegex(ValueError, "memory_epoch_stride"):
            IBUMAP(diagnostics_memory_epoch_stride=0)
        with self.assertRaisesRegex(ValueError, "memory_sample_interval_ms"):
            IBUMAP(diagnostics_memory_sample_interval_ms=0)

    def test_disabled_memory_diagnostics_does_not_create_recorder(self):
        model = IBUMAP(diagnostics_memory_sample_interval_ms=5)

        with patch("ibumap.api.StageMemoryRecorder") as recorder_class:
            self.assertIsNone(model._start_memory_diagnostics("test"))

        recorder_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
