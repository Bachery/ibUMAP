import unittest

from ibumap.kernels.p2m import (
    normalize_workspace_policy,
    resolve_p2m_mode,
)


class P2MResolutionTest(unittest.TestCase):
    def test_workspace_aliases_normalize(self):
        self.assertEqual(normalize_workspace_policy("fast"), "performance")
        self.assertEqual(normalize_workspace_policy("low_memory"), "minimal")

    def test_auto_maps_fast_and_deterministic_profiles(self):
        fast = resolve_p2m_mode(
            "auto",
            deterministic=False,
            device="cuda",
            n_points=100,
            workspace_limit_bytes=None,
        )
        strict = resolve_p2m_mode(
            "auto",
            deterministic=True,
            device="cuda",
            n_points=100,
            workspace_limit_bytes=None,
        )
        self.assertEqual(fast.resolved, "atomic")
        self.assertEqual(strict.resolved, "segmented")

    def test_cpu_auto_preserves_serial_mode(self):
        resolved = resolve_p2m_mode(
            "auto",
            deterministic=True,
            device="cpu",
            n_points=100,
            workspace_limit_bytes=1,
        )
        self.assertEqual(resolved.resolved, "serial")
        self.assertEqual(resolved.reason, "cpu_serial_benchmark_default")

    def test_metal_auto_selects_atomic_mode(self):
        resolved = resolve_p2m_mode(
            "auto",
            deterministic=False,
            device="metal",
            n_points=100,
            workspace_limit_bytes=None,
        )
        self.assertEqual(resolved.resolved, "atomic")
        self.assertEqual(resolved.reason, "maximum_throughput")

    def test_explicit_segmented_rejects_insufficient_budget(self):
        with self.assertRaises(MemoryError):
            resolve_p2m_mode(
                "segmented",
                deterministic=True,
                device="cpu",
                n_points=100,
                workspace_limit_bytes=1,
            )


if __name__ == "__main__":
    unittest.main()
