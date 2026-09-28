from unittest.mock import Mock, patch

from ibumap.runtime import GPUWorkspace


def test_gpu_workspace_fft_plan_cache_is_bounded_by_role():
    workspace = GPUWorkspace()
    first_forward = object()
    replacement_forward = object()
    inverse = object()

    plan, hit = workspace.get_fft_plan("r2c", ("shape-a",), lambda: first_forward)
    assert plan is first_forward
    assert not hit

    plan, hit = workspace.get_fft_plan(
        "r2c",
        ("shape-a",),
        lambda: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert plan is first_forward
    assert hit

    plan, hit = workspace.get_fft_plan("r2c", ("shape-b",), lambda: replacement_forward)
    assert plan is replacement_forward
    assert not hit
    assert len(workspace.fft_plans) == 1

    workspace.get_fft_plan("c2r", ("shape-b",), lambda: inverse)
    assert workspace.fft_snapshot() == {
        "plan_entries": 2,
        "plan_hits": 1,
        "plan_misses": 3,
    }


def test_gpu_workspace_clear_drops_fft_plans_and_counters():
    workspace = GPUWorkspace()
    workspace.get_fft_plan("r2c", ("shape",), object)
    pool = Mock()

    with patch.object(GPUWorkspace, "memory_pool", return_value=pool):
        workspace.clear()

    assert workspace.fft_snapshot() == {
        "plan_entries": 0,
        "plan_hits": 0,
        "plan_misses": 0,
    }
    pool.free_all_blocks.assert_called_once_with()
