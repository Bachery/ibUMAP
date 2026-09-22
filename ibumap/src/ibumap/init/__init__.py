from .cpu_init import initialize_embedding_cpu

try:
    from .gpu_init import initialize_embedding_gpu
except Exception:  # pragma: no cover
    initialize_embedding_gpu = None

__all__ = ["initialize_embedding_cpu", "initialize_embedding_gpu"]
