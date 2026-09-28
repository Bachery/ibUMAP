from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_FFT_KERNEL_CACHE_MAX_ENTRIES = 4
DEFAULT_CUDA_FFT_KERNEL_CACHE_MAX_ENTRIES = int(
    os.environ.get("IBUMAP_CUDA_FFT_KERNEL_CACHE_MAX_ENTRIES", "32")
)
DEFAULT_FFT_KERNEL_CACHE_LIMIT_BYTES = int(
    os.environ.get("IBUMAP_FFT_KERNEL_CACHE_LIMIT_BYTES", str(1024 * 1024 * 1024))
)
VALID_FFT_KERNEL_CACHE_POLICIES = ("auto", "legacy", "byte_lru", "disabled")


def fft_kernel_cache_key_token(key: Any) -> str:
    """Return a stable short token for diagnostics without serializing the key."""

    data = repr(key).encode("utf-8", errors="replace")
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def estimate_fft_kernel_nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return 0


@dataclass(frozen=True)
class FFTKernelCacheLookup:
    value: Any
    hit: bool
    miss_type: Optional[str]
    key_token: str
    entry_bytes: int


@dataclass(frozen=True)
class FFTKernelCacheStore:
    stored: bool
    entry_bytes: int
    evictions: int
    evicted_bytes: int
    oversized: bool


@dataclass
class _FFTKernelCacheEntry:
    value: Any
    nbytes: int


class FFTKernelLRUCache:
    """LRU cache for ibFFT kernel spectra with exact-key and byte diagnostics."""

    def __init__(
        self,
        *,
        scope: str = "module",
        max_entries: Optional[int] = DEFAULT_FFT_KERNEL_CACHE_MAX_ENTRIES,
        limit_bytes: Optional[int] = None,
        enabled: bool = True,
    ) -> None:
        self.scope = str(scope)
        self.entries: OrderedDict[Any, _FFTKernelCacheEntry] = OrderedDict()
        self.seen_keys: set[Any] = set()
        self.enabled = bool(enabled)
        self.max_entries = self._normalize_max_entries(max_entries)
        self.limit_bytes = self._normalize_limit_bytes(limit_bytes)
        self.current_bytes = 0
        self.peak_bytes = 0
        self.peak_entries = 0

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _normalize_max_entries(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        value = int(value)
        if value < 1:
            raise ValueError("fft kernel cache max_entries must be >= 1 or None")
        return value

    @staticmethod
    def _normalize_limit_bytes(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        value = int(value)
        if value < 0:
            raise ValueError("fft kernel cache limit_bytes must be non-negative")
        return value

    def configure(
        self,
        *,
        policy: str = "auto",
        limit_bytes: Optional[int] = None,
        max_entries: Optional[int] = None,
    ) -> str:
        policy = str(policy)
        if policy not in VALID_FFT_KERNEL_CACHE_POLICIES:
            raise ValueError(
                "fft kernel cache policy must be one of: "
                f"{', '.join(VALID_FFT_KERNEL_CACHE_POLICIES)}"
            )
        if policy == "auto":
            policy = "byte_lru" if self.scope == "workspace" else "legacy"
        self.enabled = policy != "disabled"
        if policy == "legacy":
            resolved_entries = (
                DEFAULT_FFT_KERNEL_CACHE_MAX_ENTRIES
                if max_entries is None
                else max_entries
            )
            self.max_entries = self._normalize_max_entries(resolved_entries)
            self.limit_bytes = self._normalize_limit_bytes(limit_bytes)
        elif policy == "byte_lru":
            self.max_entries = self._normalize_max_entries(max_entries)
            resolved_limit = (
                DEFAULT_FFT_KERNEL_CACHE_LIMIT_BYTES
                if limit_bytes is None
                else limit_bytes
            )
            self.limit_bytes = self._normalize_limit_bytes(resolved_limit)
        else:
            self.max_entries = self._normalize_max_entries(max_entries)
            self.limit_bytes = self._normalize_limit_bytes(limit_bytes)
        if not self.enabled:
            self.entries.clear()
            self.current_bytes = 0
        else:
            self._enforce_limits()
        return policy

    def clear(self) -> None:
        self.entries.clear()
        self.seen_keys.clear()
        self.current_bytes = 0
        self.peak_bytes = 0
        self.peak_entries = 0

    def lookup(self, key: Any) -> FFTKernelCacheLookup:
        token = fft_kernel_cache_key_token(key)
        if not self.enabled:
            return FFTKernelCacheLookup(None, False, "disabled", token, 0)
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            return FFTKernelCacheLookup(entry.value, True, None, token, entry.nbytes)
        was_seen = key in self.seen_keys
        self.seen_keys.add(key)
        return FFTKernelCacheLookup(
            None,
            False,
            "eviction" if was_seen else "compulsory",
            token,
            0,
        )

    def store(self, key: Any, value: Any) -> FFTKernelCacheStore:
        if not self.enabled:
            return FFTKernelCacheStore(False, 0, 0, 0, False)
        nbytes = estimate_fft_kernel_nbytes(value)
        if self.limit_bytes is not None and nbytes > self.limit_bytes:
            old_entry = self.entries.pop(key, None)
            if old_entry is not None:
                self.current_bytes -= old_entry.nbytes
            return FFTKernelCacheStore(False, nbytes, 0, 0, True)

        old_entry = self.entries.pop(key, None)
        if old_entry is not None:
            self.current_bytes -= old_entry.nbytes
        self.entries[key] = _FFTKernelCacheEntry(value, nbytes)
        self.current_bytes += nbytes
        evictions, evicted_bytes = self._enforce_limits()
        self._update_peaks()
        return FFTKernelCacheStore(True, nbytes, evictions, evicted_bytes, False)

    def _enforce_limits(self) -> tuple[int, int]:
        evictions = 0
        evicted_bytes = 0
        while self.entries and self.max_entries is not None and len(self.entries) > self.max_entries:
            _, entry = self.entries.popitem(last=False)
            self.current_bytes -= entry.nbytes
            evictions += 1
            evicted_bytes += entry.nbytes
        while self.entries and self.limit_bytes is not None and self.current_bytes > self.limit_bytes:
            _, entry = self.entries.popitem(last=False)
            self.current_bytes -= entry.nbytes
            evictions += 1
            evicted_bytes += entry.nbytes
        self._update_peaks()
        return evictions, evicted_bytes

    def _update_peaks(self) -> None:
        self.peak_bytes = max(self.peak_bytes, int(self.current_bytes))
        self.peak_entries = max(self.peak_entries, len(self.entries))

    def snapshot(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "enabled": bool(self.enabled),
            "entries": len(self.entries),
            "bytes": int(self.current_bytes),
            "peak_entries": int(self.peak_entries),
            "peak_bytes": int(self.peak_bytes),
            "limit_bytes": (
                None if self.limit_bytes is None else int(self.limit_bytes)
            ),
            "max_entries": (
                None if self.max_entries is None else int(self.max_entries)
            ),
            "seen_exact_keys": len(self.seen_keys),
        }
