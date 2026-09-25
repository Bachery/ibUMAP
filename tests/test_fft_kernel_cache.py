import unittest

import numpy as np

from ibumap._fft_kernel_cache import FFTKernelLRUCache


class FFTKernelLRUCacheTest(unittest.TestCase):
    def test_byte_lru_distinguishes_compulsory_and_eviction_misses(self):
        cache = FFTKernelLRUCache(scope="workspace")
        policy = cache.configure(
            policy="byte_lru",
            limit_bytes=16,
            max_entries=None,
        )
        self.assertEqual(policy, "byte_lru")

        first = cache.lookup(("kernel", 1))
        self.assertFalse(first.hit)
        self.assertEqual(first.miss_type, "compulsory")
        stored = cache.store(("kernel", 1), np.zeros(2, dtype=np.float32))
        self.assertTrue(stored.stored)
        self.assertEqual(cache.snapshot()["bytes"], 8)

        second = cache.lookup(("kernel", 2))
        self.assertFalse(second.hit)
        self.assertEqual(second.miss_type, "compulsory")
        stored = cache.store(("kernel", 2), np.zeros(4, dtype=np.float32))
        self.assertTrue(stored.stored)
        self.assertEqual(stored.evictions, 1)
        self.assertEqual(stored.evicted_bytes, 8)
        self.assertEqual(cache.snapshot()["bytes"], 16)

        first_again = cache.lookup(("kernel", 1))
        self.assertFalse(first_again.hit)
        self.assertEqual(first_again.miss_type, "eviction")

    def test_oversized_entry_is_not_cached(self):
        cache = FFTKernelLRUCache(scope="workspace")
        cache.configure(policy="byte_lru", limit_bytes=4, max_entries=None)

        lookup = cache.lookup(("too", "large"))
        self.assertEqual(lookup.miss_type, "compulsory")
        stored = cache.store(("too", "large"), np.zeros(2, dtype=np.float32))

        self.assertFalse(stored.stored)
        self.assertTrue(stored.oversized)
        self.assertEqual(stored.entry_bytes, 8)
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.snapshot()["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
