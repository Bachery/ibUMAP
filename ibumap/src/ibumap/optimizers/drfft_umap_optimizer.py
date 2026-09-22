"""Compatibility module for the historical optimizer module name.

New code should import :mod:`umap_fft.optimizers.ibumap_optimizer`.  Replacing
this module object with the canonical implementation preserves old imports and
also ensures monkeypatches affect the optimizer that the pipelines execute.
"""

from __future__ import annotations

import sys

from . import ibumap_optimizer as _implementation

sys.modules[__name__] = _implementation
