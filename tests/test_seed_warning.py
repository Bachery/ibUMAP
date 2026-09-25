"""A seed without deterministic=True is flagged; deterministic runs are not warned about."""
import warnings

import pytest

from ibumap import IBUMAP


def test_seed_without_deterministic_warns():
    with pytest.warns(UserWarning, match="deterministic=True"):
        IBUMAP(random_state=42)


@pytest.mark.parametrize("kwargs", [{}, {"random_state": 42, "deterministic": True}, {"deterministic": True}])
def test_no_warning(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IBUMAP(**kwargs)
