from pathlib import Path
import os
import runpy
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_setup_namespace():
    with patch("setuptools.setup"):
        return runpy.run_path(str(ROOT / "setup.py"))


class SetupBuildConfigTest(unittest.TestCase):
    def test_standalone_cccl_headers_precede_cuda_toolkit_headers(self):
        include_dirs = _load_setup_namespace()["_conda_include_dirs"]

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            cccl_version = prefix / "include" / "cccl" / "cuda" / "version"
            cccl_version.parent.mkdir(parents=True)
            cccl_version.touch()
            cuda_include = prefix / "targets" / "x86_64-linux" / "include"
            cuda_include.mkdir(parents=True)

            self.assertEqual(
                include_dirs(prefix),
                [
                    str(prefix / "include" / "cccl"),
                    str(prefix / "include"),
                    str(cuda_include),
                ],
            )

    def test_missing_standalone_cccl_directory_is_not_added(self):
        include_dirs = _load_setup_namespace()["_conda_include_dirs"]

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            self.assertEqual(include_dirs(prefix), [str(prefix / "include")])

    def test_default_cuml_libraries_match_rapids_26_06_library_names(self):
        libraries = _load_setup_namespace()["_cuml_libraries"]

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(libraries(), ["cuml", "rmm"])

    def test_cuml_libraries_can_be_overridden(self):
        libraries = _load_setup_namespace()["_cuml_libraries"]

        with patch.dict(
            os.environ,
            {"IBUMAP_CUML_LIBRARIES": " first, second "},
            clear=True,
        ):
            self.assertEqual(libraries(), ["first", "second"])

    def test_cuml_graph_extension_source_is_relative_to_setup_directory(self):
        source = _load_setup_namespace()["_cuml_graph_source"]()

        self.assertFalse(Path(source).is_absolute())
        self.assertEqual(
            Path(source),
            Path("src") / "ibumap" / "graph" / "_cuml_graph_ext.pyx",
        )


if __name__ == "__main__":
    unittest.main()
