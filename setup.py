from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _split_env(name: str) -> list[str]:
    value = os.environ.get(name, "")
    return [item for item in value.split(os.pathsep) if item]


def _existing_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    return None


def _conda_include_dirs(conda_prefix: str | Path) -> list[str]:
    prefix = Path(conda_prefix)
    prefix_include = prefix / "include"
    include_dirs: list[str] = []

    # Conda's standalone CCCL package installs headers below include/cccl,
    # while the CUDA toolkit also exposes its bundled (and possibly older)
    # CCCL headers below targets/x86_64-linux/include. Put the standalone
    # package first so RAPIDS sees the CCCL version it was built against.
    cccl_include = prefix_include / "cccl"
    if (cccl_include / "cuda" / "version").is_file():
        include_dirs.append(str(cccl_include))

    include_dirs.append(str(prefix_include))
    cuda_target_include = prefix / "targets" / "x86_64-linux" / "include"
    if cuda_target_include.exists():
        include_dirs.append(str(cuda_target_include))
    return include_dirs


def _cuml_libraries() -> list[str]:
    # RAPIDS 26.06's libcuml Conda package exports lib/libcuml.so.
    value = os.environ.get("IBUMAP_CUML_LIBRARIES", "cuml,rmm")
    return [item.strip() for item in value.split(",") if item.strip()]


def _cuml_graph_source() -> str:
    # setuptools 83 rejects absolute Extension source paths when assembling an
    # editable wheel. Builds run from the directory containing setup.py, so
    # keep this path relative to that directory.
    return str(
        Path("src") / "ibumap" / "graph" / "_cuml_graph_ext.pyx"
    )


def _find_nvcc() -> str:
    configured = _existing_path(os.environ.get("IBUMAP_CUML_NVCC"))
    if configured is not None:
        return configured

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    configured = _existing_path(Path(cuda_home) / "bin" / "nvcc") if cuda_home else None
    if configured is not None:
        return configured

    candidate = shutil.which("nvcc")
    if candidate is not None:
        return candidate

    raise RuntimeError(
        "Building the optional cuML graph extension requires nvcc. Set "
        "IBUMAP_CUML_NVCC=/path/to/nvcc if nvcc is not on PATH."
    )


def _find_host_compiler(default: str | None) -> str | None:
    configured = _existing_path(os.environ.get("IBUMAP_CUML_HOST_COMPILER"))
    if configured is not None:
        return configured
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        configured = _existing_path(
            Path(conda_prefix) / "bin" / "x86_64-conda-linux-gnu-g++"
        )
        if configured is not None:
            return configured
    return default


class OptionalCumlGraphBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name != "ibumap.graph._cuml_graph_ext":
            return super().build_extension(ext)
        self._build_cuml_graph_ext_with_nvcc(ext)

    def _build_cuml_graph_ext_with_nvcc(self, ext) -> None:
        nvcc = _find_nvcc()
        default_cxx = None
        if getattr(self.compiler, "compiler_cxx", None):
            default_cxx = self.compiler.compiler_cxx[0]
        host_compiler = _find_host_compiler(default_cxx)

        source = Path(ext.sources[0])
        if source.suffix == ".pyx":
            source = source.with_suffix(".cpp")
        if not source.exists():
            raise RuntimeError(f"Cython output does not exist: {source}")

        output_path = Path(self.get_ext_fullpath(ext.name))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        object_path = Path(self.build_temp) / (ext.name.replace(".", "_") + ".o")
        object_path.parent.mkdir(parents=True, exist_ok=True)

        include_dirs = list(dict.fromkeys([*ext.include_dirs, *self._python_include_dirs()]))
        library_dirs = list(dict.fromkeys(ext.library_dirs))
        runtime_library_dirs = list(dict.fromkeys(ext.runtime_library_dirs or []))

        compile_cmd = [
            nvcc,
            "-x",
            "cu",
            *ext.extra_compile_args,
            "-Xcompiler",
            "-fPIC",
        ]
        if host_compiler:
            compile_cmd.extend(["-ccbin", host_compiler])
        for include_dir in include_dirs:
            compile_cmd.extend(["-I", include_dir])
        compile_cmd.extend(["-c", str(source), "-o", str(object_path)])

        link_cmd = [
            nvcc,
            "-shared",
        ]
        if host_compiler:
            link_cmd.extend(["-ccbin", host_compiler])
        link_cmd.extend(["-Xcompiler", "-fPIC", str(object_path)])
        for library_dir in library_dirs:
            link_cmd.extend(["-L", library_dir])
        for library in ext.libraries:
            link_cmd.append(f"-l{library}")
        for runtime_dir in runtime_library_dirs:
            link_cmd.extend(["-Xlinker", "-rpath", "-Xlinker", runtime_dir])
        link_cmd.extend(ext.extra_link_args)
        link_cmd.extend(["-o", str(output_path)])

        self.announce("building cuML graph extension with nvcc", level=2)
        subprocess.check_call(compile_cmd)
        subprocess.check_call(link_cmd)

    @staticmethod
    def _python_include_dirs() -> list[str]:
        values = [
            sysconfig.get_path("include"),
            sysconfig.get_path("platinclude"),
        ]
        return [str(value) for value in values if value]


def _optional_cuml_graph_extensions():
    if os.environ.get("IBUMAP_BUILD_CUML_GRAPH_EXT") != "1":
        return []

    try:
        from Cython.Build import cythonize
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Building the optional cuML graph extension requires Cython and numpy. "
            "Install them in the CUDA build environment, or unset "
            "IBUMAP_BUILD_CUML_GRAPH_EXT."
        ) from exc

    conda_prefix = os.environ.get("CONDA_PREFIX")

    include_dirs = [np.get_include()]
    library_dirs: list[str] = []
    runtime_library_dirs: list[str] = []

    if conda_prefix:
        include_dirs.extend(_conda_include_dirs(conda_prefix))
        lib_dir = str(Path(conda_prefix) / "lib")
        library_dirs.append(lib_dir)
        runtime_library_dirs.append(lib_dir)

    include_dirs.extend(_split_env("IBUMAP_CUML_INCLUDE_DIRS"))
    library_dirs.extend(_split_env("IBUMAP_CUML_LIBRARY_DIRS"))
    runtime_library_dirs.extend(_split_env("IBUMAP_CUML_RUNTIME_LIBRARY_DIRS"))

    libraries = _cuml_libraries()
    extra_compile_args = [
        "-std=c++17",
        "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
        *_split_env("IBUMAP_CUML_EXTRA_COMPILE_ARGS"),
    ]
    extra_link_args = _split_env("IBUMAP_CUML_EXTRA_LINK_ARGS")

    ext = Extension(
        "ibumap.graph._cuml_graph_ext",
        [_cuml_graph_source()],
        language="c++",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    return cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
    )


def _cpu_atomic_extensions():
    if os.environ.get("IBUMAP_DISABLE_CPU_ATOMIC_EXT") == "1":
        return []
    compile_args = ["/std:c++20"] if sys.platform == "win32" else ["-std=c++20"]
    return [
        Extension(
            "ibumap.kernels.cpu._p2m_atomic",
            [
                str(
                    Path("src")
                    / "ibumap"
                    / "kernels"
                    / "cpu"
                    / "_p2m_atomic.cpp"
                )
            ],
            language="c++",
            extra_compile_args=compile_args,
        )
    ]


setup(
    ext_modules=[*_cpu_atomic_extensions(), *_optional_cuml_graph_extensions()],
    cmdclass={"build_ext": OptionalCumlGraphBuildExt},
)
