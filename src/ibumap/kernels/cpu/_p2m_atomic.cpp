#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

template <typename T>
void add_atomic(T* address, T value)
{
  std::atomic_ref<T> target(*address);
  target.fetch_add(value, std::memory_order_relaxed);
}

template <typename T>
void accumulate_range(T* mat,
                      const std::int32_t* box,
                      const T* points,
                      Py_ssize_t grid,
                      Py_ssize_t begin,
                      Py_ssize_t end)
{
  const Py_ssize_t term_stride = grid * grid;
  for (Py_ssize_t i = begin; i < end; ++i) {
    const Py_ssize_t x = static_cast<Py_ssize_t>(box[2 * i]);
    const Py_ssize_t y = static_cast<Py_ssize_t>(box[2 * i + 1]);
    const Py_ssize_t cell = x * grid + y;
    add_atomic(mat + cell, points[2 * i]);
    add_atomic(mat + term_stride + cell, points[2 * i + 1]);
    add_atomic(mat + 2 * term_stride + cell, static_cast<T>(1));
  }
}

template <typename T>
bool launch(T* mat,
            const std::int32_t* box,
            const T* points,
            Py_ssize_t grid,
            Py_ssize_t n,
            int requested_threads)
{
  unsigned int hardware = std::thread::hardware_concurrency();
  int threads = requested_threads > 0 ? requested_threads : static_cast<int>(hardware);
  threads = std::max(1, std::min<int>(threads > 0 ? threads : 1, static_cast<int>(n)));
  if (threads == 1 || n < 1024) {
    accumulate_range(mat, box, points, grid, 0, n);
    return true;
  }

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(threads));
  const Py_ssize_t chunk = (n + threads - 1) / threads;
  for (int worker = 0; worker < threads; ++worker) {
    const Py_ssize_t begin = static_cast<Py_ssize_t>(worker) * chunk;
    const Py_ssize_t end = std::min(n, begin + chunk);
    if (begin < end) {
      workers.emplace_back(accumulate_range<T>, mat, box, points, grid, begin, end);
    }
  }
  for (auto& worker : workers) worker.join();
  return true;
}

PyObject* p2m_atomic(PyObject*, PyObject* args)
{
  PyObject* mat_obj = nullptr;
  PyObject* box_obj = nullptr;
  PyObject* points_obj = nullptr;
  int n_threads = 0;
  if (!PyArg_ParseTuple(args, "OOO|i", &mat_obj, &box_obj, &points_obj, &n_threads)) {
    return nullptr;
  }

  Py_buffer mat{};
  Py_buffer box{};
  Py_buffer points{};
  const int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
  if (PyObject_GetBuffer(mat_obj, &mat, flags) < 0) return nullptr;
  if (PyObject_GetBuffer(box_obj, &box, flags) < 0) {
    PyBuffer_Release(&mat);
    return nullptr;
  }
  if (PyObject_GetBuffer(points_obj, &points, flags) < 0) {
    PyBuffer_Release(&box);
    PyBuffer_Release(&mat);
    return nullptr;
  }

  auto release = [&]() {
    PyBuffer_Release(&points);
    PyBuffer_Release(&box);
    PyBuffer_Release(&mat);
  };
  if (mat.ndim != 3 || mat.shape[0] < 3 || mat.shape[1] != mat.shape[2] ||
      box.ndim != 2 || box.shape[1] != 2 || box.itemsize != 4 ||
      points.ndim != 2 || points.shape[1] != 2 ||
      box.shape[0] < points.shape[0] || mat.itemsize != points.itemsize ||
      (mat.itemsize != 4 && mat.itemsize != 8)) {
    release();
    PyErr_SetString(PyExc_ValueError, "invalid contiguous mat_w, box_idx, or Y layout");
    return nullptr;
  }

  bool ok = false;
  PyThreadState* state = PyEval_SaveThread();
  try {
    if (mat.itemsize == 4) {
      ok = launch(
          static_cast<float*>(mat.buf),
          static_cast<const std::int32_t*>(box.buf),
          static_cast<const float*>(points.buf),
          mat.shape[1], points.shape[0], n_threads);
    } else {
      ok = launch(
          static_cast<double*>(mat.buf),
          static_cast<const std::int32_t*>(box.buf),
          static_cast<const double*>(points.buf),
          mat.shape[1], points.shape[0], n_threads);
    }
  } catch (...) {
    ok = false;
  }
  PyEval_RestoreThread(state);
  release();
  if (!ok) {
    PyErr_SetString(PyExc_RuntimeError, "CPU atomic P2M worker failed");
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"p2m_atomic", p2m_atomic, METH_VARARGS, "Accumulate p=1 P2M with CPU atomics."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_p2m_atomic",
    nullptr,
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__p2m_atomic() { return PyModule_Create(&module); }
