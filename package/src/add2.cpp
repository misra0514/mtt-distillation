#include <torch/extension.h>
// #include <torch/extension.h>


#include "add2.h"

void bind(float hitRatio,  uintptr_t stream_handle,at::Tensor base) {
    launch_add2( hitRatio, stream_handle, base);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bind",
          &bind,
          "kernel warpper");
}
