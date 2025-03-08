#include <torch/extension.h>
// #include <torch/extension.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/div_rtn.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/Resize.h>
// #include <ATen/native/cuda/im2col.cuh>
#include <ATen/ATen.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_slow_conv2d_forward_native.h>
#include <ATen/ops/_slow_conv2d_backward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#endif

#include "add2.h"

namespace at { namespace native {
namespace {

void bind(float hitRatio,  uintptr_t stream_handle,at::Tensor base) {
    launch_add2( hitRatio, stream_handle, base);
}

Tensor new_view_weight_MM2d(const Tensor& weight_) {
  auto weight = weight_.expect_contiguous();
  const auto w_sizes = weight->sizes();
  TORCH_CHECK(w_sizes.size() == 4);
  int64_t s1 = w_sizes[0];
  int64_t s2 = c10::multiply_integers(w_sizes.slice(1));
  return weight->view({s1, s2});
}


void slow_conv2d_forward(
           const Tensor &input,
           const Tensor &output,
           const Tensor &weight_,
           const Tensor &bias,
           int64_t kH, int64_t kW,
           int64_t dH, int64_t dW,
           int64_t padH, int64_t padW) {
  auto weight = new_view_weight_MM2d(weight_);
//   slow_conv2d_shape_check(
//       input, {}, weight, bias, kH, kW, dH, dW, padH, padW, /*weight_nullable*/false);

  constexpr int dimf = 1;
  constexpr int dimh = 2;
  constexpr int dimw = 3;

  auto in_sizes = input.sizes();
  int64_t batchSize = in_sizes[0];
  int64_t nInputPlane  = in_sizes[dimf];
  int64_t inputHeight  = in_sizes[dimh];
  int64_t inputWidth   = in_sizes[dimw];
  int64_t nOutputPlane = weight.sizes()[0];
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  // Resize output
//   resize_output(output, {batchSize, nOutputPlane, outputHeight, outputWidth});

  // Create temporary columns
  at::Tensor columns;

  const bool requires_columns = (
      kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);

  if (requires_columns) {
    columns = at::empty({nInputPlane * kW * kH, outputHeight * outputWidth}, input.options());
  }

  if (bias.defined()) {
    TORCH_CHECK(bias.scalar_type() == input.scalar_type(),
                "Expected bias to have type ", input.scalar_type(),
                " but got ", bias.scalar_type());
    output.copy_(bias.view({-1, 1, 1}));
  } else {
    output.zero_();
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "slow_conv2d_cuda", [&] {
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
      // Matrix mulitply per output:
      auto input_n = input.select(0, elt);
      auto output_n = output.select(0, elt);

      if (requires_columns) {
        // Extract columns:
        // at::native::im2col(
        //   c10::cuda::getCurrentCUDAStream(),
        //   input_n.data_ptr<scalar_t>(),
        //   nInputPlane, inputHeight, inputWidth,
        //   outputHeight, outputWidth,
        //   kH, kW, padH, padW, dH, dW,
        //   1, 1,
        //   columns.data_ptr<scalar_t>()
        // );
      }

    //   // M,N,K are dims of matrix A and B
    //   // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    //   int64_t m = nOutputPlane;
    //   int64_t n = outputHeight * outputWidth;
    //   int64_t k = nInputPlane*kH*kW;

    //   // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    //   auto gemm_in_ptr = requires_columns ?
    //       columns.data_ptr<scalar_t>() :
    //       input_n.data_ptr<scalar_t>();
    //   at::cuda::blas::gemm(
    //       'n', 'n',
    //       n, m, k,
    //       scalar_t(1),
    //       gemm_in_ptr, n,
    //       weight.data_ptr<scalar_t>(), k,
    //       scalar_t(1),
    //       output_n.data_ptr<scalar_t>(), n
    //   );
    }
  });
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bind",
          &bind,
          "kernel warpper");
}

}}}