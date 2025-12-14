#include <torch/extension.h>
#include <torch/library.h>
#include <vector>
using torch::Tensor;

// =========================================
// 原始 linear_double_backward 实现
// =========================================
std::tuple<Tensor, Tensor, Tensor> linear_double_backward(
    const std::vector<Tensor>& grads,
    const Tensor& self,
    const Tensor& grad_output,
    const Tensor& weight) {

  if (!grad_output.defined()) {
    return std::make_tuple(Tensor(), Tensor(), Tensor());
  }

  Tensor grad_self, grad_grad_output, grad_weight;

  if (grads[1].defined()) {
    grad_self =
        (grad_output.dim() == 1 ? grad_output.unsqueeze(0) : grad_output)
            .matmul(grads[1]);
    if (grad_output.dim() == 1) {
      grad_self = grad_self.squeeze(0);
    }
  }
  if (grads[0].defined()) {
    grad_weight =
        (grad_output.dim() == 1 ? grad_output.unsqueeze(1) : grad_output.mT())
            .matmul(grads[0].dim() == 1 ? grads[0].unsqueeze(0) : grads[0]);
  }

  if (grads[0].defined() || grads[1].defined() || grads[2].defined()) {
    grad_grad_output = at::zeros_like(grad_output);
    if (grad_output.dim() == 1) {
      grad_grad_output = grad_grad_output.unsqueeze(0);
    }
  }

  if (grads[0].defined()) {
    grad_grad_output =
        grad_grad_output +
        (grads[0].dim() == 1 ? grads[0].unsqueeze(0) : grads[0])
            .matmul(weight.mT());
  }
  if (grads[1].defined()) {
    grad_grad_output =
        grad_grad_output +
        (self.dim() == 1 ? self.unsqueeze(0) : self).matmul(grads[1].mT());
  }
  if (grads[2].defined()) {
    grad_grad_output = grad_grad_output + grads[2];
  }
  if (grad_grad_output.defined() && grad_output.dim() == 1) {
    grad_grad_output = grad_grad_output.squeeze(0);
  }

  return std::make_tuple(
      std::move(grad_self),
      std::move(grad_grad_output),
      std::move(grad_weight));
}

// =========================================
// wrapper 给 Python op 调用
// =========================================
std::tuple<Tensor, Tensor, Tensor> linear_double_backward_wrapper(
    const Tensor& grad_grad_input,
    const Tensor& grad_grad_weight,
    const Tensor& grad_grad_bias,
    const Tensor& self,
    const Tensor& grad_output,
    const Tensor& weight) {

    std::vector<Tensor> grads = {
        grad_grad_input,
        grad_grad_weight,
        grad_grad_bias
    };
    return linear_double_backward(grads, self, grad_output, weight);
}

// =========================================
// 注册为 torch.ops.autograd.linear_double_backward
// =========================================
TORCH_LIBRARY(autograd, m) {
    m.def("linear_double_backward(Tensor ggi, Tensor ggw, Tensor ggb, Tensor self, Tensor grad_output, Tensor weight)"
          " -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(autograd, CPU, m) {
    m.impl("linear_double_backward", linear_double_backward_wrapper);
}

// =========================================
// 关键：这是 Python module 入口！必须加！
// =========================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 不需要写内容，但必须存在
}
