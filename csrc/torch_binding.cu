#include "siblas.h"
#include <torch/library.h>
#include <torch/extension.h>

// ======== Op registration via TORCH_LIBRARY ========

// Schema declaration
TORCH_LIBRARY(siblas, m) {
    m.def("linear_forward(Tensor input, Tensor weight, Tensor bias) -> Tensor");
    m.def("linear_backward(Tensor grad_output, Tensor input, Tensor weight) -> (Tensor, Tensor, Tensor)");
    m.def("cublas_linear_forward(Tensor input, Tensor weight, Tensor bias) -> Tensor");
    m.def("cublas_linear_backward(Tensor grad_output, Tensor input, Tensor weight) -> (Tensor, Tensor, Tensor)");
}

// CUDA implementation registration
TORCH_LIBRARY_IMPL(siblas, CUDA, m) {
    m.impl("linear_forward", &siblas_linear_forward);
    m.impl("linear_backward", &siblas_linear_backward);
    m.impl("cublas_linear_forward", &siblas_cublas_linear_forward);
    m.impl("cublas_linear_backward", &siblas_cublas_linear_backward);
}

// ======== Autograd: CUTLASS BF16x6 ========

class SiblasLinearFunction : public torch::autograd::Function<SiblasLinearFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias) {

        ctx->save_for_backward({input, weight});
        return siblas_linear_forward(input, weight, bias);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {

        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];

        auto [grad_input, grad_weight, grad_bias] =
            siblas_linear_backward(grad_outputs[0], input, weight);

        return {grad_input, grad_weight, grad_bias};
    }
};

torch::Tensor siblas_linear_autograd(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
    return SiblasLinearFunction::apply(input, weight, bias);
}

// ======== Autograd: cuBLAS TF32 ========

class CublasLinearFunction : public torch::autograd::Function<CublasLinearFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias) {

        ctx->save_for_backward({input, weight});
        return siblas_cublas_linear_forward(input, weight, bias);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {

        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];

        auto [grad_input, grad_weight, grad_bias] =
            siblas_cublas_linear_backward(grad_outputs[0], input, weight);

        return {grad_input, grad_weight, grad_bias};
    }
};

torch::Tensor siblas_cublas_linear_autograd(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
    return CublasLinearFunction::apply(input, weight, bias);
}

// ======== Register autograd dispatch ========

TORCH_LIBRARY_IMPL(siblas, Autograd, m) {
    m.impl("linear_forward", &siblas_linear_autograd);
    m.impl("cublas_linear_forward", &siblas_cublas_linear_autograd);
}

// Python module init (required for pip install to work)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "siblas: CUTLASS BF16x6 & cuBLAS TF32 GEMM for linear layers";
}
