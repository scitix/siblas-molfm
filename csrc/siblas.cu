#include "siblas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <stdexcept>

namespace {

// Get or create a per-device cuBLAS handle (thread-local cache)
cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
        // Enable TF32 for FP32 GEMM (Ampere+)
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return handle;
}

}  // namespace

// Forward: Y = X @ W^T + bias
// X: [M, K], W: [N, K], bias: [N] (optional) -> Y: [M, N]
torch::Tensor siblas_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    // input: [M, K], weight: [N, K]
    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K,
                "weight K dimension mismatch: expected ", K, " got ", weight.size(1));

    // Ensure contiguous
    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();

    // Allocate output [M, N]
    auto output = torch::empty({M, N}, input.options());

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, c10::cuda::getCurrentCUDAStream());

    // Y = X @ W^T
    // X: [M, K], W^T: [K, N]
    // A = X [M, K], B = W^T [K, N], C = Y [M, N]
    // In row-major: C[M,N] = A[M,K] * B[K,N]
    // But W is [N, K], so W^T is [K, N]
    // cuBLAS col-major: C^T[N,M] = (W^T)^T[N,K] * X^T[K,M] = W[N,K] * X^T[K,M]
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            weight_c.data_ptr<float>(), K,   // W[N,K] row-major -> col-major lda=K, op=T
            input_c.data_ptr<float>(), K,    // X[M,K] row-major -> col-major lda=K, op=N
            &beta,
            output.data_ptr<float>(), N);    // Y[M,N] row-major -> col-major ld=N

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS SGEMM failed in forward");
        }
    }

    // Add bias if provided
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.size(0) == N, "bias size mismatch");
        output.add_(bias.unsqueeze(0));
    }

    return output;
}

// Backward:
// grad_input  = grad_output @ W        -> [M, K]
// grad_weight = grad_output^T @ input   -> [N, K]
// grad_bias   = sum(grad_output, dim=0) -> [N]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> siblas_linear_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight) {

    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    const int M = grad_output.size(0);
    const int N = grad_output.size(1);
    const int K = weight.size(1);

    auto grad_output_c = grad_output.contiguous();
    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, c10::cuda::getCurrentCUDAStream());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // grad_input = grad_output @ W -> [M, K]
    // grad_output: [M, N], W: [N, K]
    // col-major: result^T[K,M] = W^T[K,N] * grad_output^T[N,M]
    auto grad_input = torch::empty({M, K}, input.options());
    {
        cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            K, M, N,
            &alpha,
            weight_c.data_ptr<float>(), K,        // W[N,K] row-major -> col-major as [K,N], ld=K
            grad_output_c.data_ptr<float>(), N,    // dY[M,N] row-major -> col-major as [N,M], ld=N
            &beta,
            grad_input.data_ptr<float>(), K);      // dX[M,K] row-major -> col-major as [K,M], ld=K

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS SGEMM failed in backward (grad_input)");
        }
    }

    // grad_weight = grad_output^T @ input -> [N, K]
    // grad_output^T: [N, M], input: [M, K]
    // col-major: result^T[K,N] = input^T[K,M] * grad_output[M,N]
    auto grad_weight = torch::empty({N, K}, weight.options());
    {
        cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            K, N, M,
            &alpha,
            input_c.data_ptr<float>(), K,          // X[M,K] row-major -> col-major as [K,M], ld=K
            grad_output_c.data_ptr<float>(), N,    // dY[M,N] row-major -> col-major as [N,M], ld=N, op=T -> [M,N]
            &beta,
            grad_weight.data_ptr<float>(), K);     // dW[N,K] row-major -> col-major as [K,N], ld=K

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS SGEMM failed in backward (grad_weight)");
        }
    }

    // grad_bias = sum(grad_output, dim=0) -> [N]
    auto grad_bias = grad_output.sum(0);

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}
