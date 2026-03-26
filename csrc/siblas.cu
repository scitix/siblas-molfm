#include "siblas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h> // 必须 include 这个头文件

#include <stdexcept>
#include <mutex>

// CUTLASS includes
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm/collective/sm100_mma_warpspecialized_emulated_optimized.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// CUTLASS BF16x6 emulated FP32 GEMM kernel configurations
/// Based on CUTLASS example 78b (Blackwell SM100)
///
/// We define two GEMM kernel types:
///   GemmNN: A=RowMajor,   B=ColumnMajor  (used for forward Y=X@W^T and backward dX=dY@W)
///   GemmNT: A=RowMajor,   B=RowMajor     (used for backward dW=dY^T@X)
/////////////////////////////////////////////////////////////////////////////////////////////////

// Common configuration
using ElementInput        = float;
using ElementOutput       = float;
using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm100;
using OperatorClass       = cutlass::arch::OpClassTensorOp;
constexpr int Alignment   = 128 / cutlass::sizeof_bits<float>::value;  // = 4

// Kernel perf config
using ClusterShape        = Shape<_2,_1,_1>;
using MmaTileShape        = Shape<_256,_128,_16>;

// Schedule for BF16x6 emulated GEMM (non-Smem variant: A operand in TMEM)
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmFastFP32Sm100;

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmNN: A = RowMajor, B = ColumnMajor -> C = RowMajor
// Used for:
//   Forward: Y[M,N] = X[M,K](RowMajor) @ W^T  =>  B = W[N,K] treated as ColMajor[K,N]
//   B is implicitly transposed (RowMajor storage read as ColMajor).
/////////////////////////////////////////////////////////////////////////////////////////////////

using LayoutA_NN          = cutlass::layout::RowMajor;
using LayoutB_NN          = cutlass::layout::ColumnMajor;
using LayoutC_NN          = cutlass::layout::RowMajor;

using CollectiveEpilogue_NN = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementOutput, LayoutC_NN, Alignment,
    ElementOutput, LayoutC_NN, Alignment,
    cutlass::epilogue::FastF32NoSmemWarpSpecialized2Sm
  >::CollectiveOp;

using CollectiveMainloop_NN = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementInput, LayoutA_NN, Alignment,
    ElementInput, LayoutB_NN, Alignment,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue_NN::SharedStorage))>,
    MainloopSchedule
  >::CollectiveOp;

using GemmKernel_NN = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop_NN,
    CollectiveEpilogue_NN,
    void>;

using GemmNN = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_NN>;

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmNT: A = RowMajor, B = RowMajor -> C = ColumnMajor
// Used for:
//   Backward dW: dW[N,K] = dY^T[N,M] @ X[M,K]
//   We compute: C[N,K] = dY^T @ X
//   In CUTLASS terms: A = dY viewed as ColMajor (to get dY^T as RowMajor), B = X RowMajor
//   Actually simpler: use A=ColumnMajor for dY[M,N] (gives us dY^T[N,M] in row-major sense)
//                     B=RowMajor for X[M,K]
/////////////////////////////////////////////////////////////////////////////////////////////////

using LayoutA_NT          = cutlass::layout::ColumnMajor;  // dY[M,N] col-major = dY^T[N,M] row-major
using LayoutB_NT          = cutlass::layout::RowMajor;
using LayoutC_NT          = cutlass::layout::RowMajor;

using CollectiveEpilogue_NT = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementOutput, LayoutC_NT, Alignment,
    ElementOutput, LayoutC_NT, Alignment,
    cutlass::epilogue::FastF32NoSmemWarpSpecialized2Sm
  >::CollectiveOp;

using CollectiveMainloop_NT = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementInput, LayoutA_NT, Alignment,
    ElementInput, LayoutB_NT, Alignment,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue_NT::SharedStorage))>,
    MainloopSchedule
  >::CollectiveOp;

using GemmKernel_NT = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop_NT,
    CollectiveEpilogue_NT,
    void>;

using GemmNT = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_NT>;

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmRR: A = RowMajor, B = RowMajor -> C = RowMajor
// Used for:
//   Backward dX: dX[M,K] = dY[M,N](RowMajor) @ W[N,K](RowMajor)
//   No implicit transpose on either operand.
/////////////////////////////////////////////////////////////////////////////////////////////////

using LayoutA_RR          = cutlass::layout::RowMajor;
using LayoutB_RR          = cutlass::layout::RowMajor;
using LayoutC_RR          = cutlass::layout::RowMajor;

using CollectiveEpilogue_RR = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementOutput, LayoutC_RR, Alignment,
    ElementOutput, LayoutC_RR, Alignment,
    cutlass::epilogue::FastF32NoSmemWarpSpecialized2Sm
  >::CollectiveOp;

using CollectiveMainloop_RR = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementInput, LayoutA_RR, Alignment,
    ElementInput, LayoutB_RR, Alignment,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue_RR::SharedStorage))>,
    MainloopSchedule
  >::CollectiveOp;

using GemmKernel_RR = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop_RR,
    CollectiveEpilogue_RR,
    void>;

using GemmRR = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_RR>;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Stride types
/////////////////////////////////////////////////////////////////////////////////////////////////

using StrideA_NN = typename GemmNN::GemmKernel::StrideA;
using StrideB_NN = typename GemmNN::GemmKernel::StrideB;
using StrideC_NN = typename GemmNN::GemmKernel::StrideC;
using StrideD_NN = typename GemmNN::GemmKernel::StrideD;

using StrideA_NT = typename GemmNT::GemmKernel::StrideA;
using StrideB_NT = typename GemmNT::GemmKernel::StrideB;
using StrideC_NT = typename GemmNT::GemmKernel::StrideC;
using StrideD_NT = typename GemmNT::GemmKernel::StrideD;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Helper: run a CUTLASS GEMM
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename GemmType>
void run_cutlass_gemm(
    int M, int N, int K,
    float alpha, float beta,
    const float* A, const float* B,
    const float* C, float* D,
    cudaStream_t stream) {

    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename GemmType::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {A, stride_A, B, stride_B},
        {{alpha, beta}, C, stride_C, D, stride_D}
    };

    GemmType gemm;

    size_t workspace_size = GemmType::get_workspace_size(arguments);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, 
                              torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8));
    cutlass::Status status;

    status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS GEMM can_implement failed: ") +
            cutlass::cutlassGetStatusString(status));
    }

    status = gemm.initialize(arguments, workspace.data_ptr(), stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS GEMM initialize failed: ") +
            cutlass::cutlassGetStatusString(status));
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS GEMM run failed: ") +
            cutlass::cutlassGetStatusString(status));
    }
}

}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////
// Forward: Y = X @ W^T + bias
// X: [M, K] RowMajor, W: [N, K] RowMajor, bias: [N] -> Y: [M, N]
//
// CUTLASS GEMM (GemmNN):
//   A = X [M, K] RowMajor
//   B = W [N, K] RowMajor memory = [K, N] ColumnMajor
//   C = Y [M, N] (output, ColumnMajor for CUTLASS)
/////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor siblas_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K,
                "weight K dimension mismatch: expected ", K, " got ", weight.size(1));

    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();

    // Allocate output [M, N]
    auto output = torch::empty({M, N}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Y = X @ W^T using CUTLASS BF16x6 emulated FP32
    // GemmNN: A[M,K]=RowMajor, B[N,K]=RowMajor-as-ColMajor, C/D[M,N]=ColMajor
    run_cutlass_gemm<GemmNN>(
        M, N, K,
        1.0f, 0.0f,
        input_c.data_ptr<float>(),
        weight_c.data_ptr<float>(),
        output.data_ptr<float>(),   // C (unused with beta=0)
        output.data_ptr<float>(),   // D
        stream);

    // Add bias if provided
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.size(0) == N, "bias size mismatch");
        output.add_(bias.unsqueeze(0));
    }

    return output;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Backward:
// grad_input  = grad_output @ W        -> [M, K]
// grad_weight = grad_output^T @ input   -> [N, K]
// grad_bias   = sum(grad_output, dim=0) -> [N]
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> siblas_linear_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight) {
    const c10::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    const int M = grad_output.size(0);
    const int N = grad_output.size(1);
    const int K = weight.size(1);

    auto grad_output_c = grad_output.contiguous();
    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // ---- grad_input = grad_output @ W -> [M, K] ----
    // grad_output: [M, N] RowMajor, W: [N, K] RowMajor
    // This is C[M,K] = A[M,N] * B[N,K]  (no transpose on either operand)
    // GemmRR: A=RowMajor, B=RowMajor -> C=RowMajor
    auto grad_input = torch::empty({M, K}, input.options());
    run_cutlass_gemm<GemmRR>(
        M, K, N,      // GEMM dimensions: M, N_out=K, K_inner=N
        1.0f, 0.0f,
        grad_output_c.data_ptr<float>(),   // A [M, N] RowMajor
        weight_c.data_ptr<float>(),        // B [N, K] RowMajor
        grad_input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        stream);

    // ---- grad_weight = grad_output^T @ input -> [N, K] ----
    // grad_output^T: [N, M], input: [M, K]
    // This is C[N,K] = A^T[N,M] * B[M,K]
    // For GemmNT: A=ColMajor (dY[M,N] stored col-major = dY^T[N,M] row-major),
    //             B=RowMajor (X[M,K])
    auto grad_weight = torch::empty({N, K}, weight.options()).zero_();
    run_cutlass_gemm<GemmNT>(
        N, K, M,      // GEMM dimensions: M_out=N, N_out=K, K_inner=M
        1.0f, 0.0f,
        grad_output_c.data_ptr<float>(),   // A: dY[M,N] treated as ColMajor -> dY^T[N,M]
        input_c.data_ptr<float>(),         // B: X[M,K] RowMajor
        grad_weight.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        stream);

    // ---- grad_bias = sum(grad_output, dim=0) -> [N] ----
    auto grad_bias = grad_output.sum(0);

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// cuBLAS TF32 Linear implementation
/// Uses cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 for TF32 Tensor Core acceleration.
///
/// Row-major PyTorch tensors in cuBLAS column-major convention:
///   PyTorch X[M,K] row-major  = cuBLAS X[K,M] col-major
///   PyTorch W[N,K] row-major  = cuBLAS W[K,N] col-major
///   PyTorch Y[M,N] row-major  = cuBLAS Y[N,M] col-major
///
///   Y = X @ W^T  =>  cuBLAS: Y[N,M] = W[N,K] @ X[K,M]
///   i.e. cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, W, X, Y)
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

// Thread-safe cuBLAS handle singleton
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
        // Set math mode to TF32
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    });
    return handle;
}

// Helper: run a cuBLAS GEMM with TF32
// C[m, n] = alpha * op(A)[m, k] @ op(B)[k, n] + beta * C[m, n]
// All dimensions are in cuBLAS column-major convention.
void run_cublas_gemm(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    float alpha, float beta,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    cudaStream_t stream) {

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    cublasStatus_t st = cublasGemmEx(
        handle,
        transa, transb,
        m, n, k,
        &alpha,
        A, CUDA_R_32F, lda,
        B, CUDA_R_32F, ldb,
        &beta,
        C, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLAS GemmEx failed with status ") + std::to_string(st));
    }
}

}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLAS TF32 Forward: Y = X @ W^T + bias
/////////////////////////////////////////////////////////////////////////////////////////////////

torch::Tensor siblas_cublas_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K,
                "weight K dimension mismatch: expected ", K, " got ", weight.size(1));

    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();

    auto output = torch::empty({M, N}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Y = X @ W^T
    // Row-major in cuBLAS col-major convention:
    //   Y[N,M] = W[N,K] @ X[K,M]
    //   W[N,K] row-major = W stored as [K,N] col-major, need CUBLAS_OP_T on lda=K
    //   X[M,K] row-major = X stored as [K,M] col-major, CUBLAS_OP_N with lda=K
    //   Y[M,N] row-major = Y stored as [N,M] col-major, ldc=N
    run_cublas_gemm(
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        1.0f, 0.0f,
        weight_c.data_ptr<float>(), K,   // W[N,K] row-major -> lda=K
        input_c.data_ptr<float>(),  K,   // X[M,K] row-major -> ldb=K
        output.data_ptr<float>(),   N,   // Y[M,N] row-major -> ldc=N
        stream);

    // Add bias if provided
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.size(0) == N, "bias size mismatch");
        output.add_(bias.unsqueeze(0));
    }

    return output;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLAS TF32 Backward
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> siblas_cublas_linear_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& weight) {

    const c10::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    const int M = grad_output.size(0);
    const int N = grad_output.size(1);
    const int K = weight.size(1);

    auto grad_output_c = grad_output.contiguous();
    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // ---- grad_input = grad_output @ W -> [M, K] ----
    // Row-major: dX[M,K] = dY[M,N] @ W[N,K]
    // cuBLAS col-major: dX[K,M] = W^T[K,N] @ dY[N,M]
    //   W[N,K] row-major stored as [K,N] col-major, CUBLAS_OP_N, lda=K
    //   dY[M,N] row-major stored as [N,M] col-major, CUBLAS_OP_N, ldb=N
    //   dX[M,K] row-major stored as [K,M] col-major, ldc=K
    auto grad_input = torch::empty({M, K}, input.options());
    run_cublas_gemm(
        CUBLAS_OP_N, CUBLAS_OP_N,
        K, M, N,
        1.0f, 0.0f,
        weight_c.data_ptr<float>(),      K,   // W[N,K] row-major -> lda=K
        grad_output_c.data_ptr<float>(), N,   // dY[M,N] row-major -> ldb=N
        grad_input.data_ptr<float>(),    K,   // dX[M,K] row-major -> ldc=K
        stream);

    // ---- grad_weight = grad_output^T @ input -> [N, K] ----
    // Row-major dW[N,K] = col-major [K,N]
    // dW = dY^T @ X
    // cuBLAS: C(K,N) = op(A)(K,M) * op(B)(M,N)
    //   op(A) = X[M,K]_row = [K,M]_col, CUBLAS_OP_N -> (K,M)
    //   op(B) = dY[M,N]_row = [N,M]_col, CUBLAS_OP_T -> (M,N)
    auto grad_weight = torch::empty({N, K}, weight.options());
    run_cublas_gemm(
        CUBLAS_OP_N, CUBLAS_OP_T,
        K, N, M,
        1.0f, 0.0f,
        input_c.data_ptr<float>(),       K,   // X[M,K] row-major -> lda=K
        grad_output_c.data_ptr<float>(), N,   // dY[M,N] row-major -> ldb=N
        grad_weight.data_ptr<float>(),   K,   // dW[N,K] row-major -> ldc=K
        stream);

    // ---- grad_bias = sum(grad_output, dim=0) -> [N] ----
    auto grad_bias = grad_output.sum(0);

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}
