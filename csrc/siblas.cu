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
#include "cutlass/epilogue/fusion/operations.hpp"

// Optimized SM100 BF16x6 persistent-B mainloop kernel
#include "sm100_mma_warpspecialized_emulated_optimized.hpp"

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
/// Uses optimized persistent-B mainloop (MainloopSm100TmaUmmaWarpSpecializedFastF32PersistentB)
///
/// Output layout is ColumnMajor (CUTLASS SM100 native). PyTorch wrappers handle the
/// row-major ↔ column-major reinterpretation via transposed GEMM calls.
///
///   GemmNN: A=RowMajor, B=ColumnMajor, C=ColumnMajor
///     Forward: Y=X@W^T   (A=X[M,K], B=W[N,K] as ColMajor = W^T, D=Y[M,N] ColMajor)
///   GemmNT: A=ColumnMajor, B=RowMajor, C=ColumnMajor
///     Backward dW: dW=dY^T@X  (A=dY[M,N] ColMajor = dY^T[N,M], B=X[M,K], D=dW[N,K])
///   GemmRR: A=RowMajor, B=RowMajor, C=ColumnMajor
///     Backward dX: dX=dY@W  (A=dY[M,N], B=W[N,K], D=dX[M,K] ColMajor)
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
using MmaTileShape        = Shape<_256,_64,_32>;

// Persistent-B carveout (matches bench_gemm.cu calculation)
constexpr int NumBands    = 3;
constexpr int MaxPBTiles  = 8;
constexpr int PBCarveout  =
    (int(get<1>(MmaTileShape{})) / int(get<0>(ClusterShape{}))) *
    int(get<2>(MmaTileShape{})) *
    int(sizeof(cutlass::bfloat16_t)) * NumBands * MaxPBTiles + 1024 + 64;

// All output layouts are ColumnMajor (SM100 native)
using LayoutOut = cutlass::layout::ColumnMajor;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Build the optimized persistent-B policy from a standard builder
/////////////////////////////////////////////////////////////////////////////////////////////////

// Helper: build OptPolicy for a given A/B layout pair
template <typename LayoutA, typename LayoutB>
struct MakeGemm {
    using EpiTmp = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementOutput, LayoutOut, Alignment,
        ElementOutput, LayoutOut, Alignment,
        cutlass::epilogue::FastF32NoSmemWarpSpecialized2Sm
    >::CollectiveOp;

    using StdBuilder = cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementInput, LayoutA, Alignment,
        ElementInput, LayoutB, Alignment,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename EpiTmp::SharedStorage)) + PBCarveout>,
        cutlass::gemm::KernelTmaWarpSpecialized2SmFastFP32Sm100
    >;

    using OptPolicy = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedFastF32PersistentB<
        4, 6,
        StdBuilder::SchedulerPipelineStageCount,
        StdBuilder::AccumulatorPipelineStageCount,
        NumBands,
        StdBuilder::ScalingFactor,
        StdBuilder::AccPromotionInterval,
        ClusterShape,
        typename StdBuilder::AccumulatorCopyAtom
    >;

    using Mainloop = cutlass::gemm::collective::CollectiveMma<
        OptPolicy, MmaTileShape, ElementInput,
        cutlass::gemm::TagToStrideA_t<LayoutA>, ElementInput,
        cutlass::gemm::TagToStrideB_t<LayoutB>,
        typename StdBuilder::TiledMma,
        typename StdBuilder::GmemTiledCopyA, typename StdBuilder::SmemLayoutAtomPairA,
        typename StdBuilder::CopyAtomPairA, cute::identity,
        typename StdBuilder::GmemTiledCopyB, typename StdBuilder::SmemLayoutAtomPairB,
        typename StdBuilder::CopyAtomPairB, cute::identity
    >;

    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementOutput, LayoutOut, Alignment,
        ElementOutput, LayoutOut, Alignment,
        cutlass::epilogue::FastF32NoSmemWarpSpecialized2Sm
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>, Mainloop, Epilogue, void>;
    using Type = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// GemmNN: A=RowMajor, B=ColumnMajor  (forward: Y=X@W^T)
using GemmNN = MakeGemm<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::Type;

// GemmNT: A=ColumnMajor, B=RowMajor  (backward dW: dW=dY^T@X)
using GemmNT = MakeGemm<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::Type;

// GemmRR: A=RowMajor, B=RowMajor  (backward dX: dX=dY@W)
using GemmRR = MakeGemm<cutlass::layout::RowMajor, cutlass::layout::RowMajor>::Type;

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
    // Output is ColumnMajor[M,N]: outer dim is N, inner dim is M
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

    // Allocate output [M, N] as column-major (CUTLASS SM100 native output layout).
    // col-major [M,N] = row-major [N,M], so we allocate [N,M] and transpose the view.
    auto output_cm = torch::empty({N, M}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Y = X @ W^T using CUTLASS BF16x6 optimized persistent-B kernel
    // GemmNN: A[M,K]=RowMajor, B[N,K]=RowMajor-as-ColMajor, D[M,N]=ColMajor
    run_cutlass_gemm<GemmNN>(
        M, N, K,
        1.0f, 0.0f,
        input_c.data_ptr<float>(),
        weight_c.data_ptr<float>(),
        output_cm.data_ptr<float>(),   // C (unused with beta=0)
        output_cm.data_ptr<float>(),   // D (col-major [M,N] stored as [N,M] row-major)
        stream);

    // Reinterpret col-major [M,N] buffer as row-major [M,N] via transpose
    auto output = output_cm.t().contiguous();

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
    // GemmRR: A=dY[M,N] RowMajor, B=W[N,K] RowMajor -> D=dX[M,K] ColMajor
    auto grad_input_cm = torch::empty({K, M}, input.options());
    run_cutlass_gemm<GemmRR>(
        M, K, N,      // GEMM dims: M, N_out=K, K_inner=N
        1.0f, 0.0f,
        grad_output_c.data_ptr<float>(),   // A [M, N] RowMajor
        weight_c.data_ptr<float>(),        // B [N, K] RowMajor
        grad_input_cm.data_ptr<float>(),
        grad_input_cm.data_ptr<float>(),
        stream);
    auto grad_input = grad_input_cm.t().contiguous();

    // ---- grad_weight = grad_output^T @ input -> [N, K] ----
    // GemmNT: A=dY[M,N] ColMajor (= dY^T[N,M]), B=X[M,K] RowMajor -> D=dW[N,K] ColMajor
    auto grad_weight_cm = torch::empty({K, N}, weight.options());
    run_cutlass_gemm<GemmNT>(
        N, K, M,      // GEMM dims: M_out=N, N_out=K, K_inner=M
        1.0f, 0.0f,
        grad_output_c.data_ptr<float>(),   // A: dY[M,N] treated as ColMajor -> dY^T[N,M]
        input_c.data_ptr<float>(),         // B: X[M,K] RowMajor
        grad_weight_cm.data_ptr<float>(),
        grad_weight_cm.data_ptr<float>(),
        stream);
    auto grad_weight = grad_weight_cm.t().contiguous();

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
