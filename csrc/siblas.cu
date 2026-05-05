#include "siblas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <stdexcept>
#include <mutex>

// CUTLASS includes
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm/collective/sm100_mma_warpspecialized_emulated_optimized.hpp"

#include "cutlass/util/packed_stride.hpp"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// CUTLASS BF16x6 emulated FP32 GEMM kernel configurations
/// Based on CUTLASS example 78b (Blackwell SM100)
///
/// GemmNN:   A=RowMajor, B=ColMajor  -> forward Y=X@W^T (with fused bias)
/// GemmNT:   A=ColMajor, B=RowMajor  -> backward dW=dY^T@X
/// GemmRR:   A=RowMajor, B=RowMajor  -> backward dX=dY@W
/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementInput        = float;
using ElementOutput       = float;
using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm100;
using OperatorClass       = cutlass::arch::OpClassTensorOp;
constexpr int Alignment   = 128 / cutlass::sizeof_bits<float>::value;  // 4

using ClusterShape        = Shape<_2,_1,_1>;
using MmaTileShape        = Shape<_256,_64,_32>;

// BF16x6 persistent-band constants (matches bench_gemm.cu)
constexpr int NumBands_   = 3;
constexpr int MaxPBTiles_ = 8;
// Extra SMEM needed for the persistent buffer: same formula as bench_gemm.cu
constexpr int PBCarveout_ =
    (int(get<1>(MmaTileShape{})) / int(get<0>(ClusterShape{}))) *
    int(get<2>(MmaTileShape{})) *
    int(sizeof(cutlass::bfloat16_t)) * NumBands_ * MaxPBTiles_ + 1024 + 64;

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmNN_Bias: A=RowMajor, B=ColMajor -> D=RowMajor  (forward, fused bias)
// D = alpha * X @ W^T + bias[n]
/////////////////////////////////////////////////////////////////////////////////////////////////

using LayoutA_NN          = cutlass::layout::RowMajor;
using LayoutB_NN          = cutlass::layout::ColumnMajor;
using LayoutC_NN          = cutlass::layout::RowMajor;

// Fused per-row bias (row-major output: bias[n] broadcast over M rows)
using FusedBiasOp = cutlass::epilogue::fusion::LinCombPerRowBias<
    float,   // ElementOutput
    float,   // ElementCompute
    float,   // ElementBias
    float,   // ElementSource (C)
    float,   // ElementScalar
    Alignment>;

// Plain epilogue (no bias) — built first to get SharedStorage size for carveout
using CollectiveEpilogue_NN = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, float, LayoutC_NN, Alignment, float, LayoutC_NN, Alignment,
    cutlass::epilogue::FastF32NoSmemWarpSpecialized2Sm>::CollectiveOp;

// Bias epilogue
using CollectiveEpilogue_NN_Bias = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementOutput, LayoutC_NN, Alignment,
    ElementOutput, LayoutC_NN, Alignment,
    cutlass::epilogue::FastF32NoSmemWarpSpecialized2Sm,
    FusedBiasOp
  >::CollectiveOp;

// StdBuilder extracts TiledMma and copy types; carveout reserves SMEM for PB + epilogue
using StdBuilder_NN = cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementInput, LayoutA_NN, Alignment,
    ElementInput, LayoutB_NN, Alignment,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_NN::SharedStorage)) + PBCarveout_>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmFastFP32Sm100>;

// Optimized BF16x6 persistent-band mainloop policy (same as bench_gemm.cu)
using OptPolicy_NN = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedFastF32PersistentB<
    4, 6,
    StdBuilder_NN::SchedulerPipelineStageCount,
    StdBuilder_NN::AccumulatorPipelineStageCount,
    NumBands_,
    StdBuilder_NN::ScalingFactor,
    StdBuilder_NN::AccPromotionInterval,
    ClusterShape,
    typename StdBuilder_NN::AccumulatorCopyAtom>;

// Mainloop using the optimized BF16x6 policy
using CollectiveMainloop_NN = cutlass::gemm::collective::CollectiveMma<
    OptPolicy_NN, MmaTileShape, float,
    cutlass::gemm::TagToStrideA_t<LayoutA_NN>, float,
    cutlass::gemm::TagToStrideB_t<LayoutB_NN>,
    typename StdBuilder_NN::TiledMma,
    typename StdBuilder_NN::GmemTiledCopyA, typename StdBuilder_NN::SmemLayoutAtomPairA,
    typename StdBuilder_NN::CopyAtomPairA, cute::identity,
    typename StdBuilder_NN::GmemTiledCopyB, typename StdBuilder_NN::SmemLayoutAtomPairB,
    typename StdBuilder_NN::CopyAtomPairB, cute::identity>;

using GemmKernel_NN_Bias = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop_NN,
    CollectiveEpilogue_NN_Bias,
    void>;

using GemmKernel_NN = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop_NN,
    CollectiveEpilogue_NN,
    void>;

using GemmNN_Bias = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_NN_Bias>;
using GemmNN      = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_NN>;

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmNT: A=ColMajor, B=RowMajor -> C=RowMajor  (backward dW = dY^T @ X)
/////////////////////////////////////////////////////////////////////////////////////////////////

using LayoutA_NT          = cutlass::layout::ColumnMajor;
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
    cutlass::gemm::KernelTmaWarpSpecialized2SmFastFP32Sm100
  >::CollectiveOp;

using GemmKernel_NT = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop_NT,
    CollectiveEpilogue_NT,
    void>;

using GemmNT = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_NT>;

/////////////////////////////////////////////////////////////////////////////////////////////////
// GemmRR: A=RowMajor, B=RowMajor -> C=RowMajor  (backward dX = dY @ W)
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
    cutlass::gemm::KernelTmaWarpSpecialized2SmFastFP32Sm100
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
// Helpers: run CUTLASS GEMM (plain) and run CUTLASS GEMM with fused row-bias
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
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUTLASS can_implement: ") + cutlass::cutlassGetStatusString(status));

    status = gemm.initialize(arguments, workspace.data_ptr(), stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUTLASS initialize: ") + cutlass::cutlassGetStatusString(status));

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUTLASS run: ") + cutlass::cutlassGetStatusString(status));
}

// Forward with fused per-row bias: D = alpha * A*B + bias[n]
void run_cutlass_gemm_bias(
    int M, int N, int K,
    const float* A, const float* B,
    const float* bias,
    float* D,
    cudaStream_t stream) {

    using GemmType = GemmNN_Bias;
    using StrideA  = typename GemmType::GemmKernel::StrideA;
    using StrideB  = typename GemmType::GemmKernel::StrideB;
    using StrideD  = typename GemmType::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    using EpilogueArgs = typename GemmType::GemmKernel::CollectiveEpilogue::Arguments;
    EpilogueArgs epi_args{};
    epi_args.thread.alpha    = 1.0f;
    epi_args.thread.beta     = 0.0f;
    epi_args.thread.bias_ptr = bias;
    epi_args.ptr_C           = D;
    epi_args.dC              = stride_D;
    epi_args.ptr_D           = D;
    epi_args.dD              = stride_D;

    typename GemmType::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {A, stride_A, B, stride_B},
        epi_args
    };

    GemmType gemm;
    size_t workspace_size = GemmType::get_workspace_size(arguments);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                              torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8));
    cutlass::Status status;

    status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUTLASS bias can_implement: ") + cutlass::cutlassGetStatusString(status));

    status = gemm.initialize(arguments, workspace.data_ptr(), stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUTLASS bias initialize: ") + cutlass::cutlassGetStatusString(status));

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("CUTLASS bias run: ") + cutlass::cutlassGetStatusString(status));
}

}  // namespace


/////////////////////////////////////////////////////////////////////////////////////////////////
// Forward: Y = X @ W^T + bias
// X: [M, K] RowMajor, W: [N, K] RowMajor, bias: [N] -> Y: [M, N]
//
// Bias is fused into the CUTLASS epilogue (one kernel, no extra add_ pass).
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

    auto input_c  = input.contiguous();
    auto weight_c = weight.contiguous();
    auto output   = torch::empty({M, N}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const bool has_bias = bias.defined() && bias.numel() > 0;
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.size(0) == N, "bias size mismatch");
        auto bias_c = bias.contiguous();
        // Single kernel: GEMM + bias fused in epilogue
        run_cutlass_gemm_bias(
            M, N, K,
            input_c.data_ptr<float>(),
            weight_c.data_ptr<float>(),
            bias_c.data_ptr<float>(),
            output.data_ptr<float>(),
            stream);
    } else {
        // No bias: plain GEMM
        run_cutlass_gemm<GemmNN>(
            M, N, K,
            1.0f, 0.0f,
            input_c.data_ptr<float>(),
            weight_c.data_ptr<float>(),
            output.data_ptr<float>(),
            output.data_ptr<float>(),
            stream);
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
/// cuBLAS LT FP32 Linear implementation with fused bias
///
/// Uses cublasLtMatmul (TF32 Tensor Core) with CUBLASLT_MATMUL_DESC_BIAS_POINTER
/// so the bias addition is fused inside the GEMM kernel — no separate add_ kernel.
///
/// Row-major PyTorch tensors in cuBLAS column-major convention:
///   Y = X @ W^T   =>   cuBLAS: Y[N,M] = W[N,K](T) @ X[K,M](N)
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

struct CublasLtState {
    cublasLtHandle_t handle = nullptr;

    CublasLtState() {
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasLtCreate failed");
    }
    ~CublasLtState() {
        if (handle) cublasLtDestroy(handle);
    }
};

cublasLtHandle_t get_cublaslt_handle() {
    static CublasLtState state;
    return state.handle;
}

// cuBLAS LT: C[m,n] = alpha * op(A)[m,k] @ op(B)[k,n] + beta*C + bias[m]
// bias_ptr: column-major bias, length = m (i.e., N in our linear layer)
void run_cublaslt_gemm_bias(
    int m, int n, int k,
    float alpha, float beta,
    const float* A, int lda, cublasOperation_t opA,
    const float* B, int ldb, cublasOperation_t opB,
    float* C, int ldc,
    const float* bias_ptr,
    cudaStream_t stream) {

    cublasLtHandle_t lt = get_cublaslt_handle();

    cublasLtMatmulDesc_t   matmul_desc = nullptr;
    cublasLtMatrixLayout_t layout_A    = nullptr;
    cublasLtMatrixLayout_t layout_B    = nullptr;
    cublasLtMatrixLayout_t layout_C    = nullptr;
    cublasLtMatmulPreference_t pref    = nullptr;

    auto cleanup = [&]() {
        if (pref)        cublasLtMatmulPreferenceDestroy(pref);
        if (layout_C)    cublasLtMatrixLayoutDestroy(layout_C);
        if (layout_B)    cublasLtMatrixLayoutDestroy(layout_B);
        if (layout_A)    cublasLtMatrixLayoutDestroy(layout_A);
        if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
    };

    auto check = [&](cublasStatus_t st, const char* msg) {
        if (st != CUBLAS_STATUS_SUCCESS) {
            cleanup();
            throw std::runtime_error(std::string(msg) + std::to_string(static_cast<int>(st)));
        }
    };

    check(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F),
          "cublasLtMatmulDescCreate: ");
    check(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)),
          "set TRANSA: ");
    check(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
          "set TRANSB: ");

    if (bias_ptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        check(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                             &epilogue, sizeof(epilogue)), "set EPILOGUE: ");
        check(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                             &bias_ptr, sizeof(bias_ptr)), "set BIAS_POINTER: ");
    }

    // Matrix layouts (col-major sizes for cublasLt)
    check(cublasLtMatrixLayoutCreate(&layout_A, CUDA_R_32F,
          opA == CUBLAS_OP_N ? m : k, opA == CUBLAS_OP_N ? k : m, lda), "layout_A: ");
    check(cublasLtMatrixLayoutCreate(&layout_B, CUDA_R_32F,
          opB == CUBLAS_OP_N ? k : n, opB == CUBLAS_OP_N ? n : k, ldb), "layout_B: ");
    check(cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_32F, m, n, ldc), "layout_C: ");

    constexpr size_t kWsBytes = 32 * 1024 * 1024;
    check(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate: ");
    size_t wsz = kWsBytes;
    check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &wsz, sizeof(wsz)), "set WS: ");

    cublasLtMatmulHeuristicResult_t hr{};
    int nr = 0;
    cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(lt, matmul_desc,
                             layout_A, layout_B, layout_C, layout_C, pref, 1, &hr, &nr);
    if (hst != CUBLAS_STATUS_SUCCESS || nr == 0) {
        cleanup();
        throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic: no algo");
    }

    // Shared workspace from PyTorch allocator
    auto ws_tensor = torch::empty({static_cast<int64_t>(kWsBytes)},
                        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8));

    check(cublasLtMatmul(lt, matmul_desc,
                         &alpha, A, layout_A,
                                 B, layout_B,
                         &beta,  C, layout_C,
                                 C, layout_C,
                         &hr.algo,
                         ws_tensor.data_ptr(), kWsBytes,
                         stream), "cublasLtMatmul: ");

    cleanup();
}

}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLAS LT Forward: Y = X @ W^T + bias  (fused, single kernel)
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

    auto input_c  = input.contiguous();
    auto weight_c = weight.contiguous();
    auto output   = torch::empty({M, N}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const float* bias_ptr = nullptr;
    torch::Tensor bias_c;
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.size(0) == N, "bias size mismatch");
        bias_c   = bias.contiguous();
        bias_ptr = bias_c.data_ptr<float>();
    }

    // Y = X @ W^T + bias in cuBLAS col-major:
    //   Y[N,M] = W[N,K](T) @ X[K,M](N)
    //   W[N,K] row-major -> stored [K,N] col-major, lda=K, opA=CUBLAS_OP_T
    //   X[M,K] row-major -> stored [K,M] col-major, ldb=K, opB=CUBLAS_OP_N
    //   Y[M,N] row-major -> stored [N,M] col-major, ldc=N
    run_cublaslt_gemm_bias(
        N, M, K,
        1.0f, 0.0f,
        weight_c.data_ptr<float>(), K, CUBLAS_OP_T,
        input_c.data_ptr<float>(),  K, CUBLAS_OP_N,
        output.data_ptr<float>(),   N,
        bias_ptr,
        stream);

    return output;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLAS LT Backward
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
    auto input_c       = input.contiguous();
    auto weight_c      = weight.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // ---- grad_input = dY @ W -> [M, K] ----
    // Row-major: dX[M,K] = dY[M,N] @ W[N,K]
    // cuBLAS col-major: dX[K,M] = W^T[K,N] @ dY[N,M]
    //   W[N,K] row -> lda=K, opA=N  => W[K,N] col, used as W (not transposed in col-major sense)
    //   dY[M,N] row -> ldb=N, opB=N
    //   dX[M,K] row -> ldc=K
    auto grad_input = torch::empty({M, K}, input.options());
    run_cublaslt_gemm_bias(
        K, M, N,
        1.0f, 0.0f,
        weight_c.data_ptr<float>(),      K, CUBLAS_OP_N,
        grad_output_c.data_ptr<float>(), N, CUBLAS_OP_N,
        grad_input.data_ptr<float>(),    K,
        nullptr, stream);

    // ---- grad_weight = dY^T @ X -> [N, K] ----
    // dW[N,K] = dY^T[N,M] @ X[M,K]
    // cuBLAS col-major: dW[K,N] = X[K,M](N) @ dY[N,M](T)
    //   X[M,K] row -> lda=K, opA=N
    //   dY[M,N] row -> ldb=N, opB=T -> (M,N) transposed
    //   dW[N,K] row -> ldc=K
    auto grad_weight = torch::empty({N, K}, weight.options());
    run_cublaslt_gemm_bias(
        K, N, M,
        1.0f, 0.0f,
        input_c.data_ptr<float>(),       K, CUBLAS_OP_N,
        grad_output_c.data_ptr<float>(), N, CUBLAS_OP_T,
        grad_weight.data_ptr<float>(),   K,
        nullptr, stream);

    // ---- grad_bias = sum(dY, dim=0) -> [N] ----
    auto grad_bias = grad_output.sum(0);

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}
