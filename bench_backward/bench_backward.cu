/*
 * bench_backward.cu — Backward GEMM benchmark for siblas-molfm
 *
 * Tests the two backward-pass GEMM operations used in linear layer autograd:
 *   GemmRR: dX = dY @ W      (A=RowMajor, B=RowMajor -> C=RowMajor)
 *   GemmNT: dW = dY^T @ X   (A=ColMajor, B=RowMajor -> C=RowMajor)
 *
 * Compares CUTLASS BF16x6 vs cuBLAS BF16x9 vs cuBLAS TF32 vs cuBLAS FP32 (reference).
 *
 * Build:
 *   mkdir build && cd build && cmake .. && make -j && ./bench_backward
 */

#include <cuda_runtime.h>
#include <curand.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ── Error macros ──

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e) { fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t _cs=(x); if(_cs) { fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,(int)_cs); exit(1); } } while(0)
#define CURAND_CHECK(x) do { curandStatus_t _rs=(x); if(_rs) { fprintf(stderr,"cuRAND %s:%d: %d\n",__FILE__,__LINE__,(int)_rs); exit(1); } } while(0)

// ── CUTLASS kernel configurations (matching siblas.cu) ──

using ArchTag        = cutlass::arch::Sm100;
using OperatorClass  = cutlass::arch::OpClassTensorOp;
constexpr int Align  = 128 / cutlass::sizeof_bits<float>::value;  // 4

using ClusterShape   = Shape<_2,_1,_1>;
using MmaTileShape   = Shape<_256,_128,_16>;

using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmFastFP32Sm100;

// ── GemmRR: A=RowMajor, B=RowMajor -> C=RowMajor  (backward dX = dY @ W) ──

using LayoutA_RR = cutlass::layout::RowMajor;
using LayoutB_RR = cutlass::layout::RowMajor;
using LayoutC_RR = cutlass::layout::RowMajor;

using EpiRR = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, float, LayoutC_RR, Align, float, LayoutC_RR, Align,
    cutlass::epilogue::FastF32NoSmemWarpSpecialized2Sm>::CollectiveOp;

using MainRR = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    float, LayoutA_RR, Align,
    float, LayoutB_RR, Align,
    float, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename EpiRR::SharedStorage))>,
    MainloopSchedule>::CollectiveOp;

using GemmRR = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, MainRR, EpiRR, void>>;

// ── GemmNT: A=ColMajor, B=RowMajor -> C=RowMajor  (backward dW = dY^T @ X) ──

using LayoutA_NT = cutlass::layout::ColumnMajor;
using LayoutB_NT = cutlass::layout::RowMajor;
using LayoutC_NT = cutlass::layout::RowMajor;

using EpiNT = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, float, LayoutC_NT, Align, float, LayoutC_NT, Align,
    cutlass::epilogue::FastF32NoSmemWarpSpecialized2Sm>::CollectiveOp;

using MainNT = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    float, LayoutA_NT, Align,
    float, LayoutB_NT, Align,
    float, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename EpiNT::SharedStorage))>,
    MainloopSchedule>::CollectiveOp;

using GemmNT = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, MainNT, EpiNT, void>>;

// ── Generic CUTLASS runner (pre-initialized, bench only measures kernel launch) ──

template <typename GemmType>
struct CutlassRunner {
    GemmType gemm;
    typename GemmType::Arguments args;
    void* ws;
    size_t ws_bytes;
    cudaStream_t s;

    void init(int M, int N, int K,
              const float* A, const float* B, float* D,
              void* ws_, size_t ws_bytes_, cudaStream_t s_) {
        ws = ws_; ws_bytes = ws_bytes_; s = s_;
        using StrA = typename GemmType::GemmKernel::StrideA;
        using StrB = typename GemmType::GemmKernel::StrideB;
        using StrD = typename GemmType::GemmKernel::StrideD;
        auto sA = cutlass::make_cute_packed_stride(StrA{}, {M, K, 1});
        auto sB = cutlass::make_cute_packed_stride(StrB{}, {N, K, 1});
        auto sD = cutlass::make_cute_packed_stride(StrD{}, {M, N, 1});
        args = typename GemmType::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
            {A, sA, B, sB}, {{1.0f, 0.0f}, D, sD, D, sD}};
        if (GemmType::get_workspace_size(args) > ws_bytes)
            throw std::runtime_error("CUTLASS workspace too small");
        auto ok = [](cutlass::Status st, const char* msg) {
            if (st != cutlass::Status::kSuccess)
                throw std::runtime_error(std::string(msg) + cutlass::cutlassGetStatusString(st));
        };
        ok(gemm.can_implement(args), "can_implement: ");
        ok(gemm.initialize(args, ws, s), "initialize: ");
    }

    void run() { gemm.run(s); }
};

// ── cuBLAS LT runner (pre-initialized) ──

constexpr size_t kWS = 32 * 1024 * 1024;

struct CublasLtRunner {
    cublasLtHandle_t lt;
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t Al, Bl, Dl;
    cublasLtMatmulHeuristicResult_t hr;
    const float* A; const float* B; float* D;
    void* ws;
    cudaStream_t s;
    float alpha = 1.0f, beta = 0.0f;
    int algo_id = -1;

    // m, n, k are cuBLAS column-major dimensions (i.e. rows of op(A) and op(B))
    // opA, opB: CUBLAS_OP_N or CUBLAS_OP_T
    void init(cublasLtHandle_t lt_,
              cublasComputeType_t compute_type,
              int m, int n, int k,
              const float* A_, int lda, cublasOperation_t opA,
              const float* B_, int ldb, cublasOperation_t opB,
              float* D_, int ldc,
              void* ws_, cudaStream_t s_) {
        lt = lt_; A = A_; B = B_; D = D_; ws = ws_; s = s_;

        CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, compute_type, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

        int rows_A = (opA == CUBLAS_OP_N) ? m : k;
        int cols_A = (opA == CUBLAS_OP_N) ? k : m;
        int rows_B = (opB == CUBLAS_OP_N) ? k : n;
        int cols_B = (opB == CUBLAS_OP_N) ? n : k;
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Al, CUDA_R_32F, rows_A, cols_A, lda));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bl, CUDA_R_32F, rows_B, cols_B, ldb));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Dl, CUDA_R_32F, m, n, ldc));

        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        size_t wsz = kWS;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));

        int nr = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, Al, Bl, Dl, Dl, pref, 1, &hr, &nr));
        if (!nr) throw std::runtime_error("cublasLt: no algorithm found");
        cublasLtMatmulPreferenceDestroy(pref);

        int id = -1; size_t ret = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&hr.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &ret);
        algo_id = id;
    }

    void run() {
        CUBLAS_CHECK(cublasLtMatmul(lt, op, &alpha, A, Al, B, Bl, &beta, D, Dl, D, Dl, &hr.algo, ws, kWS, s));
    }

    void destroy() {
        cublasLtMatrixLayoutDestroy(Dl);
        cublasLtMatrixLayoutDestroy(Bl);
        cublasLtMatrixLayoutDestroy(Al);
        cublasLtMatmulDescDestroy(op);
    }
};

// ── Benchmark helper ──

double bench(std::function<void()> fn, cudaStream_t s, int warmup = 20, int reps = 100) {
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaStreamSynchronize(s));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, s));
    for (int i = 0; i < reps; i++) fn();
    CUDA_CHECK(cudaEventRecord(t1, s));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double)ms / reps * 1000.0;  // avg us per iteration
}

// ── Correctness verification ──

struct VerifyResult {
    double max_abs_err;
    double max_rel_err;
    double rmse;
};

// Both matrices are row-major M×N
VerifyResult verify(const float* d_test, const float* d_ref, int M, int N) {
    size_t elems = (size_t)M * N;
    std::vector<float> h_test(elems), h_ref(elems);
    CUDA_CHECK(cudaMemcpy(h_test.data(), d_test, elems * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(),  d_ref,  elems * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs = 0, max_rel = 0, sum_sq = 0;
    for (size_t i = 0; i < elems; i++) {
        double diff    = std::abs((double)h_test[i] - (double)h_ref[i]);
        double ref_abs = std::abs((double)h_ref[i]);
        sum_sq += diff * diff;
        if (diff > max_abs) max_abs = diff;
        if (ref_abs > 1e-12 && diff / ref_abs > max_rel) max_rel = diff / ref_abs;
    }
    return {max_abs, max_rel, std::sqrt(sum_sq / elems)};
}

// ── One-shot CUTLASS run for correctness (no pre-init overhead) ──
template <typename GemmType>
void run_cutlass_once(int M, int N, int K,
                      const float* A, const float* B, float* D,
                      void* ws, size_t ws_bytes, cudaStream_t s) {
    CutlassRunner<GemmType> r;
    r.init(M, N, K, A, B, D, ws, ws_bytes, s);
    r.run();
}

// ── Print helpers ──

void print_sep() {
    printf("%-26s | %12s | %12s | %12s | %12s | %8s | %8s | %8s | %8s %8s %8s\n",
           "---------------------", "------------", "--------------", "------------",
           "------------", "--------", "--------", "--------",
           "--------", "----------", "--------");
}

// ── main ──

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)  CUDA %d.%d\n\n",
           prop.name, prop.major, prop.minor, CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);

    cublasLtHandle_t lt; CUBLAS_CHECK(cublasLtCreate(&lt));
    void* ws; CUDA_CHECK(cudaMalloc(&ws, kWS));

    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, 42));

    // Problem sizes matching bench_gemm.cu and bench_linear.py
    // For a linear layer (M, K, N): input[M,K], weight[N,K], output[M,N]
    // Backward:
    //   dX = dY[M,N] @ W[N,K]  -> GemmRR(M, K, N)
    //   dW = dY^T[N,M] @ X[M,K] -> GemmNT(N, K, M)
    struct Size { int M, K, N; };
    std::vector<Size> sizes = {
        {10000, 256, 768}, {10000, 256, 512}, {10000, 256, 256}, {10000, 256, 128},
        {16384, 256, 256}, {24576, 256, 256}, {32768, 256, 256},
        {40000, 256, 256}, {65536, 256, 256},
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Section 1: GemmRR — dX = dY @ W  (A=RowMajor, B=RowMajor)
    //   dY[M,N] x W[N,K] -> dX[M,K]
    //   GEMM problem: M x K x N  (output M×K, contraction over N)
    // ═══════════════════════════════════════════════════════════════════════
    printf("=== Backward: dX = dY @ W  (GemmRR: A=RowMajor B=RowMajor -> C=RowMajor) ===\n\n");
    printf("%-26s | %12s | %12s | %12s | %12s | %8s | %8s | %8s | %8s %10s %8s\n",
           "Problem (M,K,N)", "CUTLASS(us)", "cBL-BF16x9(us)", "cBL-TF32(us)", "cBL-FP32(us)",
           "CUT/FP32", "BF16x9/FP32", "TF32/FP32",
           "CUT-Err", "BF16x9-Err", "TF32-Err");
    print_sep();

    struct TflopsRow { std::string label; double cut, bf16, tf32, fp32; };
    std::vector<TflopsRow> rows_rr, rows_nt;

    for (auto& sz : sizes) {
        int M = sz.M, K = sz.K, N = sz.N;
        // Allocate: dY[M,N], W[N,K], dX_*[M,K]
        float *dDY, *dW, *dDX_cut, *dDX_bf16, *dDX_tf32, *dDX_ref;
        CUDA_CHECK(cudaMalloc(&dDY,     (size_t)M*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dW,      (size_t)N*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dDX_cut, (size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dDX_bf16,(size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dDX_tf32,(size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dDX_ref, (size_t)M*K*sizeof(float)));

        CURAND_CHECK(curandGenerateUniform(rng, dDY, (size_t)M*N));
        CURAND_CHECK(curandGenerateUniform(rng, dW,  (size_t)N*K));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Correctness: CUTLASS GemmRR(M, K, N): dX[M,K] = dY[M,N] @ W[N,K]
        // In CUTLASS terms: M_out=M, N_out=K, K_inner=N
        CUDA_CHECK(cudaMemset(dDX_cut,  0, (size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMemset(dDX_bf16, 0, (size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMemset(dDX_tf32, 0, (size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMemset(dDX_ref,  0, (size_t)M*K*sizeof(float)));

        run_cutlass_once<GemmRR>(M, K, N, dDY, dW, dDX_cut, ws, kWS, stream);

        // cuBLAS LT: dX[M,K] = dY[M,N] @ W[N,K]
        // col-major view: dX^T[K,M] = W^T[K,N] @ dY^T[N,M]
        // cublasLt(m=K, n=M, k=N): A=W^T[K,N](opN), B=dY^T[N,M](opN)
        // W[N,K] row -> W^T[K,N] col, lda=K
        // dY[M,N] row -> dY^T[N,M] col, ldb=N
        // dX[M,K] row -> dX^T[K,M] col, ldc=K
        {
            CublasLtRunner g;
            g.init(lt, CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
                   K, M, N,
                   dW, K, CUBLAS_OP_N,
                   dDY, N, CUBLAS_OP_N,
                   dDX_bf16, K, ws, stream);
            g.run(); g.destroy();
        }
        {
            CublasLtRunner g;
            g.init(lt, CUBLAS_COMPUTE_32F_FAST_TF32,
                   K, M, N,
                   dW, K, CUBLAS_OP_N,
                   dDY, N, CUBLAS_OP_N,
                   dDX_tf32, K, ws, stream);
            g.run(); g.destroy();
        }
        {
            CublasLtRunner g;
            g.init(lt, CUBLAS_COMPUTE_32F_PEDANTIC,
                   K, M, N,
                   dW, K, CUBLAS_OP_N,
                   dDY, N, CUBLAS_OP_N,
                   dDX_ref, K, ws, stream);
            g.run(); g.destroy();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto vr_cut  = verify(dDX_cut,  dDX_ref, M, K);
        auto vr_bf16 = verify(dDX_bf16, dDX_ref, M, K);
        auto vr_tf32 = verify(dDX_tf32, dDX_ref, M, K);

        // Performance: pre-init runners
        CutlassRunner<GemmRR> g_cut;
        g_cut.init(M, K, N, dDY, dW, dDX_cut, ws, kWS, stream);

        CublasLtRunner g_bf16, g_tf32, g_fp32;
        g_bf16.init(lt, CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
                    K, M, N, dW, K, CUBLAS_OP_N, dDY, N, CUBLAS_OP_N, dDX_bf16, K, ws, stream);
        g_tf32.init(lt, CUBLAS_COMPUTE_32F_FAST_TF32,
                    K, M, N, dW, K, CUBLAS_OP_N, dDY, N, CUBLAS_OP_N, dDX_tf32, K, ws, stream);
        g_fp32.init(lt, CUBLAS_COMPUTE_32F_PEDANTIC,
                    K, M, N, dW, K, CUBLAS_OP_N, dDY, N, CUBLAS_OP_N, dDX_ref,  K, ws, stream);

        auto t_cut  = bench([&]{ g_cut.run(); }, stream);
        auto t_bf16 = bench([&]{ g_bf16.run(); }, stream);
        auto t_tf32 = bench([&]{ g_tf32.run(); }, stream);
        auto t_fp32 = bench([&]{ g_fp32.run(); }, stream);

        g_bf16.destroy(); g_tf32.destroy(); g_fp32.destroy();

        double flops = 2.0 * M * K * N;
        double tfl_cut  = flops / (t_cut  * 1e-6) / 1e12;
        double tfl_bf16 = flops / (t_bf16 * 1e-6) / 1e12;
        double tfl_tf32 = flops / (t_tf32 * 1e-6) / 1e12;
        double tfl_fp32 = flops / (t_fp32 * 1e-6) / 1e12;

        char label[48]; snprintf(label, sizeof(label), "M=%d,K=%d,N=%d", M, K, N);
        printf("%-26s | %12.1f | %12.1f | %12.1f | %12.1f | %7.2fx | %7.2fx | %7.2fx | %8.1e %10.1e %8.1e  algo: bf16=%d tf32=%d fp32=%d\n",
               label, t_cut, t_bf16, t_tf32, t_fp32,
               t_fp32/t_cut, t_fp32/t_bf16, t_fp32/t_tf32,
               vr_cut.max_abs_err, vr_bf16.max_abs_err, vr_tf32.max_abs_err,
               g_bf16.algo_id, g_tf32.algo_id, g_fp32.algo_id);

        rows_rr.push_back({label, tfl_cut, tfl_bf16, tfl_tf32, tfl_fp32});

        CUDA_CHECK(cudaFree(dDY)); CUDA_CHECK(cudaFree(dW));
        CUDA_CHECK(cudaFree(dDX_cut)); CUDA_CHECK(cudaFree(dDX_bf16));
        CUDA_CHECK(cudaFree(dDX_tf32)); CUDA_CHECK(cudaFree(dDX_ref));
    }

    printf("\n%-26s | %12s | %12s | %12s | %12s\n",
           "Problem (M,K,N)", "CUTLASS", "cBL-BF16x9", "cBL-TF32", "cBL-FP32");
    printf("---------------------------+--------------+--------------+--------------+--------------\n");
    for (auto& r : rows_rr) {
        printf("%-26s | %9.2f TF | %9.2f TF | %9.2f TF | %9.2f TF\n",
               r.label.c_str(), r.cut, r.bf16, r.tf32, r.fp32);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Section 2: GemmNT — dW = dY^T @ X  (A=ColMajor, B=RowMajor)
    //   dY[M,N] treated as ColMajor = dY^T[N,M], X[M,K] RowMajor
    //   GEMM problem: N x K x M  (output N×K, contraction over M)
    // ═══════════════════════════════════════════════════════════════════════
    printf("\n\n=== Backward: dW = dY^T @ X  (GemmNT: A=ColMajor B=RowMajor -> C=RowMajor) ===\n\n");
    printf("%-26s | %12s | %12s | %12s | %12s | %8s | %8s | %8s | %8s %10s %8s\n",
           "Problem (M,K,N)", "CUTLASS(us)", "cBL-BF16x9(us)", "cBL-TF32(us)", "cBL-FP32(us)",
           "CUT/FP32", "BF16x9/FP32", "TF32/FP32",
           "CUT-Err", "BF16x9-Err", "TF32-Err");
    print_sep();

    for (auto& sz : sizes) {
        int M = sz.M, K = sz.K, N = sz.N;
        // Allocate: dY[M,N], X[M,K], dW_*[N,K]
        float *dDY, *dX, *dDW_cut, *dDW_bf16, *dDW_tf32, *dDW_ref;
        CUDA_CHECK(cudaMalloc(&dDY,     (size_t)M*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dX,      (size_t)M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dDW_cut, (size_t)N*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dDW_bf16,(size_t)N*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dDW_tf32,(size_t)N*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dDW_ref, (size_t)N*K*sizeof(float)));

        CURAND_CHECK(curandGenerateUniform(rng, dDY, (size_t)M*N));
        CURAND_CHECK(curandGenerateUniform(rng, dX,  (size_t)M*K));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemset(dDW_cut,  0, (size_t)N*K*sizeof(float)));
        CUDA_CHECK(cudaMemset(dDW_bf16, 0, (size_t)N*K*sizeof(float)));
        CUDA_CHECK(cudaMemset(dDW_tf32, 0, (size_t)N*K*sizeof(float)));
        CUDA_CHECK(cudaMemset(dDW_ref,  0, (size_t)N*K*sizeof(float)));

        // GemmNT(N, K, M): dW[N,K] = dY^T[N,M] @ X[M,K]
        // A=dY[M,N] stored as ColMajor = dY^T[N,M], B=X[M,K] RowMajor
        run_cutlass_once<GemmNT>(N, K, M, dDY, dX, dDW_cut, ws, kWS, stream);

        // cuBLAS LT: dW[N,K] = dY^T[N,M] @ X[M,K]
        // col-major: dW^T[K,N] = X^T[K,M] @ dY[M,N]  (since (dY^T@X)^T = X^T@dY)
        // cublasLt(m=K, n=N, k=M): A=X[M,K] row->col^T, opA=N; B=dY[M,N] row->col, opB=T
        // X[M,K] row -> lda=K, col-major [K,M], opA=N -> use as [K,M]
        // dY[M,N] row -> ldb=N, col-major [N,M], opB=T -> use as [N,M] transposed = [M,N]
        // dW[N,K] row -> ldc=K, col-major [K,N]
        {
            CublasLtRunner g;
            g.init(lt, CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
                   K, N, M,
                   dX,  K, CUBLAS_OP_N,
                   dDY, N, CUBLAS_OP_T,
                   dDW_bf16, K, ws, stream);
            g.run(); g.destroy();
        }
        {
            CublasLtRunner g;
            g.init(lt, CUBLAS_COMPUTE_32F_FAST_TF32,
                   K, N, M,
                   dX,  K, CUBLAS_OP_N,
                   dDY, N, CUBLAS_OP_T,
                   dDW_tf32, K, ws, stream);
            g.run(); g.destroy();
        }
        {
            CublasLtRunner g;
            g.init(lt, CUBLAS_COMPUTE_32F_PEDANTIC,
                   K, N, M,
                   dX,  K, CUBLAS_OP_N,
                   dDY, N, CUBLAS_OP_T,
                   dDW_ref, K, ws, stream);
            g.run(); g.destroy();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto vr_cut  = verify(dDW_cut,  dDW_ref, N, K);
        auto vr_bf16 = verify(dDW_bf16, dDW_ref, N, K);
        auto vr_tf32 = verify(dDW_tf32, dDW_ref, N, K);

        // Performance: pre-init runners
        CutlassRunner<GemmNT> g_cut;
        g_cut.init(N, K, M, dDY, dX, dDW_cut, ws, kWS, stream);

        CublasLtRunner g_bf16, g_tf32, g_fp32;
        g_bf16.init(lt, CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
                    K, N, M, dX, K, CUBLAS_OP_N, dDY, N, CUBLAS_OP_T, dDW_bf16, K, ws, stream);
        g_tf32.init(lt, CUBLAS_COMPUTE_32F_FAST_TF32,
                    K, N, M, dX, K, CUBLAS_OP_N, dDY, N, CUBLAS_OP_T, dDW_tf32, K, ws, stream);
        g_fp32.init(lt, CUBLAS_COMPUTE_32F_PEDANTIC,
                    K, N, M, dX, K, CUBLAS_OP_N, dDY, N, CUBLAS_OP_T, dDW_ref,  K, ws, stream);

        auto t_cut  = bench([&]{ g_cut.run(); }, stream);
        auto t_bf16 = bench([&]{ g_bf16.run(); }, stream);
        auto t_tf32 = bench([&]{ g_tf32.run(); }, stream);
        auto t_fp32 = bench([&]{ g_fp32.run(); }, stream);

        g_bf16.destroy(); g_tf32.destroy(); g_fp32.destroy();

        double flops = 2.0 * N * K * M;
        double tfl_cut  = flops / (t_cut  * 1e-6) / 1e12;
        double tfl_bf16 = flops / (t_bf16 * 1e-6) / 1e12;
        double tfl_tf32 = flops / (t_tf32 * 1e-6) / 1e12;
        double tfl_fp32 = flops / (t_fp32 * 1e-6) / 1e12;

        char label[48]; snprintf(label, sizeof(label), "M=%d,K=%d,N=%d", M, K, N);
        printf("%-26s | %12.1f | %12.1f | %12.1f | %12.1f | %7.2fx | %7.2fx | %7.2fx | %8.1e %10.1e %8.1e  algo: bf16=%d tf32=%d fp32=%d\n",
               label, t_cut, t_bf16, t_tf32, t_fp32,
               t_fp32/t_cut, t_fp32/t_bf16, t_fp32/t_tf32,
               vr_cut.max_abs_err, vr_bf16.max_abs_err, vr_tf32.max_abs_err,
               g_bf16.algo_id, g_tf32.algo_id, g_fp32.algo_id);

        rows_nt.push_back({label, tfl_cut, tfl_bf16, tfl_tf32, tfl_fp32});

        CUDA_CHECK(cudaFree(dDY)); CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dDW_cut)); CUDA_CHECK(cudaFree(dDW_bf16));
        CUDA_CHECK(cudaFree(dDW_tf32)); CUDA_CHECK(cudaFree(dDW_ref));
    }

    printf("\n%-26s | %12s | %12s | %12s | %12s\n",
           "Problem (M,K,N)", "CUTLASS", "cBL-BF16x9", "cBL-TF32", "cBL-FP32");
    printf("---------------------------+--------------+--------------+--------------+--------------\n");
    for (auto& r : rows_nt) {
        printf("%-26s | %9.2f TF | %9.2f TF | %9.2f TF | %9.2f TF\n",
               r.label.c_str(), r.cut, r.bf16, r.tf32, r.fp32);
    }

    CURAND_CHECK(curandDestroyGenerator(rng));
    CUDA_CHECK(cudaFree(ws));
    CUBLAS_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
