/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/detail/cluster.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/mma_sm100.hpp"
#include "cutlass/trace.h"
#include "cutlass/kernel_hardware_info.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////
// Dispatch policy for persistent-B optimized variant.
// Inherits from the standard FastF32 dispatch policy so the kernel template
// (GemmUniversal / sm100_gemm_tma_warpspecialized_input_transform) resolves identically.
// The distinct type lets us provide a separate CollectiveMma specialization.
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {

template <
  int Load2TransformPipelineStageCount_,
  int Transform2MmaPipelineStageCount_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  int NumBandsToCompute_,
  int ScalingFactor_,
  int AccPromotionInterval_,
  class ClusterShape_ = cute::Shape<cute::_1, cute::_1, cute::_1>,
  class AccumulatorCopyAtom_ = cute::SM100_TMEM_LOAD_32dp32b32x
>
struct MainloopSm100TmaUmmaWarpSpecializedFastF32PersistentB
    : MainloopSm100TmaUmmaWarpSpecializedFastF32<
        Load2TransformPipelineStageCount_,
        Transform2MmaPipelineStageCount_,
        SchedulerPipelineStageCount_,
        AccumulatorPipelineStageCount_,
        NumBandsToCompute_,
        ScalingFactor_,
        AccPromotionInterval_,
        ClusterShape_,
        AccumulatorCopyAtom_>
{};

} // namespace cutlass::gemm

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop for FastF32 Kernels with Persistent-B Optimization
template <
  int Load2TransformPipelineStageCount_,
  int Transform2MmaPipelineStageCount_,
  int SchedulerPipelineStageCount_,
  int AccumulatorPipelineStageCount_,
  int NumBandsToCompute_,
  int ScalingFactor_,
  int AccPromotionInterval_,
  class AccumulatorCopyAtom_,
  class ClusterShape,
  class TileShape_,
  class StrideA_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomsA_,
  class CopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomsB_,
  class CopyAtomsB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm100TmaUmmaWarpSpecializedFastF32PersistentB<
      Load2TransformPipelineStageCount_,
      Transform2MmaPipelineStageCount_,
      SchedulerPipelineStageCount_,
      AccumulatorPipelineStageCount_,
      NumBandsToCompute_,
      ScalingFactor_,
      AccPromotionInterval_,
      ClusterShape,
      AccumulatorCopyAtom_>,
    TileShape_,
    float,
    StrideA_,
    float,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomsA_,
    CopyAtomsA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomsB_,
    CopyAtomsB_,
    TransformB_>
{
  //
  // Type Aliases
  //

  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma_::ThrLayoutVMNK{})), _1, _1>;
  using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedFastF32PersistentB<
                            Load2TransformPipelineStageCount_,
                            Transform2MmaPipelineStageCount_,
                            SchedulerPipelineStageCount_,
                            AccumulatorPipelineStageCount_,
                            NumBandsToCompute_,
                            ScalingFactor_,
                            AccPromotionInterval_,
                            ClusterShape,
                            AccumulatorCopyAtom_>;
  using TileShape = TileShape_;
  using TiledMma = TiledMma_;
  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  using CtaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using CtaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

  using ElementA = float;
  using PackedElementA = float2;
  using StrideA = StrideA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using PackedElementAMma = uint32_t;
  using ElementB = float;
  using PackedElementB = float2;
  using StrideB = StrideB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using PackedElementBMma = uint32_t;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomsA = SmemLayoutAtomsA_;
  using SmemLayoutAtomsB = SmemLayoutAtomsB_;
  using CopyAtomsA = CopyAtomsA_;
  using CopyAtomsB = CopyAtomsB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(cute::is_same_v<ElementA, float>, "Input type A should be float");
  static_assert(cute::is_same_v<ElementB, float>, "Input type B should be float");
  static_assert(cute::is_same_v<ElementAMma, cutlass::bfloat16_t>, "Compute type A should be cutlass::bfloat16_t");
  static_assert(cute::is_same_v<ElementBMma, cutlass::bfloat16_t>, "Compute type B should be cutlass::bfloat16_t");

  using Load2TransformPipeline = cutlass::PipelineTmaTransformAsync<
                             DispatchPolicy::Load2TransformPipelineStageCount,
                             AtomThrShapeMNK>;
  using Load2TransformPipelineState = typename Load2TransformPipeline::PipelineState;

  using Transform2MmaPipeline = cutlass::PipelineUmmaConsumerAsync<
                              DispatchPolicy::Transform2MmaPipelineStageCount,
                              AtomThrShapeMNK>;
  using Transform2MmaPipelineState = typename Transform2MmaPipeline::PipelineState;

  using Mma2AccumPipeline =  cutlass::PipelineUmmaAsync<
                              DispatchPolicy::Schedule::AccumulatorPipelineStageCount,
                              AtomThrShapeMNK>;
  using Mma2AccumPipelineState = typename Mma2AccumPipeline::PipelineState;

  static constexpr uint32_t NumTransformationThreads = 128;
  static constexpr uint32_t NumAccumThreads = 128;

  static constexpr uint32_t GenericRegisterRequirement = 64;
  static constexpr uint32_t TransformRegisterRequirement = 184;
  static constexpr uint32_t AccumRegisterRequirement = 256;

  constexpr static int NumComputeMtxs = 3;
  constexpr static int NumBandsToCompute = DispatchPolicy::NumBandsToCompute;
  constexpr static int ScalingFactor = DispatchPolicy::ScalingFactor;
  constexpr static int AccPromotionInterval = DispatchPolicy::AccPromotionInterval;
  constexpr static int AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  constexpr static int StagesPerTile = size<2>(CtaShapeA_MK{}) / DispatchPolicy::AccPromotionInterval;
  constexpr static int NumBandsMax = 5;
  static_assert(NumBandsToCompute <= NumBandsMax && NumBandsToCompute >= 3, "NumBandsToCompute should be less than maximum number of bands");

  // Max k-tiles that can be cached for B (K=256, K_tile=32 → 8 tiles)
  constexpr static int MaxPersistentBtiles = 8;

  using AccumulatorCopyAtom = typename DispatchPolicy::AccumulatorCopyAtom;

  static_assert((NumBandsToCompute == 5 || NumBandsToCompute == 4 || NumBandsToCompute == 3),
                 "9xBF16 with 5/4/3 Bands are supported");

  using SmemLayoutAtomA = typename SmemLayoutAtomsA::InputLayoutAtom;
  using SmemLayoutAtomACompute = typename SmemLayoutAtomsA::ComputeLayoutAtom;
  using SmemLayoutAtomB = typename SmemLayoutAtomsB::InputLayoutAtom;
  using SmemLayoutAtomBCompute = typename SmemLayoutAtomsB::ComputeLayoutAtom;

  using InputCopyAtomA = typename CopyAtomsA::InputCopyAtom;
  using ComputeCopyAtomA = typename CopyAtomsA::ComputeCopyAtom;
  using InputCopyAtomB = typename CopyAtomsB::InputCopyAtom;
  using ComputeCopyAtomB = typename CopyAtomsB::ComputeCopyAtom;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(CtaShapeA_MK{}) * size<1>(CtaShapeA_MK{})) % size<0>(SmemLayoutAtomACompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");
  static_assert(((size<0,1>(CtaShapeA_MK{}) * size<2>(CtaShapeA_MK{})) % size<1>(SmemLayoutAtomACompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(((size<0,0>(CtaShapeB_NK{}) * size<1>(CtaShapeB_NK{})) % size<0>(SmemLayoutAtomBCompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");
  static_assert(((size<0,1>(CtaShapeB_NK{}) * size<2>(CtaShapeB_NK{})) % size<1>(SmemLayoutAtomBCompute{})) == 0, "SmemLayoutAtomCompute must evenly divide tile shape.");

  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(CtaShapeA_MK{}, Int<DispatchPolicy::Load2TransformPipelineStageCount>{}),
             (cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{})));

  using SmemLayoutACompute = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomACompute{},
      append(append(CtaShapeA_MK{}, Int<NumComputeMtxs>{}), Int<DispatchPolicy::Transform2MmaPipelineStageCount>{})));

  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(CtaShapeB_NK{}, Int<DispatchPolicy::Load2TransformPipelineStageCount>{}),
             (cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{})));

  using SmemLayoutBCompute = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomBCompute{},
      append(append(CtaShapeB_NK{}, Int<NumComputeMtxs>{}), Int<DispatchPolicy::Transform2MmaPipelineStageCount>{})));

  // Persistent B: (MMA, MMA_N, MMA_K, NumComputeMtxs, MaxPersistentBtiles)
  using SmemLayoutBComputePersist = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomBCompute{},
      append(append(CtaShapeB_NK{}, Int<NumComputeMtxs>{}), Int<MaxPersistentBtiles>{})));

  static_assert(DispatchPolicy::Load2TransformPipelineStageCount >= 2 && DispatchPolicy::Load2TransformPipelineStageCount >= 2,
                "Specialization requires Stages set to value 2 or more.");
  static_assert((cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value ||
                 cute::is_base_of<cute::UMMA::tmem_frg_base,      typename TiledMma::FrgTypeA>::value  ) &&
                 cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                 "MMA atom must A operand from SMEM or TMEM and B operand from SMEM for this mainloop.");
  static_assert((cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>),
                 "GmemTiledCopyA - invalid TMA copy atom specified.");
  static_assert((cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>),
                 "GmemTiledCopyB -  invalid TMA copy atom specified.");

  struct PipelineStorage {
    using Load2TransformPipelineStorage = typename Load2TransformPipeline::SharedStorage;
    alignas(16) Load2TransformPipelineStorage load2transform_pipeline;
    using Transform2MmaPipelineStorage = typename Transform2MmaPipeline::SharedStorage;
    alignas(16) Transform2MmaPipelineStorage transform2mma_pipeline;
    using Mma2AccumPipelineStorage = typename Mma2AccumPipeline::SharedStorage;
    alignas(16) Mma2AccumPipelineStorage mma2accum_pipeline;
  };

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      struct TensorStorageUntransformed {
        cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
        cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
      };

      struct TensorStorageTransformedAinSmem {
        alignas(1024) cute::ArrayEngine<ElementAMma, cute::cosize_v<SmemLayoutACompute>> smem_ACompute;
      };

      union TensorStorageTransformedAinTmem {
        alignas(1024) cute::ArrayEngine<ElementAMma, 1> smem_ACompute;
      };

      using TensorStorageTransformed = cute::conditional_t<
                                      cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value,
                                      TensorStorageTransformedAinSmem,
                                      TensorStorageTransformedAinTmem>;

      // Persistent B storage for caching transformed B across work tiles
      struct PersistentBStorage {
        alignas(1024) cute::ArrayEngine<ElementBMma, cute::cosize_v<SmemLayoutBComputePersist>> smem_BComputePersist;
      };

      struct PersistentBMeta {
        int32_t b_loaded;  // set by load warp: 1 = B was TMA-loaded this work tile, 0 = use persistent
      };

      TensorStorageUntransformed input;
      TensorStorageTransformed compute;
      PersistentBStorage persistent_b;
      PersistentBMeta persistent_b_meta;
    } tensors;

    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr uint32_t TmaTransactionBytesA =
    cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * size<2>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesB =
    cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * size<2>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));
  // Pipeline base expects A only; B is added dynamically via producer_expect_transaction
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesA;

  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
  };

  struct Params {
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                                                     make_tile(typename TiledMma::AtomThrID{})));

    using TMA_A = decltype(make_tma_atom_A_sm100<ElementA>(
        GmemTiledCopyA{},
        make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );
    using TMA_B = decltype(make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_A tma_load_a_fallback;
    TMA_B tma_load_b_fallback;
    dim3 cluster_shape_fallback;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
    : cluster_shape_(cluster_shape)
    , block_rank_in_cluster_(block_rank_in_cluster) {
    if constexpr (IsDynamicCluster) {
      const bool is_fallback_cluster = (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x &&
                                        cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
      observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
      observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
    }
    else {
      observed_tma_load_a_ = &params.tma_load_a;
      observed_tma_load_b_ = &params.tma_load_b;
    }
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{}) {
    (void) workspace;

    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor tensor_a = make_tensor(args.ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N,K,L), args.dB));

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));

    auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
    auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));

    typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<ElementA>(
        GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{}, TiledMma{}, cluster_layout_vmnk);

    typename Params::TMA_B tma_load_b = make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{}, tensor_b, SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{}, TiledMma{}, cluster_layout_vmnk);

    typename Params::TMA_A tma_load_a_fallback = make_tma_atom_A_sm100<ElementA>(
        GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{}, TiledMma{}, cluster_layout_vmnk_fallback);

    typename Params::TMA_B tma_load_b_fallback = make_tma_atom_B_sm100<ElementB>(
        GmemTiledCopyB{}, tensor_b, SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{}, TiledMma{}, cluster_layout_vmnk_fallback);

    return {
      tma_load_a, tma_load_b,
      tma_load_a_fallback, tma_load_b_fallback,
      hw_info.cluster_shape_fallback
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  CUTLASS_DEVICE static void
  prefetch_tma_descriptors(Params const& params) {
    if constexpr (IsDynamicCluster) {
      dim3 cs = cute::cluster_shape();
      const bool is_fallback_cluster = (cs.x == params.cluster_shape_fallback.x && cs.y == params.cluster_shape_fallback.y);
      if (is_fallback_cluster) {
        cute::prefetch_tma_descriptor(params.tma_load_a_fallback.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b_fallback.get_tma_descriptor());
      }
      else {
        cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
      }
    }
    else {
      cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    }
  }

  CUTLASS_DEVICE auto
  partition_accumulator_shape() {
    auto acc_shape = partition_shape_C(TiledMma{}, take<0,2>(TileShape{}));
    return acc_shape;
  }

  /// Load: conditionally loads B via TMA.  On the first work tile (or when
  /// the N/L coordinate changes) B is loaded alongside A; on subsequent tiles
  /// with the same N/L only A is loaded and the persistent B cache is reused.
  template <
    class GTensorA, class GTensorB,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class TileCoordMNKL,
    class KTileIterator
  >
  CUTLASS_DEVICE auto
  load(
      Params const& params,
      Load2TransformPipeline pipeline,
      Load2TransformPipelineState load2xform_pipeline_state,
      cute::tuple<GTensorA, GTensorB,
                  GTensorPartitionedA, GTensorPartitionedB,
                  STensorA, STensorB,
                  uint16_t, uint16_t> const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter, int k_tile_count) {

    auto [unused_gA, unused_gB,
          tAgA_mkl, tBgB_nkl, tAsA, tBsB,
          mcast_mask_a, mcast_mask_b] = load_inputs;

    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    int32_t current_n = static_cast<int32_t>(get<1>(cta_coord_mnkl));

    // The kernel calls load() twice per work tile (prologue + remaining).
    // Compute need_load_b on the first call and reuse on the second so that
    // (a) the b_loaded flag is written only once and (b) B TMA is issued for
    // ALL k-tiles of the work tile, not just the prologue.
    bool is_first_call = (load_call_count_ % 2 == 0);
    ++load_call_count_;
    if (is_first_call) {
      need_load_b_for_tile_ = current_n != last_loaded_n_;
      last_loaded_n_ = current_n;
      persistent_b_meta_ptr_->b_loaded = need_load_b_for_tile_ ? 1 : 0;
      __threadfence_block();
    }
    bool need_load_b = need_load_b_for_tile_;

    uint32_t skip_wait = (k_tile_count <= 0);
    auto pipeline_flag = pipeline.producer_try_acquire(load2xform_pipeline_state, skip_wait);

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      pipeline.producer_acquire(load2xform_pipeline_state, pipeline_flag);
      int write_stage = load2xform_pipeline_state.index();

      using BarrierType = typename Load2TransformPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(load2xform_pipeline_state);

      if (need_load_b) {
        cutlass::arch::ClusterTransactionBarrier::expect_transaction(tma_barrier, TmaTransactionBytesB);
      }

      ++load2xform_pipeline_state;
      skip_wait = (k_tile_count <= 1);
      pipeline_flag = pipeline.producer_try_acquire(load2xform_pipeline_state, skip_wait);

      copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
      if (need_load_b) {
        copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));
      }
      ++k_tile_iter;
    }
    return cute::make_tuple(load2xform_pipeline_state, k_tile_iter);
  }

  /// load_init: unchanged from original
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& params,
      TensorStorage& shared_storage) const {
    auto [gA_mkl, gB_nkl] = tile_input_tensors(params, problem_shape_MNKL);

    ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.input.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.input.smem_B.begin()), SmemLayoutB{});

    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));

    auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));

    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    return cute::make_tuple(
        gA_mkl, gB_nkl,
        tAgA_mkl, tBgB_nkl, tAsA, tBsB,
        mcast_mask_a, mcast_mask_b);
  }

  /// Transform: optimized — caches B on first work tile, reuses cache on subsequent tiles
  template<
    class KTileIterator, class Accumulator,
    class GTensorA, class DstCopyA, class SrcTensorA, class DstTensorA,
    class GTensorB,                 class SrcTensorB, class DstTensorB
  >
  CUTLASS_DEVICE auto
  transform(
      Load2TransformPipeline load2transform_pipeline,
      Load2TransformPipelineState load2transform_pipeline_consumer_state,
      Transform2MmaPipeline transform2mma_pipeline,
      Transform2MmaPipelineState transform2mma_pipeline_producer_state,
      Accumulator accumulators,
      cute::tuple<GTensorA, DstCopyA, SrcTensorA, DstTensorA,
                  GTensorB,           SrcTensorB, DstTensorB> input_operands,
      KTileIterator k_tile_iter, int k_tile_count) {

    static_assert(cute::is_same_v<ElementA, ElementB>, "ElementA and ElementB types should be the same.");
    static_assert(cute::is_same_v<ElementAMma, ElementBMma>, "ElementAMma and ElementBMma types should be the same.");

    cutlass::arch::NamedBarrier transform_bar(NumTransformationThreads, cutlass::arch::ReservedNamedBarriers::TransformBarrier);

    auto [unused_tAgA, dst_copy_A, tAsA, tAdACompute,
          unused_tBgB,             tBsB, tBsBPersist] = input_operands;

    auto tArA = make_tensor<ElementA>(tAsA(_,_,_,_,0).shape());
    auto tArACompute = make_tensor<ElementAMma>(tAsA(_,_,_,_,0).shape());

    auto tArA_x2 = recast<Array<ElementA,2>>(tArA);
    auto tArACompute_x2 = recast<Array<ElementAMma,2>>(tArACompute);

    constexpr int ScalingFactor_ = DispatchPolicy::ScalingFactor;
    constexpr float kResidualScale = (ScalingFactor_ != 0) ? float(1 << ScalingFactor_) : 1.0f;
    auto fused_residual = [](Array<float,2> const& input, Array<cutlass::bfloat16_t,2> const& truncated) {
      Array<float,2> r;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 2; ++i) {
        r[i] = (input[i] - static_cast<float>(truncated[i])) * kResidualScale;
      }
      return r;
    };

    auto* meta_ptr = persistent_b_meta_ptr_;

    bool b_was_loaded = true;
    bool b_checked = false;

    uint32_t skip_wait = (k_tile_count <= 0);
    auto load2transform_flag = load2transform_pipeline.consumer_try_wait(load2transform_pipeline_consumer_state, skip_wait);
    auto transform2mma_flag = transform2mma_pipeline.producer_try_acquire(transform2mma_pipeline_producer_state, skip_wait);

    int abs_k_tile = 0;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      load2transform_pipeline.consumer_wait(load2transform_pipeline_consumer_state, load2transform_flag);
      transform2mma_pipeline.producer_acquire(transform2mma_pipeline_producer_state, transform2mma_flag);

      int load2transform_consumer_index = load2transform_pipeline_consumer_state.index();

      auto curr_load2transform_pipeline_consumer_state = load2transform_pipeline_consumer_state;
      auto curr_transform2mma_pipeline_producer_state = transform2mma_pipeline_producer_state;

      if (!b_checked) {
        b_checked = true;
        b_was_loaded = (meta_ptr->b_loaded == 1);
      }

      copy(AutoVectorizingCopy{}, tAsA(_,_,_,_,load2transform_consumer_index), tArA);

      if (b_was_loaded && abs_k_tile < MaxPersistentBtiles) {
        auto tBrB = make_tensor<ElementB>(tBsB(_,_,_,_,0).shape());
        auto tBrBCompute = make_tensor<ElementBMma>(tBsB(_,_,_,_,0).shape());
        auto tBrB_x2 = recast<Array<ElementB,2>>(tBrB);
        auto tBrBCompute_x2 = recast<Array<ElementBMma,2>>(tBrBCompute);

        copy(AutoVectorizingCopy{}, tBsB(_,_,_,_,load2transform_consumer_index), tBrB);

        CUTE_UNROLL
        for (int comp_mtx_index = 0; comp_mtx_index < NumComputeMtxs; ++comp_mtx_index) {
          cute::transform(tBrB_x2, tBrBCompute_x2, cutlass::NumericArrayConverter<ElementBMma, ElementB, 2, cutlass::FloatRoundStyle::round_to_nearest_satfinite>::convert);
          copy(AutoVectorizingCopy{}, tBrBCompute, tBsBPersist(_,_,_,_,comp_mtx_index,abs_k_tile));

          if (comp_mtx_index < NumComputeMtxs - 1) {
            cute::transform(tBrB_x2, tBrBCompute_x2, tBrB_x2, fused_residual);
          }
        }
      }

      transform_bar.sync();
      load2transform_pipeline.consumer_release(curr_load2transform_pipeline_consumer_state);

      CUTE_UNROLL
      for (int comp_mtx_index = 0; comp_mtx_index < NumComputeMtxs; ++comp_mtx_index) {
        cute::transform(tArA_x2, tArACompute_x2, cutlass::NumericArrayConverter<ElementAMma, ElementA, 2, cutlass::FloatRoundStyle::round_to_nearest_satfinite>::convert);
        copy(dst_copy_A, tArACompute, tAdACompute(_,_,_,_,comp_mtx_index,transform2mma_pipeline_producer_state.index()));

        if (comp_mtx_index < NumComputeMtxs - 1) {
          cute::transform(tArA_x2, tArACompute_x2, tArA_x2, fused_residual);
        }
      }

      cutlass::arch::fence_view_async_shared();
      if constexpr (is_tmem<decltype(tAdACompute)>::value) {
        cutlass::arch::fence_view_async_tmem_store();
      }

      transform2mma_pipeline.producer_commit(curr_transform2mma_pipeline_producer_state);
      ++load2transform_pipeline_consumer_state;
      ++transform2mma_pipeline_producer_state;

      skip_wait = (k_tile_count <= 1);
      load2transform_flag = load2transform_pipeline.consumer_try_wait(load2transform_pipeline_consumer_state, skip_wait);
      transform2mma_flag = transform2mma_pipeline.producer_try_acquire(transform2mma_pipeline_producer_state, skip_wait);

      ++abs_k_tile;
    }

    return cute::make_tuple(load2transform_pipeline_consumer_state, transform2mma_pipeline_producer_state);
  }

  template<class ProblemShape_MNKL, class Accumulator>
  CUTLASS_DEVICE auto
  transform_init(
      Params const& params,
      ProblemShape_MNKL const& problem_shape_MNKL,
      Accumulator accumulators,
      TensorStorage& shared_storage) {

    if (threadIdx.x == 0) {
      shared_storage.persistent_b_meta.b_loaded = 0;
    }

    persistent_b_meta_ptr_ = &shared_storage.persistent_b_meta;
    last_loaded_n_ = -1;

    auto [gA_mkl, gB_nkl] = tile_input_tensors(params, problem_shape_MNKL);

    Tensor sA_orig = make_tensor(make_smem_ptr(shared_storage.input.smem_A.begin()), SmemLayoutA{});
    Tensor sA = as_position_independent_swizzle_tensor(sA_orig);
    Tensor sACompute = make_tensor(make_smem_ptr(shared_storage.compute.smem_ACompute.begin()), SmemLayoutACompute{});

    Tensor sB_orig = make_tensor(make_smem_ptr(shared_storage.input.smem_B.begin()), SmemLayoutB{});
    Tensor sB = as_position_independent_swizzle_tensor(sB_orig);

    // A: use standard setup_copy_ops for smem_ACompute pipeline
    auto setup_copy_ops_A = [&] (
        auto tensor_input,
        auto input_copy_atom,
        auto tensor_compute,
        auto make_fragment,
        auto compute_copy_atom) constexpr {
      auto fragment_compute = make_fragment(tensor_compute);
      if constexpr (cute::is_tmem<cute::remove_cvref_t<decltype(fragment_compute)>>::value) {
        Tensor tensor_input2x = [&] () constexpr {
        if constexpr (decltype(size<0,0>(fragment_compute) == Int<128>{} && size<0,0>(tensor_input) == Int<64>{})::value) {
          return make_tensor(tensor_input.data(),
                             logical_product(tensor_input.layout(),
                                             make_tile(make_tile(Layout<_2,_0>{},_),_,_,_)));
          }
          else {
            return tensor_input;
          }
        }();

        fragment_compute.data() = accumulators.data().get() + cutlass::detail::find_tmem_tensor_col_offset(accumulators);
        auto reg2tmem_tiled_copy = make_tmem_copy(compute_copy_atom, fragment_compute(_,_,_,0,0));
        auto thr_reg2tmem_tiled_copy = reg2tmem_tiled_copy.get_slice(threadIdx.x % NumTransformationThreads);
        auto partitioned_tensor_input = thr_reg2tmem_tiled_copy.partition_S(tensor_input2x);
        auto partitioned_tensor_compute = thr_reg2tmem_tiled_copy.partition_D(fragment_compute);
        return cute::make_tuple(reg2tmem_tiled_copy, partitioned_tensor_input, partitioned_tensor_compute);
      }
      else {
        auto tensor_compute_ind_sw = as_position_independent_swizzle_tensor(tensor_compute);
        auto reg2smem_tiled_copy = make_cotiled_copy(compute_copy_atom, Layout<Shape <_128,_8>, Stride<  _8,_1>>{},
                                                     tensor_compute(_,_,_,0,0).layout());

        auto thr_reg2smem_tiled_copy = reg2smem_tiled_copy.get_slice(threadIdx.x % NumTransformationThreads);
        auto partitioned_tensor_input = thr_reg2smem_tiled_copy.partition_S(tensor_input);
        auto partitioned_tensor_compute = thr_reg2smem_tiled_copy.partition_D(tensor_compute_ind_sw);

        return cute::make_tuple(AutoVectorizingCopy{}, partitioned_tensor_input, partitioned_tensor_compute);
      }
    };

    auto [dst_copy_A, tAsA, tAsACompute] =
        setup_copy_ops_A(sA, InputCopyAtomA{}, sACompute, [&](auto &arg) {return TiledMma::make_fragment_A(arg);}, ComputeCopyAtomA{});

    // B: single tiled copy for consistent input/compute partitioning
    auto sBPersist_raw = make_tensor(make_smem_ptr(shared_storage.persistent_b.smem_BComputePersist.begin()), SmemLayoutBComputePersist{});
    auto sBPersist_ind_sw = as_position_independent_swizzle_tensor(sBPersist_raw);
    auto b_tiled_copy = make_cotiled_copy(
        ComputeCopyAtomB{},
        Layout<Shape<_128, _8>, Stride<_8, _1>>{},
        sBPersist_raw(_,_,_,0,0).layout());
    auto b_thr_copy = b_tiled_copy.get_slice(threadIdx.x % NumTransformationThreads);
    auto tBsB = b_thr_copy.partition_S(sB);
    auto tBsBPersist = b_thr_copy.partition_D(sBPersist_ind_sw);

    return cute::make_tuple(gA_mkl, dst_copy_A, tAsA, tAsACompute,
                            gB_nkl,             tBsB, tBsBPersist);
  }

  /// MMA: reads A from pipeline compute buffer, reads B always from persistent SMEM.
  template <
    class FrgEngine, class FrgLayout,
    class InputOperandsTuple
  >
  CUTLASS_DEVICE auto
  mma(
      Transform2MmaPipeline transform2mma_pipeline,
      Transform2MmaPipelineState transform2mma_pipeline_consumer_state,
      Mma2AccumPipeline mma2accum_pipeline,
      Mma2AccumPipelineState mma2accum_pipeline_producer_state,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      InputOperandsTuple const& input_operands,
      int k_tile_count
  ) {
    TiledMma tiled_mma;

    auto curr_transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state;
    auto next_transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state;
    uint32_t skip_wait = (k_tile_count <= 0);
    auto transform2mma_flag = transform2mma_pipeline.consumer_try_wait(next_transform2mma_pipeline_consumer_state, skip_wait);
    ++next_transform2mma_pipeline_consumer_state;

    auto const& tCrA         = get<0>(input_operands);
    auto const& tCrBPersist  = get<1>(input_operands);

    using ZeroScaler = cute::integral_constant<uint32_t, 0>;
    using Scaler = cute::integral_constant<uint32_t, ScalingFactor>;

    int remaining_accum_promotions = k_tile_count * StagesPerTile;
    uint32_t mma2accum_skip_wait = (remaining_accum_promotions <= 0);
    auto mma2accum_flag = mma2accum_pipeline.producer_try_acquire(mma2accum_pipeline_producer_state, mma2accum_skip_wait);

    int abs_k_tile = 0;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {

      transform2mma_pipeline.consumer_wait(curr_transform2mma_pipeline_consumer_state, transform2mma_flag);

      int transform2mma_pipeline_consumer_state_index = curr_transform2mma_pipeline_consumer_state.index();

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); k_block += DispatchPolicy::AccPromotionInterval, --remaining_accum_promotions) {
        mma2accum_pipeline.producer_acquire(mma2accum_pipeline_producer_state, mma2accum_flag);

        int mma2accum_pipeline_producer_state_index = mma2accum_pipeline_producer_state.index();
        auto tCtC = accumulators(_,_,_,mma2accum_pipeline_producer_state_index);
        auto curr_mma2accum_pipeline_producer_state = mma2accum_pipeline_producer_state;

        ++mma2accum_pipeline_producer_state;
        mma2accum_skip_wait = (remaining_accum_promotions <= 1);
        mma2accum_flag = mma2accum_pipeline.producer_try_acquire(mma2accum_pipeline_producer_state, mma2accum_skip_wait);

        auto tCrA0 = tCrA(_,_,_,0,transform2mma_pipeline_consumer_state_index);
        auto tCrA1 = tCrA(_,_,_,1,transform2mma_pipeline_consumer_state_index);
        auto tCrA2 = tCrA(_,_,_,2,transform2mma_pipeline_consumer_state_index);

        auto B0 = tCrBPersist(_,_,_,0,abs_k_tile);
        auto B1 = tCrBPersist(_,_,_,1,abs_k_tile);
        auto B2 = tCrBPersist(_,_,_,2,abs_k_tile);

        auto accumulate = UMMA::ScaleOut::Zero;
        // Band 5
        if constexpr (NumBandsToCompute == 5) {
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA2(_,_,k_block), B2(_,_,k_block), tCtC);
          accumulate = UMMA::ScaleOut::One;
          CUTLASS_PRAGMA_UNROLL
          for (int s = 1; s < DispatchPolicy::AccPromotionInterval; s++) {
            cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA2(_,_,k_block+s), B2(_,_,k_block+s), tCtC);
          }
        }
        // Band 4
        if constexpr (NumBandsToCompute >= 4) {
          cute::gemm(tiled_mma.with(accumulate, Scaler{}), tCrA1(_,_,k_block), B2(_,_,k_block), tCtC);
          accumulate = UMMA::ScaleOut::One;
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA2(_,_,k_block), B1(_,_,k_block), tCtC);
          CUTLASS_PRAGMA_UNROLL
          for (int s = 1; s < DispatchPolicy::AccPromotionInterval; s++) {
            cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA1(_,_,k_block+s), B2(_,_,k_block+s), tCtC);
            cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA2(_,_,k_block+s), B1(_,_,k_block+s), tCtC);
          }
        }
        // Band 3
        cute::gemm(tiled_mma.with(accumulate, Scaler{}), tCrA0(_,_,k_block), B2(_,_,k_block), tCtC);
        accumulate = UMMA::ScaleOut::One;
        cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA1(_,_,k_block), B1(_,_,k_block), tCtC);
        cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA2(_,_,k_block), B0(_,_,k_block), tCtC);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 1; s < DispatchPolicy::AccPromotionInterval; s++) {
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA0(_,_,k_block+s), B2(_,_,k_block+s), tCtC);
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA1(_,_,k_block+s), B1(_,_,k_block+s), tCtC);
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA2(_,_,k_block+s), B0(_,_,k_block+s), tCtC);
        }
        // Band 2
        cute::gemm(tiled_mma.with(accumulate, Scaler{}), tCrA0(_,_,k_block), B1(_,_,k_block), tCtC);
        cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA1(_,_,k_block), B0(_,_,k_block), tCtC);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 1; s < DispatchPolicy::AccPromotionInterval; s++) {
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA0(_,_,k_block+s), B1(_,_,k_block+s), tCtC);
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA1(_,_,k_block+s), B0(_,_,k_block+s), tCtC);
        }
        // Band 1
        cute::gemm(tiled_mma.with(accumulate, Scaler{}), tCrA0(_,_,k_block), B0(_,_,k_block), tCtC);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 1; s < DispatchPolicy::AccPromotionInterval; s++) {
          cute::gemm(tiled_mma.with(accumulate, ZeroScaler{}), tCrA0(_,_,k_block+s), B0(_,_,k_block+s), tCtC);
        }

        mma2accum_pipeline.producer_commit(curr_mma2accum_pipeline_producer_state);
      }

      transform2mma_pipeline.consumer_release(curr_transform2mma_pipeline_consumer_state);

      skip_wait = (k_tile_count <= 1);
      transform2mma_flag = transform2mma_pipeline.consumer_try_wait(next_transform2mma_pipeline_consumer_state, skip_wait);

      curr_transform2mma_pipeline_consumer_state = next_transform2mma_pipeline_consumer_state;
      ++next_transform2mma_pipeline_consumer_state;

      ++abs_k_tile;
    }
    return cute::make_tuple(curr_transform2mma_pipeline_consumer_state, mma2accum_pipeline_producer_state);
  }

  template<class FrgEngine, class FrgLayout>
  CUTLASS_DEVICE auto
  mma_init(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TensorStorage& shared_storage) const {
    TiledMma tiled_mma;

    auto get_tCrA = [&] () constexpr {
      if constexpr (cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value) {
        Tensor sACompute = make_tensor(make_smem_ptr(shared_storage.compute.smem_ACompute.begin()), SmemLayoutACompute{});
        return tiled_mma.make_fragment_A(sACompute);
      }
      else {
        auto tCrA = tiled_mma.make_fragment_A(shape(SmemLayoutACompute{}));
        tCrA.data() = accumulators.data().get() + cutlass::detail::find_tmem_tensor_col_offset(accumulators);
        return tCrA;
      }
    };

    Tensor tCrA = get_tCrA();

    Tensor sBComputePersist = make_tensor(make_smem_ptr(shared_storage.persistent_b.smem_BComputePersist.begin()), SmemLayoutBComputePersist{});
    Tensor tCrBPersist = tiled_mma.make_fragment_B(sBComputePersist);

    return cute::make_tuple(tCrA, tCrBPersist);
  }

  template<class FrgEngine, class FrgLayout, class TmemCopyAtom, class EpilogueTile>
  CUTLASS_DEVICE auto
  accum_init(cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TmemCopyAtom tmem_cp_atom, EpilogueTile epilogue_tile) {
    Tensor tAcc = tensor<0>(accumulators(_,_,_,_0{}));
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});
    auto tiled_t2r = make_tmem_copy(tmem_cp_atom, tAcc_epi(_,_,_0{},_0{}));
    auto thread_t2r = tiled_t2r.get_slice(threadIdx.x % size(tiled_t2r));
    Tensor tTR_gC   = thread_t2r.partition_D(tAcc_epi);
    Tensor tTR_rAcc = make_tensor<ElementAccumulator>(shape(tTR_gC));
    Tensor tTR_rGlobAcc = make_tensor<ElementAccumulator>(shape(tTR_gC));
    Tensor tTR_rAcc_float2 = recast<Array<ElementAccumulator,2>>(tTR_rAcc);
    Tensor tTR_rGlobAcc_float2 = recast<Array<ElementAccumulator,2>>(tTR_rGlobAcc);

    Tensor tBulkAcc_epi = flat_divide(accumulators(make_coord(_,_),_0{},_0{}, _), EpilogueTile{});
    Tensor tTR_tBulkAcc = thread_t2r.partition_S(tBulkAcc_epi);
    return cute::make_tuple(tiled_t2r, thread_t2r, tTR_tBulkAcc, tTR_rAcc, tTR_rGlobAcc);
  }

  template<class TiledCopy, class ThrCopy, class AccumulatorTensor, class LocalAccFrg, class GlobalAccFrg>
  CUTLASS_DEVICE auto
  accum(cute::tuple<TiledCopy, ThrCopy, AccumulatorTensor, LocalAccFrg, GlobalAccFrg> accum_inputs,
        Mma2AccumPipeline mma2accum_pipeline,
        Mma2AccumPipelineState mma2accum_pipeline_consumer_state,
        int k_tile_count) {
    auto [tiled_t2r, thread_t2r, tTR_tBulkAcc,
          tTR_rAcc, tTR_rGlobAcc] = accum_inputs;

    Tensor tTR_rAcc_float2 = recast<Array<ElementAccumulator,2>>(tTR_rAcc);
    Tensor tTR_rGlobAcc_float2 = recast<Array<ElementAccumulator,2>>(tTR_rGlobAcc);

    CUTE_UNROLL
    for (int i = 0; i<size(tTR_rGlobAcc); i++) {
      tTR_rGlobAcc(i) = ElementAccumulator(0);
    }

    uint32_t skip_wait = 0;
    auto mma2accum_flag = mma2accum_pipeline.consumer_try_wait(mma2accum_pipeline_consumer_state, skip_wait);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_block = 0; k_block<StagesPerTile; k_block++) {
        int mma2accum_pipeline_consumer_state_index = mma2accum_pipeline_consumer_state.index();
        mma2accum_pipeline.consumer_wait(mma2accum_pipeline_consumer_state, mma2accum_flag);
        auto prev_state = mma2accum_pipeline_consumer_state;

        copy(tiled_t2r, tTR_tBulkAcc(_,_,_,_,_,mma2accum_pipeline_consumer_state_index), tTR_rAcc);
        cute::transform(tTR_rGlobAcc_float2, tTR_rAcc_float2, tTR_rGlobAcc_float2, cutlass::plus<Array<ElementAccumulator,2>>{});

        cutlass::arch::fence_view_async_tmem_load();
        mma2accum_pipeline.consumer_release(mma2accum_pipeline_consumer_state);

        ++mma2accum_pipeline_consumer_state;
        skip_wait = ((k_tile_count <= 1) && (k_block >= (StagesPerTile-1)));
        mma2accum_flag = mma2accum_pipeline.consumer_try_wait(mma2accum_pipeline_consumer_state, skip_wait);
      }
    }
    return cute::make_tuple(mma2accum_pipeline_consumer_state, tTR_rGlobAcc);
  }

protected:

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  constexpr auto
  tile_input_tensors(Params const& params, ProblemShape_MNKL const& problem_shape_MNKL) const {
    using X = cute::Underscore;
    auto [M,N,K,L] = problem_shape_MNKL;

    Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M,K,L));
    Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N,K,L));

    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  typename Params::TMA_A const* observed_tma_load_a_ = nullptr;
  typename Params::TMA_B const* observed_tma_load_b_ = nullptr;

  // Persistent B state
  typename SharedStorage::TensorStorage::PersistentBMeta* persistent_b_meta_ptr_ = nullptr;
  int32_t last_loaded_n_ = -1;
  int32_t load_call_count_ = 0;
  bool need_load_b_for_tile_ = true;

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
