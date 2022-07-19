/** @file
 *****************************************************************************

 Declaration of interfaces for multi-exponentiation routines.

 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef MULTIEXP_HPP_
#define MULTIEXP_HPP_

#include <cstddef>
#include <vector>
#include "prover_config.hpp"
#include "cgbn_math.h"
#include "cgbn_fp.h"
#include "cgbn_alt_bn128_g1.h"
#include "cgbn_alt_bn128_g2.h"
#include "cgbn_multi_exp.h"
#include "cgbn_multi_exp_g2.h"
#include "low_func_gpu.h"
#include <cuda_runtime.h>

namespace libff {

enum multi_exp_method {
 /**
  * Naive multi-exponentiation individually multiplies each base by the
  * corresponding scalar and adds up the results.
  * multi_exp_method_naive uses opt_window_wnaf_exp for exponentiation,
  * while multi_exp_method_plain uses operator *.
  */
 multi_exp_method_naive,
 multi_exp_method_naive_plain,
 /**
  * A variant of the Bos-Coster algorithm [1],
  * with implementation suggestions from [2].
  *
  * [1] = Bos and Coster, "Addition chain heuristics", CRYPTO '89
  * [2] = Bernstein, Duif, Lange, Schwabe, and Yang, "High-speed high-security signatures", CHES '11
  */
 multi_exp_method_bos_coster,
 /**
  * A special case of Pippenger's algorithm from Page 15 of
  * Bernstein, Doumen, Lange, Oosterwijk,
  * "Faster batch forgery identification", INDOCRYPT 2012
  * (https://eprint.iacr.org/2012/549.pdf)
  * When compiled with USE_MIXED_ADDITION, assumes input is in special form.
  * Requires that T implements .dbl() (and, if USE_MIXED_ADDITION is defined,
  * .to_special(), .mixed_add(), and batch_to_special()).
  */
 multi_exp_method_BDLO12
};

template<typename T, typename FieldT>
struct GpuMclData{
    gpu::mcl_bn128_g1 h_values, d_values, d_partial, d_t_zero, d_t_one;
    std::vector<gpu::mcl_bn128_g1> d_values2, d_buckets, d_buckets2, d_block_sums, d_block_sums2;
    gpu::Fp_model h_scalars, d_scalars, d_field_zero, d_field_one;

    gpu::gpu_meta d_counters, d_counters2, d_index_it, d_firsts, d_seconds, d_bucket_counters, d_starts, d_indexs, d_ids, d_instance_bucket_ids, d_density, d_flags;
    gpu::gpu_buffer dmax_value, d_bn_exponents, h_bn_exponents, d_modulus, d_field_modulus;
    cudaStream_t stream;
    gpu::Fp_model d_one, d_p, d_a;
    int max_depth = 0;
    GpuMclData(){
        gpu::create_stream(&stream);
        dmax_value.resize(1);
        d_t_zero.init(1);
        d_t_one.init(1);
        d_field_zero.init(1);
        d_field_one.init(1);
        d_modulus.resize(1);
        d_field_modulus.resize(1);
        d_one.init(1);
        d_p.init(1);
        d_a.init(1);
        d_values2.resize(1);
        d_buckets.resize(1);
        d_buckets2.resize(1);
        d_block_sums.resize(1);
        d_block_sums2.resize(1);
        d_density.resize(1);
    }

    ~GpuMclData(){
        h_values.release_host();
        h_bn_exponents.release_host();
        d_values.release();
        d_partial.release();
        for(int i = 0; i < 1; i++){
            d_values2[i].release();
            d_buckets[i].release();
            d_buckets2[i].release();
            d_block_sums[i].release();
            d_block_sums2[i].release();
        }
        d_scalars.release();
        h_scalars.release_host();
        d_field_one.release();
        d_field_zero.release();
        d_density.release();
        d_flags.release();
        d_counters.release();
        d_counters2.release();
        d_index_it.release();
        d_firsts.release();
        d_seconds.release();
        d_bucket_counters.release();
        d_starts.release();
        d_indexs.release();
        d_ids.release();
        d_instance_bucket_ids.release();
        d_bn_exponents.release();
        dmax_value.release();
        d_modulus.release();
        d_field_modulus.release();
        d_t_zero.release();
        d_t_one.release();
        d_one.release();
        d_p.release();
        d_a.release();
        gpu::release_stream(stream);
    }

};

template<typename T, typename FieldT>
static void copy_back(T& dst, const gpu::mcl_bn128_g1& src, const int offset, gpu::CudaStream stream){
    uint64_t tmp[4];
    gpu::copy_gpu_to_cpu(tmp, src.x.mont_repr_data + offset, 32, stream);
    gpu::sync(stream);
    dst.pt.x.copy(tmp);
    gpu::copy_gpu_to_cpu(tmp, src.y.mont_repr_data + offset, 32, stream);
    gpu::sync(stream);
    dst.pt.y.copy(tmp);
    gpu::copy_gpu_to_cpu(tmp, src.z.mont_repr_data + offset, 32, stream);
    gpu::sync(stream);
    dst.pt.z.copy(tmp);
}

template<typename T, typename FieldT>
static void copy_t(const T& src, gpu::mcl_bn128_g1& dst, const int offset, gpu::CudaStream stream){
    gpu::copy_cpu_to_gpu(dst.x.mont_repr_data + offset, src.pt.x.getUnit(), 32, stream);
    gpu::copy_cpu_to_gpu(dst.y.mont_repr_data + offset, src.pt.y.getUnit(), 32, stream);
    gpu::copy_cpu_to_gpu(dst.z.mont_repr_data + offset, src.pt.z.getUnit(), 32, stream);
}

template<typename T, typename FieldT>
static void copy_t_h(const T& src, gpu::alt_bn128_g1& dst, const int offset){
    memcpy(dst.x.mont_repr_data + offset, src.pt.x.getUnit(), 32);
    memcpy(dst.y.mont_repr_data + offset, src.pt.y.getUnit(), 32);
    memcpy(dst.z.mont_repr_data + offset, src.pt.z.getUnit(), 32);
}

template<typename T, typename FieldT>
static void copy_field(const FieldT& src, gpu::Fp_model& dst, const int offset, gpu::CudaStream stream){
    gpu::copy_cpu_to_gpu(dst.mont_repr_data + offset, src.mont_repr.data, 32, stream);
}

template<typename T, typename FieldT>
static void copy_field_h(const FieldT& src, gpu::Fp_model& dst, const int offset){
    memcpy(dst.mont_repr_data + offset, src.mont_repr.data, 32);
}

/**
 * Computes the sum
 * \sum_i scalar_start[i] * vec_start[i]
 * using the selected method.
 * Input is split into the given number of chunks, and, when compiled with
 * MULTICORE, the chunks are processed in parallel.
 */
template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            typename std::vector<FieldT>::const_iterator scalar_start,
            typename std::vector<FieldT>::const_iterator scalar_end,
            std::vector<bigint<FieldT::num_limbs>>& scratch_exponents,
            const libsnark::Config& config = libsnark::Config());


/**
 * A variant of multi_exp that takes advantage of the method mixed_add (instead
 * of the operator '+').
 * Assumes input is in special form, and includes special pre-processing for
 * scalars equal to 0 or 1.
 */
template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp_with_mixed_addition(typename std::vector<T>::const_iterator vec_start,
                                typename std::vector<T>::const_iterator vec_end,
                                typename std::vector<FieldT>::const_iterator scalar_start,
                                typename std::vector<FieldT>::const_iterator scalar_end,
                                std::vector<bigint<FieldT::num_limbs>>& scratch_exponents,
                                const libsnark::Config& config = libsnark::Config());

/**
 * A convenience function for calculating a pure inner product, where the
 * more complicated methods are not required.
 */
template <typename T>
T inner_product(typename std::vector<T>::const_iterator a_start,
                typename std::vector<T>::const_iterator a_end,
                typename std::vector<T>::const_iterator b_start,
                typename std::vector<T>::const_iterator b_end);

/**
 * A window table stores window sizes for different instance sizes for fixed-base multi-scalar multiplications.
 */
template<typename T>
using window_table = std::vector<std::vector<T> >;

/**
 * Compute window size for the given number of scalars.
 */
template<typename T>
size_t get_exp_window_size(const size_t num_scalars);

/**
 * Compute table of window sizes.
 */
template<typename T>
window_table<T> get_window_table(const size_t scalar_size,
                                 const size_t window,
                                 const T &g);

template<typename T, typename FieldT>
T windowed_exp(const size_t scalar_size,
               const size_t window,
               const window_table<T> &powers_of_g,
               const FieldT &pow);

template<typename T, typename FieldT>
std::vector<T> batch_exp(const size_t scalar_size,
                         const size_t window,
                         const window_table<T> &table,
                         const std::vector<FieldT> &v);

template<typename T, typename FieldT>
std::vector<T> batch_exp_with_coeff(const size_t scalar_size,
                                    const size_t window,
                                    const window_table<T> &table,
                                    const FieldT &coeff,
                                    const std::vector<FieldT> &v);

template<typename T>
void batch_to_special(std::vector<T> &vec);

} // libff

#include <libff/algebra/scalar_multiplication/multiexp.tcc>

#endif // MULTIEXP_HPP_
