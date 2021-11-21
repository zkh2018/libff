/** @file
 *****************************************************************************

 Implementation of interfaces for multi-exponentiation routines.

 See multiexp.hpp .

 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef MULTIEXP_TCC_
#define MULTIEXP_TCC_

#include <algorithm>
#include <cassert>
#include <type_traits>

#include <libff/algebra/fields/bigint.hpp>
#include <libff/algebra/fields/fp_aux.tcc>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>
#include <libff/algebra/scalar_multiplication/wnaf.hpp>
#include <libff/common/profiling.hpp>
#include <libff/common/utils.hpp>

#include "prover_config.hpp"

#include "cgbn_math.h"
#include "cgbn_fp.h"
#include "cgbn_alt_bn128_g1.h"
#include "cgbn_alt_bn128_g2.h"
#include "cgbn_multi_exp.h"
#include "cgbn_multi_exp_g2.h"
#include <cuda_runtime.h>

namespace libff {

template<mp_size_t n>
class ordered_exponent {
// to use std::push_heap and friends later
public:
    size_t idx;
    bigint<n> r;

    ordered_exponent(const size_t idx, const bigint<n> &r) : idx(idx), r(r) {};

    bool operator<(const ordered_exponent<n> &other) const
    {
#if defined(__x86_64__) && defined(USE_ASM)
        if (n == 3)
        {
            long res;
            __asm__
                ("// check for overflow           \n\t"
                 "mov $0, %[res]                  \n\t"
                 ADD_CMP(16)
                 ADD_CMP(8)
                 ADD_CMP(0)
                 "jmp done%=                      \n\t"
                 "subtract%=:                     \n\t"
                 "mov $1, %[res]                  \n\t"
                 "done%=:                         \n\t"
                 : [res] "=&r" (res)
                 : [A] "r" (other.r.data), [mod] "r" (this->r.data)
                 : "cc", "%rax");
            return res;
        }
        else if (n == 4)
        {
            long res;
            __asm__
                ("// check for overflow           \n\t"
                 "mov $0, %[res]                  \n\t"
                 ADD_CMP(24)
                 ADD_CMP(16)
                 ADD_CMP(8)
                 ADD_CMP(0)
                 "jmp done%=                      \n\t"
                 "subtract%=:                     \n\t"
                 "mov $1, %[res]                  \n\t"
                 "done%=:                         \n\t"
                 : [res] "=&r" (res)
                 : [A] "r" (other.r.data), [mod] "r" (this->r.data)
                 : "cc", "%rax");
            return res;
        }
        else if (n == 5)
        {
            long res;
            __asm__
                ("// check for overflow           \n\t"
                 "mov $0, %[res]                  \n\t"
                 ADD_CMP(32)
                 ADD_CMP(24)
                 ADD_CMP(16)
                 ADD_CMP(8)
                 ADD_CMP(0)
                 "jmp done%=                      \n\t"
                 "subtract%=:                     \n\t"
                 "mov $1, %[res]                  \n\t"
                 "done%=:                         \n\t"
                 : [res] "=&r" (res)
                 : [A] "r" (other.r.data), [mod] "r" (this->r.data)
                 : "cc", "%rax");
            return res;
        }
        else
#endif
        {
            return (mpn_cmp(this->r.data, other.r.data, n) < 0);
        }
    }
};

/**
 * multi_exp_inner<T, FieldT, Method>() implementes the specified
 * multiexponentiation method.
 * this implementation relies on some rather arcane template magic:
 * function templates cannot be partially specialized, so we cannot just write
 *     template<typename T, typename FieldT>
 *     T multi_exp_inner<T, FieldT, multi_exp_method_naive>
 * thus we resort to using std::enable_if. the basic idea is that *overloading*
 * is what's actually happening here, it's just that, for any given value of
 * Method, only one of the templates will be valid, and thus the correct
 * implementation will be used.
 */

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_naive), int>::type = 0>
T multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    T result(T::zero());

    typename std::vector<T>::const_iterator vec_it;
    typename std::vector<FieldT>::const_iterator scalar_it;

    for (vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it)
    {
        bigint<FieldT::num_limbs> scalar_bigint = scalar_it->as_bigint();
        result = result + opt_window_wnaf_exp(*vec_it, scalar_bigint, scalar_bigint.num_bits());
    }
    assert(scalar_it == scalar_end);

    return result;
}

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_naive_plain), int>::type = 0>
T multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    T result(T::zero());

    typename std::vector<T>::const_iterator vec_it;
    typename std::vector<FieldT>::const_iterator scalar_it;

    for (vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it)
    {
        result = result + (*scalar_it) * (*vec_it);
    }
    assert(scalar_it == scalar_end);

    return result;
}

static inline size_t get_id(size_t c, size_t bitno, const mp_limb_t* data)
{
    static const mp_limb_t one = 1;
    const mp_limb_t mask = (one << c) - one;
    const size_t limb_num_bits = sizeof(mp_limb_t) * 8;

    const size_t part = bitno / limb_num_bits;
    const size_t bit = bitno % limb_num_bits;
    size_t id = (data[part] & (mask << bit)) >> bit;
    //const mp_limb_t next_data = (bit + c >= limb_num_bits && part < 3) ? bn_exponents[i].data[part+1] : 0;
    //id |= (next_data & (mask >> (limb_num_bits - bit))) << (limb_num_bits - bit);
    id |= (((bit + c >= limb_num_bits && part < 3) ? data[part+1] : 0) & (mask >> (limb_num_bits - bit))) << (limb_num_bits - bit);

    return id;
}

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_BDLO12), int>::type = 0>
T multi_exp_inner(
    typename std::vector<T>::const_iterator bases,
    typename std::vector<T>::const_iterator bases_end,
    typename std::vector<FieldT>::const_iterator exponents,
    typename std::vector<FieldT>::const_iterator exponents_end)
{
    UNUSED(exponents_end);
    size_t length = bases_end - bases;

    // empirically, this seems to be a decent estimate of the optimal value of c
    size_t log2_length = log2(length);
    size_t c = log2_length - (log2_length / 3 - 2);

    c = 16;

    const mp_size_t exp_num_limbs =
        std::remove_reference<decltype(*exponents)>::type::num_limbs;
    std::vector<bigint<exp_num_limbs> > bn_exponents(length);
    size_t num_bits = 0;

    enter_block("Convert to bigint");
    for (size_t i = 0; i < length; i++)
    {
        bn_exponents[i] = exponents[i].as_bigint();
        num_bits = std::max(num_bits, bn_exponents[i].num_bits());
    }
    leave_block("Convert to bigint");

    size_t num_groups = (num_bits + c - 1) / c;

    T result;

    std::vector<T> buckets(1 << c);

    for (size_t k = num_groups - 1; k <= num_groups; k--)
    {
        for (size_t i = 0; i < c; i++)
        {
            result = result.dbl();
        }

        for (size_t i = 0; i < length; i++)
        {
            size_t id = get_id(c, k*c, bn_exponents[i].data);
            if (id == 0)
            {
                continue;
            }

#ifdef USE_MIXED_ADDITION
            buckets[id] = buckets[id].mixed_add(bases[i]);
#else
            buckets[id] = buckets[id] + bases[i];
#endif
        }

#ifdef USE_MIXED_ADDITION
        batch_to_special(buckets);
#endif

        T running_sum;
        for (size_t i = (1u << c) - 1; i > 0; i--)
        {
#ifdef USE_MIXED_ADDITION
            running_sum = running_sum.mixed_add(buckets[i]);
#else
            running_sum = running_sum + buckets[i];
#endif
            buckets[i] = T::zero();
            result = result + running_sum;
        }
    }

    return result;
}

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_bos_coster), int>::type = 0>
T multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    const mp_size_t n = std::remove_reference<decltype(*scalar_start)>::type::num_limbs;

    if (vec_start == vec_end)
    {
        return T::zero();
    }

    if (vec_start + 1 == vec_end)
    {
        return (*scalar_start)*(*vec_start);
    }

    std::vector<ordered_exponent<n> > opt_q;
    const size_t vec_len = scalar_end - scalar_start;
    const size_t odd_vec_len = (vec_len % 2 == 1 ? vec_len : vec_len + 1);
    opt_q.reserve(odd_vec_len);
    std::vector<T> g;
    g.reserve(odd_vec_len);

    typename std::vector<T>::const_iterator vec_it;
    typename std::vector<FieldT>::const_iterator scalar_it;
    size_t i;
    for (i=0, vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it, ++i)
    {
        g.emplace_back(*vec_it);

        opt_q.emplace_back(ordered_exponent<n>(i, scalar_it->as_bigint()));
    }
    std::make_heap(opt_q.begin(),opt_q.end());
    assert(scalar_it == scalar_end);

    if (vec_len != odd_vec_len)
    {
        g.emplace_back(T::zero());
        opt_q.emplace_back(ordered_exponent<n>(odd_vec_len - 1, bigint<n>(0ul)));
    }
    assert(g.size() % 2 == 1);
    assert(opt_q.size() == g.size());

    T opt_result = T::zero();

    while (true)
    {
        ordered_exponent<n> &a = opt_q[0];
        ordered_exponent<n> &b = (opt_q[1] < opt_q[2] ? opt_q[2] : opt_q[1]);

        const size_t abits = a.r.num_bits();

        if (b.r.is_zero())
        {
            // opt_result = opt_result + (a.r * g[a.idx]);
            opt_result = opt_result + opt_window_wnaf_exp(g[a.idx], a.r, abits);
            break;
        }

        const size_t bbits = b.r.num_bits();
        const size_t limit = (abits-bbits >= 20 ? 20 : abits-bbits);

        if (bbits < 1ul<<limit)
        {
            /*
              In this case, exponentiating to the power of a is cheaper than
              subtracting b from a multiple times, so let's do it directly
            */
            // opt_result = opt_result + (a.r * g[a.idx]);
            opt_result = opt_result + opt_window_wnaf_exp(g[a.idx], a.r, abits);
#ifdef DEBUG
            printf("Skipping the following pair (%zu bit number vs %zu bit):\n", abits, bbits);
            a.r.print();
            b.r.print();
#endif
            a.r.clear();
        }
        else
        {
            // x A + y B => (x-y) A + y (B+A)
            mpn_sub_n(a.r.data, a.r.data, b.r.data, n);
            g[b.idx] = g[b.idx] + g[a.idx];
        }

        // regardless of whether a was cleared or subtracted from we push it down, then take back up

        /* heapify A down */
        size_t a_pos = 0;
        while (2*a_pos + 2< odd_vec_len)
        {
            // this is a max-heap so to maintain a heap property we swap with the largest of the two
            if (opt_q[2*a_pos+1] < opt_q[2*a_pos+2])
            {
                std::swap(opt_q[a_pos], opt_q[2*a_pos+2]);
                a_pos = 2*a_pos+2;
            }
            else
            {
                std::swap(opt_q[a_pos], opt_q[2*a_pos+1]);
                a_pos = 2*a_pos+1;
            }
        }

        /* now heapify A up appropriate amount of times */
        while (a_pos > 0 && opt_q[(a_pos-1)/2] < opt_q[a_pos])
        {
            std::swap(opt_q[a_pos], opt_q[(a_pos-1)/2]);
            a_pos = (a_pos-1) / 2;
        }
    }

    return opt_result;
}

template<unsigned int locality>
static inline void prefetch_range(char* addr, size_t len, unsigned int prefetch_stride)
{
    char *cp;
    char *end = addr + len;
    for (cp = addr; cp < end; cp += prefetch_stride)
    {
        __builtin_prefetch(cp, 1, locality);
    }
}

#if 0
template<typename T, typename FieldT, bool prefetch, unsigned int prefetch_locality>
T multi_exp_inner_bellman(
    typename std::vector<T>::const_iterator bases,
    typename std::vector<T>::const_iterator bases_end,
    const std::vector<bigint<4>>& exponents,
    unsigned int c,
    unsigned int k,
    unsigned int prefetch_stride,
    unsigned int look_ahead)
{
    size_t length = bases_end - bases;

    std::vector<T> buckets(1 << c);

#if 1
    for (size_t i = 0; i < length; i++)
    {
        if (prefetch)
        {
            // prefetch next bucket
            if (i < length - look_ahead)
            {
                size_t next_id = get_id(c, k*c, &exponents[i+look_ahead].data[0]);
                if (next_id != 0)
                {
                    prefetch_range<prefetch_locality>((char*) &buckets[next_id], sizeof(T), prefetch_stride);
                }
            }
        }
        size_t id = get_id(c, k*c, &exponents[i].data[0]);
        if (id != 0)
        {
            buckets[id] = buckets[id] + bases[i];
        }
    }
#else
    size_t id = get_id(c, k*c, &exponents[0].data[0]);
    size_t next_id;
    size_t i = 0;
    while(i < length)
    {
        size_t j = i + 1;
        while(j < length)
        {
            next_id = get_id(c, k*c, &exponents[j].data[0]);
            if (next_id != 0)
            {
                if (prefetch)
                {
                    // prefetch next bucket
                    prefetch_range<prefetch_locality>((char*) &buckets[next_id], sizeof(T), prefetch_stride);
                }
                break;
            }
            j++;
        }

        if (id != 0)
        {
            buckets[id] = buckets[id] + bases[i];
        }
        id = next_id;
        i = j;
    }
#endif

    T result;
    T running_sum;
    for (size_t i = (1u << c) - 1; i > 0; i--)
    {
        running_sum = running_sum + buckets[i];
        result = result + running_sum;
    }

    return result;
}
#endif

template<typename T, typename FieldT, bool with_density, bool prefetch, unsigned int prefetch_locality>
T multi_exp_inner_bellman_with_density(
    typename std::vector<T>::const_iterator bases,
    typename std::vector<T>::const_iterator bases_end,
    const std::vector<bigint<4>>& exponents,
    const std::vector<bool>& density,
    unsigned int c,
    unsigned int k,
    unsigned int prefetch_stride,
    unsigned int look_ahead)
{
    size_t length = bases_end - bases;

    std::vector<T> buckets(1 << c);
    for (size_t i = 0; i < length; i++)
    {
        if (with_density)
        {
            if (!density[i])
            {
                continue;
            }
        }

        if (prefetch)
        {
            // prefetch next bucket
            if (i < length - look_ahead)
            {
                size_t next_id = get_id(c, k*c, &exponents[i+look_ahead].data[0]);
                if (next_id != 0)
                {
                    prefetch_range<prefetch_locality>((char*) &buckets[next_id], sizeof(T), prefetch_stride);
                }
            }
        }
        size_t id = get_id(c, k*c, &exponents[i].data[0]);
        if (id != 0)
        {
            buckets[id] = buckets[id] + bases[i];
        }
    }

    T result;
    T running_sum;
    for (size_t i = (1u << c) - 1; i > 0; i--)
    {
        running_sum = running_sum + buckets[i];
        result = result + running_sum;
    }

    return result;
}

template<typename T, typename FieldT, bool with_density>
void multi_exp_inner_bellman_with_density_gpu(
    typename std::vector<T>::const_iterator bases,
    typename std::vector<T>::const_iterator bases_end,
    const std::vector<bigint<4>>& exponents,
    const std::vector<bool>& density,
    unsigned int c,
    unsigned int k,
    unsigned int prefetch_stride,
    unsigned int look_ahead,
    gpu::alt_bn128_g1 d_values,
    char* d_density,
    gpu::gpu_buffer d_bn_exponents,
    int *gpu_bucket_counters, 
    int *gpu_starts, int *gpu_indexs, int* gpu_ids,
    int* gpu_instance_bucket_ids,
    gpu::alt_bn128_g1 d_values2, 
    gpu::alt_bn128_g1 d_buckets, 
    gpu::alt_bn128_g1 d_buckets2, 
    gpu::alt_bn128_g1 d_t_zero, 
    gpu::alt_bn128_g1 d_block_sums, 
    gpu::alt_bn128_g1 d_block_sums2,
    gpu::gpu_buffer d_max_value, 
    gpu::gpu_buffer d_modulus,
    cudaStream_t stream
    ){
    size_t length = bases_end - bases;

    const int instances = gpu::BUCKET_INSTANCES;
    auto f = [](clock_t t1, clock_t t0){
      return (double)(t1-t0)/CLOCKS_PER_SEC;
    };
    auto copy_back = [&](T& dst, const gpu::alt_bn128_g1& src, const int offset){
      gpu::copy_gpu_to_cpu(dst.X.mont_repr.data, src.x.mont_repr_data + offset, 32);
      gpu::copy_gpu_to_cpu(dst.Y.mont_repr.data, src.y.mont_repr_data + offset, 32);
      gpu::copy_gpu_to_cpu(dst.Z.mont_repr.data, src.z.mont_repr_data + offset, 32);
    };
    auto copy_t = [](const T& src, gpu::alt_bn128_g1& dst, const int offset){
      gpu::copy_cpu_to_gpu(dst.x.mont_repr_data + offset, src.X.mont_repr.data, 32);
      gpu::copy_cpu_to_gpu(dst.y.mont_repr_data + offset, src.Y.mont_repr.data, 32);
      gpu::copy_cpu_to_gpu(dst.z.mont_repr_data + offset, src.Z.mont_repr.data, 32);
    };

    //const int bucket_num = 1<<c;
    //std::vector<int> bucket_counter(bucket_num, 0), starts(bucket_num, 0), ends(bucket_num, 0);
    //for (size_t i = 0; i < length; i++)
    //{
    //    if (with_density)
    //    {
    //        if (!density[i])
    //        {
    //            continue;
    //        }
    //    }

    //    size_t id = 0;//get_id(c, k*c, &exponents[i].data[0]);
    //    if (id != 0)
    //    {
    //      bucket_counter[id] += 1;
    //    }
    //}
    //int offset = 0;
    //for(int i = 0; i < bucket_num; i++){
    //  starts[i] = offset;
    //  offset += bucket_counter[i];
    //  ends[i] = offset;
    //}

    if(true){
      uint64_t const_inv = bases[0].X.inv;

      //clock_t t0 = clock();
      //printf("1\n"); 
      gpu::gpu_set_zero(gpu_bucket_counters, (1<<c)*sizeof(int), stream);
      gpu::bucket_counter(with_density, d_density, d_bn_exponents.ptr, c, k, length, (1<<c), gpu_bucket_counters, stream);
      //std::vector<int> tmp_counter(bucket_num);
      //gpu::copy_gpu_to_cpu(tmp_counter.data(), gpu_bucket_counters, bucket_num * sizeof(int));
      //int cmp_ret = memcmp(tmp_counter.data(), bucket_counter.data(), bucket_num * sizeof(int));
      //printf("1 compare result = %d\n", cmp_ret);

      gpu::prefix_sum(gpu_bucket_counters, gpu_starts, 1<<c, stream);
      //std::vector<int> tmp_starts(bucket_num);
      //gpu::copy_gpu_to_cpu(tmp_starts.data(), gpu_starts, bucket_num * sizeof(int));
      //cmp_ret = memcmp(tmp_starts.data(), starts.data(), bucket_num * sizeof(int));
      //printf("2 compare result = %d\n", cmp_ret);
      //printf("3\n");
      gpu::copy_gpu_to_gpu(gpu_indexs, gpu_starts, (1<<c) * sizeof(int), stream);
      //printf("4\n");
      //
      gpu::split_to_bucket(d_values, d_values2, with_density, d_density, d_bn_exponents.ptr, c, k, length, gpu_indexs, stream);
      //int bucket_num = 1 << c;
      //std::vector<int> tmp_ends(bucket_num), tmp_starts(bucket_num);
      //gpu::copy_gpu_to_cpu(tmp_ends.data(), gpu_indexs, bucket_num * sizeof(int));
      //gpu::copy_gpu_to_cpu(tmp_starts.data(), gpu_starts, bucket_num * sizeof(int));
      //T tmp_result;
      //for(int i = 0; i < bucket_num; i++){
      //  for(int j = tmp_starts[i]; j < tmp_ends[i]; j++){
      //    T value;
      //    copy_back(value, d_values, j);
      //    tmp_result  = tmp_result + value; 
      //  }
      //}
      //copy_t(tmp_result, d_buckets, 0);
      //return ;
      //cmp_ret = memcmp(tmp_ends.data(), ends.data(), bucket_num * sizeof(int));
      //printf("3 compare result = %d\n", cmp_ret);

      d_buckets.clear(stream);
      gpu::bucket_reduce_sum(d_values2, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids, d_buckets, 1<<c, length, d_max_value.ptr, d_t_zero, d_modulus.ptr, const_inv, stream);
      gpu::reverse(d_buckets, d_buckets2, 1<<c, instances, stream);
      //printf("7\n");
      //clock_t t2 = clock();
      gpu::prefix_sum(d_buckets2, d_block_sums, d_block_sums2, 1<<c, d_max_value.ptr, d_modulus.ptr, const_inv, stream);
      //printf("8\n");
      //clock_t t3 = clock();
      gpu::alt_bn128_g1_reduce_sum2(d_buckets2, d_buckets, (1<<c), d_max_value.ptr, d_modulus.ptr, const_inv, stream);
      //printf("9\n");
      //clock_t t4 = clock();
      //printf("%f %f %f %f\n", f(t1, t0), f(t2, t1), f(t3, t2), f(t4, t3));
      //return tmp_result;
    }
}
template<typename T, typename FieldT, bool with_density, multi_exp_method Method>
T multi_exp_with_density_gpu(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            const std::vector<bigint<FieldT::num_limbs>>& exponents,
            const std::vector<bool>& density,
            const libsnark::Config& config,
            gpu::alt_bn128_g1 d_values,
            char* d_density,
            gpu::gpu_buffer d_bn_exponents,
            gpu::gpu_buffer d_max_value,
            gpu::gpu_buffer d_modulus,
            gpu::alt_bn128_g1 d_t_zero,
            std::vector<gpu::alt_bn128_g1> d_values2,
            std::vector<gpu::alt_bn128_g1> d_buckets,
            std::vector<gpu::alt_bn128_g1> d_buckets2,
            std::vector<gpu::alt_bn128_g1> d_block_sums,
            std::vector<gpu::alt_bn128_g1> d_block_sums2,
            int* gpu_bucket_counters,
            int* gpu_starts, 
            int *gpu_indexs,
            int *gpu_ids,
            int* gpu_instance_bucket_ids
            )
{
    unsigned int chunks = config.num_threads;
    const size_t total = vec_end - vec_start;

    unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
    chunks = (254 + c - 1) / c;

    auto ranges = libsnark::get_cpu_ranges(0, total);
    std::vector<T> partial(chunks, T::zero());

    const int length = total;
    std::vector<cudaStream_t> streams(1);
    for(int i = 0; i < 1; i++){
      gpu::create_stream(&streams[i]);
    }

    auto copy_back = [&](T& dst, const gpu::alt_bn128_g1& src, const int offset){
      gpu::copy_gpu_to_cpu(dst.X.mont_repr.data, src.x.mont_repr_data + offset, 32);
      gpu::copy_gpu_to_cpu(dst.Y.mont_repr.data, src.y.mont_repr_data + offset, 32);
      gpu::copy_gpu_to_cpu(dst.Z.mont_repr.data, src.z.mont_repr_data + offset, 32);
    };
    auto copy_back_h = [&](T& dst, const gpu::alt_bn128_g1& src, const int offset){
      memcpy(dst.X.mont_repr.data, src.x.mont_repr_data + offset, 32);
      memcpy(dst.Y.mont_repr.data, src.y.mont_repr_data + offset, 32);
      memcpy(dst.Z.mont_repr.data, src.z.mont_repr_data + offset, 32);
    };

    //const int n = 1 << c;
    for (size_t i = 0; i < chunks; ++i)
    {
      //const int offset = i * n;
      multi_exp_inner_bellman_with_density_gpu<T, FieldT, with_density>(
          vec_start, vec_end, exponents, density, c, i, config.prefetch_stride, config.multi_exp_look_ahead,
          d_values, d_density, d_bn_exponents,
          gpu_bucket_counters, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids,
          d_values2[0], d_buckets[0], d_buckets2[0], d_t_zero, 
          d_block_sums[0], d_block_sums2[0],
          d_max_value, d_modulus, streams[0]);
      copy_back(partial[i], d_buckets[0], 0);
    }

    //copy_back(partial[chunks-1], d_buckets[chunks-1], 0);
    T final = partial[chunks - 1];
    for (int i = chunks - 2; i >= 0; i--)
    {
        for (size_t j = 0; j < c; j++)
        {
            final = final.dbl();
        }
        //copy_back(partial[i], d_buckets[i], 0);
        final = final + partial[i];
    }
    return final;
}

template<typename T, typename FieldT, bool with_density>
void multi_exp_inner_bellman_with_density_g2_gpu(
    typename std::vector<T>::const_iterator bases,
    typename std::vector<T>::const_iterator bases_end,
    const std::vector<bigint<4>>& exponents,
    const std::vector<bool>& density,
    unsigned int c,
    unsigned int k,
    unsigned int prefetch_stride,
    unsigned int look_ahead,
    gpu::alt_bn128_g2 d_values,
    char* d_density,
    gpu::gpu_buffer d_bn_exponents,
    int *gpu_bucket_counters, 
    int *gpu_starts, int *gpu_indexs, int* gpu_ids,
    int* gpu_instance_bucket_ids,
    gpu::alt_bn128_g2 d_values2, 
    gpu::alt_bn128_g2 d_buckets, 
    gpu::alt_bn128_g2 d_buckets2, 
    gpu::alt_bn128_g2 d_t_zero, 
    gpu::alt_bn128_g2 d_block_sums, 
    gpu::alt_bn128_g2 d_block_sums2,
    gpu::gpu_buffer d_max_value, 
    gpu::gpu_buffer d_modulus,
    gpu::Fp_model d_non_residue,
    cudaStream_t stream
    ){
    size_t length = bases_end - bases;

    const int instances = gpu::BUCKET_INSTANCES_G2;
    auto f = [](clock_t t1, clock_t t0){
      return (double)(t1-t0)/CLOCKS_PER_SEC;
    };
      auto copy_back = [&](T& dst, const gpu::alt_bn128_g2& src, const int offset){
        gpu::copy_gpu_to_cpu(dst.X.c0.mont_repr.data, src.x.c0.mont_repr_data + offset, 32);
        gpu::copy_gpu_to_cpu(dst.Y.c0.mont_repr.data, src.y.c0.mont_repr_data + offset, 32);
        gpu::copy_gpu_to_cpu(dst.Z.c0.mont_repr.data, src.z.c0.mont_repr_data + offset, 32);
        gpu::copy_gpu_to_cpu(dst.X.c1.mont_repr.data, src.x.c1.mont_repr_data + offset, 32);
        gpu::copy_gpu_to_cpu(dst.Y.c1.mont_repr.data, src.y.c1.mont_repr_data + offset, 32);
        gpu::copy_gpu_to_cpu(dst.Z.c1.mont_repr.data, src.z.c1.mont_repr_data + offset, 32);
      };
    if(true){
      uint64_t const_inv = bases[0].X.c0.inv;

      //clock_t t0 = clock();
      gpu::gpu_set_zero(gpu_bucket_counters, (1<<c)*sizeof(int), stream);
      gpu::bucket_counter(with_density, d_density, d_bn_exponents.ptr, c, k, length, (1<<c), gpu_bucket_counters, stream);
      gpu::prefix_sum(gpu_bucket_counters, gpu_starts, 1<<c, stream);
      gpu::copy_gpu_to_gpu(gpu_indexs, gpu_starts, (1<<c) * sizeof(int), stream);
      gpu::split_to_bucket_g2(d_values, d_values2, d_density, d_bn_exponents.ptr, c, k, length, gpu_indexs, stream);

      d_buckets.clear(stream);
      gpu::bucket_reduce_sum_g2(d_values2, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids, d_buckets, 1<<c, length, d_max_value.ptr, d_t_zero, d_modulus.ptr, const_inv, d_non_residue, stream);
      gpu::reverse_g2(d_buckets, d_buckets2, 1<<c, instances, stream);
      //clock_t t2 = clock();
      gpu::prefix_sum_g2(d_buckets2, d_block_sums, d_block_sums2, 1<<c, d_max_value.ptr, d_modulus.ptr, const_inv, d_non_residue, stream);
      //clock_t t3 = clock();
      gpu::alt_bn128_g2_reduce_sum2(d_buckets2, d_buckets, (1<<c), d_max_value.ptr, d_modulus.ptr, const_inv, d_non_residue, stream);
      //clock_t t4 = clock();
      //printf("%f %f %f %f\n", f(t1, t0), f(t2, t1), f(t3, t2), f(t4, t3));
      //return tmp_result;
    }
}
template<typename T, typename FieldT, bool with_density, multi_exp_method Method>
T multi_exp_with_density_g2_gpu(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            const std::vector<bigint<FieldT::num_limbs>>& exponents,
            const std::vector<bool>& density,
            const libsnark::Config& config,
            gpu::alt_bn128_g2 d_values,
            char* d_density,
            gpu::gpu_buffer d_bn_exponents,
            gpu::gpu_buffer d_max_value,
            gpu::gpu_buffer d_modulus,
            gpu::alt_bn128_g2 d_t_zero,
            gpu::Fp_model d_non_residue,
            std::vector<gpu::alt_bn128_g2> d_values2,
            std::vector<gpu::alt_bn128_g2> d_buckets,
            std::vector<gpu::alt_bn128_g2> d_buckets2,
            std::vector<gpu::alt_bn128_g2> d_block_sums,
            std::vector<gpu::alt_bn128_g2> d_block_sums2,
            int* gpu_bucket_counters,
            int* gpu_starts, 
            int *gpu_indexs,
            int *gpu_ids,
            int* gpu_instance_bucket_ids
            )
{
    unsigned int chunks = config.num_threads;
    const size_t total = vec_end - vec_start;

    unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
    chunks = (254 + c - 1) / c;

    auto ranges = libsnark::get_cpu_ranges(0, total);
    std::vector<T> partial(chunks, T::zero());

    const int length = total;
    std::vector<cudaStream_t> streams(1);
    for(int i = 0; i < 1/*chunks*/; i++){
      gpu::create_stream(&streams[i]);
    }

    auto copy_back = [&](T& dst, const gpu::alt_bn128_g2& src, const int offset){
      gpu::copy_gpu_to_cpu(dst.X.c0.mont_repr.data, src.x.c0.mont_repr_data + offset, 32);
      gpu::copy_gpu_to_cpu(dst.Y.c0.mont_repr.data, src.y.c0.mont_repr_data + offset, 32);
      gpu::copy_gpu_to_cpu(dst.Z.c0.mont_repr.data, src.z.c0.mont_repr_data + offset, 32);

      gpu::copy_gpu_to_cpu(dst.X.c1.mont_repr.data, src.x.c1.mont_repr_data + offset, 32);
      gpu::copy_gpu_to_cpu(dst.Y.c1.mont_repr.data, src.y.c1.mont_repr_data + offset, 32);
      gpu::copy_gpu_to_cpu(dst.Z.c1.mont_repr.data, src.z.c1.mont_repr_data + offset, 32);
    };
    auto copy_back_h = [&](T& dst, const gpu::alt_bn128_g2& src, const int offset){
      memcpy(dst.X.c0.mont_repr.data, src.x.c0.mont_repr_data + offset, 32);
      memcpy(dst.Y.c0.mont_repr.data, src.y.c0.mont_repr_data + offset, 32);
      memcpy(dst.Z.c0.mont_repr.data, src.z.c0.mont_repr_data + offset, 32);

      memcpy(dst.X.c1.mont_repr.data, src.x.c1.mont_repr_data + offset, 32);
      memcpy(dst.Y.c1.mont_repr.data, src.y.c1.mont_repr_data + offset, 32);
      memcpy(dst.Z.c1.mont_repr.data, src.z.c1.mont_repr_data + offset, 32);
    };

    const int n = 1 << c;
//#pragma omp parallel for
    for (size_t i = 0; i < chunks; ++i)
    {
      //const int offset = i * n;
      multi_exp_inner_bellman_with_density_g2_gpu<T, FieldT, with_density>(
          vec_start, vec_end, exponents, density, c, i, config.prefetch_stride, config.multi_exp_look_ahead,
          d_values, d_density, d_bn_exponents,
          gpu_bucket_counters, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids,
          d_values2[0], d_buckets[0], d_buckets2[0], d_t_zero, 
          d_block_sums[0], d_block_sums2[0],
          d_max_value, d_modulus, d_non_residue, streams[0]);
      copy_back(partial[i], d_buckets[0], 0);
    }

    //copy_back(partial[chunks-1], d_buckets[chunks-1], 0);
    T final = partial[chunks - 1];
    for (int i = chunks - 2; i >= 0; i--)
    {
        for (size_t j = 0; j < c; j++)
        {
            final = final.dbl();
        }
        //copy_back(partial[i], d_buckets[i], 0);
        final = final + partial[i];
    }
    return final;
}

template<typename T, typename FieldT, bool with_density, multi_exp_method Method>
T multi_exp_with_density(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            const std::vector<bigint<FieldT::num_limbs>>& exponents,
            const std::vector<bool>& density,
            const libsnark::Config& config)
{
    unsigned int chunks = config.num_threads;
    const size_t total = vec_end - vec_start;

    unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
    chunks = (254 + c - 1) / c;

    auto ranges = libsnark::get_cpu_ranges(0, total);
    std::vector<T> partial(chunks, T::zero());
#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < chunks; ++i)
    {
        switch(config.multi_exp_prefetch_locality)
        {
            case 0:
            {
                partial[i] = multi_exp_inner_bellman_with_density<T, FieldT, with_density, false, 0>(
                    vec_start, vec_end, exponents, density, c, i, config.prefetch_stride, config.multi_exp_look_ahead
                );
                break;
            }
            case 1:
            {
                partial[i] = multi_exp_inner_bellman_with_density<T, FieldT, with_density, false, 1>(
                    vec_start, vec_end, exponents, density, c, i, config.prefetch_stride, config.multi_exp_look_ahead
                );
                break;
            }
            case 2:
            {
                partial[i] = multi_exp_inner_bellman_with_density<T, FieldT, with_density, false, 2>(
                    vec_start, vec_end, exponents, density, c, i, config.prefetch_stride, config.multi_exp_look_ahead
                );
                break;
            }
            case 3:
            {
                partial[i] = multi_exp_inner_bellman_with_density<T, FieldT, with_density, false, 3>(
                    vec_start, vec_end, exponents, density, c, i, config.prefetch_stride, config.multi_exp_look_ahead
                );
                break;
            }
            default:
            {
                partial[i] = multi_exp_inner_bellman_with_density<T, FieldT, with_density, false, 0>(
                    vec_start, vec_end, exponents, density, c, i, config.prefetch_stride, config.multi_exp_look_ahead
                );
                break;
            }
        }
    }

    T final = partial[chunks - 1];
    for (int i = chunks - 2; i >= 0; i--)
    {
        for (size_t j = 0; j < c; j++)
        {
            final = final.dbl();
        }
        final = final + partial[i];
    }
    return final;
}

template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            typename std::vector<FieldT>::const_iterator scalar_start,
            typename std::vector<FieldT>::const_iterator scalar_end,
            std::vector<bigint<FieldT::num_limbs>>& scratch_exponents,
            const libsnark::Config& config)
{
    const size_t total = vec_end - vec_start;
    std::vector<bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    if (bn_exponents.size() < total)
    {
        bn_exponents.resize(total);
    }

    enter_block("Convert to bigint");
    auto ranges = libsnark::get_cpu_ranges(0, total);
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); j++)
    {
        for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
        {
            bn_exponents[i] = scalar_start[i].as_bigint();
        }
    }
    leave_block("Convert to bigint");

    // Dummy density vector
    std::vector<bool> density;
    return multi_exp_with_density<T, FieldT, false, Method>(vec_start, vec_end, bn_exponents, density, config);
}

//T = alt_bn128_g1
template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp_gpu(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            typename std::vector<FieldT>::const_iterator scalar_start,
            typename std::vector<FieldT>::const_iterator scalar_end,
            std::vector<bigint<FieldT::num_limbs>>& scratch_exponents,
            const libsnark::Config& config)
{
    const size_t total = vec_end - vec_start;
    std::vector<bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    if (bn_exponents.size() < total)
    {
        bn_exponents.resize(total);
    }

    enter_block("Convert to bigint");
    auto ranges = libsnark::get_cpu_ranges(0, total);
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); j++)
    {
        for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
        {
            bn_exponents[i] = scalar_start[i].as_bigint();
        }
    }
    leave_block("Convert to bigint");
    printf("total=%d, ranges=%d\n", total, ranges.size());

    // Dummy density vector
    std::vector<bool> density;
    enter_block("cpu multi_exp_with_density");
    //auto cpu_result = multi_exp_with_density<T, FieldT, false, Method>(vec_start, vec_end, bn_exponents, density, config);
    leave_block("cpu multi_exp_with_density");

    /****call gpu*****/
    enter_block("gpu multi_exp_with_density");
    enter_block("gpu init");
    const int local_instances = BlockDepth * 64;
    unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
    unsigned int chunks = config.num_threads;
    chunks = 1;//(254 + c - 1) / c;
    const int instances = gpu::BUCKET_INSTANCES;
    const int length = total;

    static gpu::alt_bn128_g1 h_values, d_values, d_t_zero;
    static std::vector<gpu::alt_bn128_g1> d_values2(chunks), d_buckets(chunks), d_buckets2(chunks), d_block_sums(chunks), d_block_sums2(chunks);
    const int values_size = total;
    {
      h_values.init_host(values_size);
      d_values.init(values_size);
      d_t_zero.init(1);
      for(int i = 0; i < chunks; i++){
        d_values2[i].init(length);
        d_buckets[i].init((1<<c) * instances);
        d_buckets2[i].init((1<<c));
        d_block_sums[i].init((1<<c) / 64);
        d_block_sums2[i].init((1<<c) / 64/64);
      }
    }
    static int* gpu_bucket_counters = nullptr, *gpu_starts = nullptr, *gpu_indexs, *gpu_ids, *gpu_instance_bucket_ids;
    static char *d_density;
    static gpu::gpu_buffer max_value, dmax_value, d_bn_exponents, h_bn_exponents, d_modulus, d_field_modulus;
    {
      gpu::gpu_malloc((void**)&gpu_bucket_counters, (1<<c) * sizeof(int) * chunks);
      gpu::gpu_malloc((void**)&gpu_starts, (1<<c) * sizeof(int) * chunks * 2);
      gpu::gpu_malloc((void**)&gpu_indexs, (1<<c) * sizeof(int) * chunks * 2);
      gpu::gpu_malloc((void**)&gpu_ids, ((1<<c)+1) * sizeof(int) * chunks * 2);
      gpu::gpu_malloc((void**)&gpu_instance_bucket_ids, (length+1) * sizeof(int) * chunks);
      gpu::gpu_malloc((void**)&d_density, density.size() * chunks);
      max_value.resize_host(1);
      dmax_value.resize(1);
      d_modulus.resize(1);
      for(int i = 0; i < BITS/32; i++){
        max_value.ptr->_limbs[i] = 0xffffffff;
      }
      dmax_value.copy_from_host(max_value);
      d_bn_exponents.resize(bn_exponents.size());
      h_bn_exponents.resize_host(bn_exponents.size());
    }

    auto copy_t = [](const T& src, gpu::alt_bn128_g1& dst, const int offset){
      gpu::copy_cpu_to_gpu(dst.x.mont_repr_data + offset, src.X.mont_repr.data, 32);
      gpu::copy_cpu_to_gpu(dst.y.mont_repr_data + offset, src.Y.mont_repr.data, 32);
      gpu::copy_cpu_to_gpu(dst.z.mont_repr_data + offset, src.Z.mont_repr.data, 32);
    };
    auto copy_t_h = [](const T& src, gpu::alt_bn128_g1& dst, const int offset){
      memcpy(dst.x.mont_repr_data + offset, src.X.mont_repr.data, 32);
      memcpy(dst.y.mont_repr_data + offset, src.Y.mont_repr.data, 32);
      memcpy(dst.z.mont_repr_data + offset, src.Z.mont_repr.data, 32);
    };

    gpu::copy_cpu_to_gpu(d_modulus.ptr->_limbs, vec_start[0].X.get_modulus().data, 32);
    memcpy(h_bn_exponents.ptr, bn_exponents.data(), 32 * bn_exponents.size());
    d_bn_exponents.copy_from_host(h_bn_exponents);
    uint64_t const_inv = vec_start[0].X.inv;

    const auto& modu = vec_start[0].X.get_modulus();
    for(int i = 0; i < values_size; i++){
      copy_t_h(vec_start[i], h_values, i);
    }
    d_values.copy_from_cpu(h_values);
    copy_t(T::zero(), d_t_zero, 0);

    gpu::init_error_report();
    leave_block("gpu init");
    enter_block("call gpu");
    auto gpu_result = multi_exp_with_density_gpu<T, FieldT, false, Method>(vec_start, vec_end, bn_exponents, density, config, d_values, d_density, d_bn_exponents, dmax_value, d_modulus, d_t_zero, d_values2, d_buckets, d_buckets2, d_block_sums, d_block_sums2, gpu_bucket_counters, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids);
    leave_block("call gpu");
    leave_block("gpu multi_exp_with_density");
    
    if(false){
      d_values.release();
      for(int i = 0; i < chunks; i++){
        d_values2[i].release();
        d_buckets[i].release();
        d_buckets2[i].release();
        d_block_sums[i].release();
        d_block_sums2[i].release();
      }
      gpu::gpu_free(d_density);
      gpu::gpu_free(gpu_bucket_counters);
      gpu::gpu_free(gpu_starts);
      gpu::gpu_free(gpu_indexs);
      gpu::gpu_free(gpu_ids);
      gpu::gpu_free(gpu_instance_bucket_ids);
      d_bn_exponents.release();
      dmax_value.release();
      d_modulus.release();
    }
    return gpu_result;
}

template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp_with_mixed_addition(typename std::vector<T>::const_iterator vec_start,
                                typename std::vector<T>::const_iterator vec_end,
                                typename std::vector<FieldT>::const_iterator scalar_start,
                                typename std::vector<FieldT>::const_iterator scalar_end,
                                std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                const libsnark::Config& config)
{
    assert(std::distance(vec_start, vec_end) == std::distance(scalar_start, scalar_end));
    enter_block("Process scalar vector");

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();

    //size_t num_skip = 0;
    //size_t num_add = 0;
    //size_t num_other = 0;

    const size_t scalar_length = std::distance(scalar_start, scalar_end);

    libff::enter_block("allocate density memory");
    std::vector<bool> density(scalar_length);
    libff::leave_block("allocate density memory");

    std::vector<bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    if (bn_exponents.size() < scalar_length)
    {
        bn_exponents.resize(scalar_length);
    }

    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);
    std::vector<T> partial(ranges.size(), T::zero());
    std::vector<unsigned int> counters(ranges.size(), 0);

#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); j++)
    {
        T result = T::zero();
        unsigned int count = 0;
        for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
        {
            if (scalar_start[i] == zero)
            {
                // do nothing
                //++num_skip;
            }
            else if (scalar_start[i] == one)
            {
#ifdef USE_MIXED_ADDITION
                result = result.mixed_add(vec_start[i]);
#else
                result = result + vec_start[i];
#endif
                //++num_add;
            }
            else
            {
                //if (vec_start[i] != T::zero())
                {
                    density[i] = true;
                    bn_exponents[i] = scalar_start[i].as_bigint();
                    ++count;
                }
                //++num_other;
            }
        }
        partial[j] = result;
        counters[j] = count;
    }

    T acc = T::zero();
    unsigned int totalCount = 0;
    for (unsigned int i = 0; i < ranges.size(); i++)
    {
        acc = acc + partial[i];
        totalCount += counters[i];
    }

    leave_block("Process scalar vector");

    return acc + multi_exp_with_density<T, FieldT, true, Method>(vec_start, vec_end, bn_exponents, density, config);
}

template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp_with_mixed_addition_gpu(typename std::vector<T>::const_iterator vec_start,
                                typename std::vector<T>::const_iterator vec_end,
                                typename std::vector<FieldT>::const_iterator scalar_start,
                                typename std::vector<FieldT>::const_iterator scalar_end,
                                std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                const libsnark::Config& config)
{
    assert(std::distance(vec_start, vec_end) == std::distance(scalar_start, scalar_end));
    enter_block("Process scalar vector");

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();

    //size_t num_skip = 0;
    //size_t num_add = 0;
    //size_t num_other = 0;

    const size_t scalar_length = std::distance(scalar_start, scalar_end);

    libff::enter_block("allocate density memory");
    std::vector<bool> density(scalar_length);
    libff::leave_block("allocate density memory");

    std::vector<bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    if (bn_exponents.size() < scalar_length)
    {
        bn_exponents.resize(scalar_length);
    }

    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);
    std::vector<T> partial(ranges.size(), T::zero());
    std::vector<unsigned int> counters(ranges.size(), 0);
    int max_depth = 0;
    for(int i = 0; i < ranges.size(); i++){
      max_depth = std::max(max_depth, (int)(ranges[i].second - ranges[i].first));
    }
    enter_block("cpu reduce sum");

#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); j++)
    {
        T result = T::zero();
        unsigned int count = 0;
        for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
        {
            if (scalar_start[i] == zero)
            {
                // do nothing
                //++num_skip;
            }
            else if (scalar_start[i] == one)
            {
#ifdef USE_MIXED_ADDITION
                result = result.mixed_add(vec_start[i]);
#else
                //result = result + vec_start[i];
#endif
                //++num_add;
            }
            else
            {
                //if (vec_start[i] != T::zero())
                {
                    density[i] = true;
                    bn_exponents[i] = scalar_start[i].as_bigint();
                    ++count;
                }
                //++num_other;
            }
        }
        //partial[j] = result;
        //counters[j] = count;
    }

    //T acc = T::zero();
    //unsigned int totalCount = 0;
    //for (unsigned int i = 0; i < ranges.size(); i++)
    //{
    //    acc = acc + partial[i];
    //    totalCount += counters[i];
    //}
    leave_block("cpu reduce sum");


    enter_block("cpu multi_exp_with_density");
    //auto cpu_result = acc + multi_exp_with_density<T, FieldT, true, Method>(vec_start, vec_end, bn_exponents, density, config);
    leave_block("cpu multi_exp_with_density");
    if(true){
      libff::enter_block("call gpu");
      libff::enter_block("gpu init");
      const int local_instances = BlockDepth * 64;
      const int blocks = (max_depth + local_instances - 1) / local_instances;
      unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
      unsigned int chunks = config.num_threads;
      chunks = 1;//(254 + c - 1) / c;
      const int instances = gpu::BUCKET_INSTANCES;
      const int scalar_size = scalar_end - scalar_start;
      const int values_size = vec_end - vec_start;
      const int length = values_size;

      static gpu::alt_bn128_g1 h_values, d_values, d_partial, d_partial2, d_t_zero, d_t_one;
      static std::vector<gpu::alt_bn128_g1> d_values2(chunks), d_buckets(chunks), d_buckets2(chunks), d_block_sums(chunks), d_block_sums2(chunks);
      static gpu::Fp_model h_scalars, d_scalars, d_field_zero, d_field_one;
      const int ranges_size = ranges.size();
      {
        h_values.init_host(values_size);
        d_values.init(values_size);
        h_scalars.init_host(scalar_size);
        d_scalars.init(scalar_size);
        d_partial.init(ranges_size * blocks * 64);
        d_partial2.init(ranges_size);
        d_t_zero.init(1);
        d_t_one.init(1);
        d_field_zero.init(1);
        d_field_one.init(1);
        for(int i = 0; i < chunks; i++){
          d_values2[i].init(length);
          d_buckets[i].init((1<<c) * instances);
          d_buckets2[i].init((1<<c));
          d_block_sums[i].init((1<<c) / 64);
          d_block_sums2[i].init((1<<c) / 64/64);
        }
      }

      static uint32_t *d_counters, *d_counters2;
      static size_t* d_index_it;
      static uint32_t *d_firsts, *d_seconds;
      static int* gpu_bucket_counters = nullptr, *gpu_starts = nullptr, *gpu_indexs, *gpu_ids, *gpu_instance_bucket_ids;
      static char *d_density, *flags;
      static gpu::gpu_buffer max_value, dmax_value, d_bn_exponents, h_bn_exponents, d_modulus, d_field_modulus;
      {
        gpu::gpu_malloc((void**)&d_counters, ranges_size  * blocks * sizeof(uint32_t));
        gpu::gpu_malloc((void**)&d_counters2, ranges_size  * sizeof(uint32_t));
        gpu::gpu_malloc((void**)&d_firsts, ranges_size  * sizeof(uint32_t));
        gpu::gpu_malloc((void**)&d_seconds, ranges_size  * sizeof(uint32_t));
        gpu::gpu_malloc((void**)&gpu_bucket_counters, (1<<c) * sizeof(int) * chunks);
        gpu::gpu_malloc((void**)&gpu_starts, (1<<c) * sizeof(int) * chunks * 2);
        gpu::gpu_malloc((void**)&gpu_indexs, (1<<c) * sizeof(int) * chunks * 2);
        gpu::gpu_malloc((void**)&gpu_ids, ((1<<c)+1) * sizeof(int) * chunks);
        gpu::gpu_malloc((void**)&gpu_instance_bucket_ids, length * sizeof(int) * chunks);
        gpu::gpu_malloc((void**)&flags, scalar_size * sizeof(char));
        max_value.resize_host(1);
        dmax_value.resize(1);
        d_modulus.resize(1);
        d_field_modulus.resize(1);
        for(int i = 0; i < BITS/32; i++){
          max_value.ptr->_limbs[i] = 0xffffffff;
        }
        dmax_value.copy_from_host(max_value);
        int total_indices = 0;
        for(int i = 0; i < ranges_size; i++){
          total_indices = std::max(total_indices, (int)ranges[i].second);
        }
        std::vector<uint32_t> firsts(ranges_size), seconds(ranges_size);
        std::vector<size_t> indices(total_indices);
        for(int i = 0; i < total_indices; i++){
          indices[i] = i;
        }
        for(int i = 0; i < ranges_size; i++){
          firsts[i] = ranges[i].first;
          seconds[i] = ranges[i].second;
        }
        gpu::gpu_malloc((void**)&d_index_it, total_indices * sizeof(size_t));
        gpu::gpu_malloc((void**)&d_density, density.size() * sizeof(char));
        d_bn_exponents.resize(bn_exponents.size());
        h_bn_exponents.resize_host(bn_exponents.size());
        gpu::copy_cpu_to_gpu(d_index_it, indices.data(), sizeof(size_t) * total_indices);
        gpu::copy_cpu_to_gpu(d_firsts, firsts.data(), sizeof(uint32_t) * ranges_size);
        gpu::copy_cpu_to_gpu(d_seconds, seconds.data(), sizeof(uint32_t) * ranges_size);
      }


      auto copy_t = [](const T& src, gpu::alt_bn128_g1& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.x.mont_repr_data + offset, src.X.mont_repr.data, 32);
        gpu::copy_cpu_to_gpu(dst.y.mont_repr_data + offset, src.Y.mont_repr.data, 32);
        gpu::copy_cpu_to_gpu(dst.z.mont_repr_data + offset, src.Z.mont_repr.data, 32);
      };
      auto copy_t_h = [](const T& src, gpu::alt_bn128_g1& dst, const int offset){
        memcpy(dst.x.mont_repr_data + offset, src.X.mont_repr.data, 32);
        memcpy(dst.y.mont_repr_data + offset, src.Y.mont_repr.data, 32);
        memcpy(dst.z.mont_repr_data + offset, src.Z.mont_repr.data, 32);
      };
      auto copy_field = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        gpu::copy_cpu_to_gpu(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };
      auto copy_field_h = [](const FieldT& src, gpu::Fp_model& dst, const int offset){
        memcpy(dst.mont_repr_data + offset, src.mont_repr.data, 32);
      };
      libff::enter_block("gpu copy");
      gpu::copy_cpu_to_gpu(d_modulus.ptr->_limbs, vec_start[0].X.get_modulus().data, 32);
      gpu::copy_cpu_to_gpu(d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32);
      uint64_t const_inv = vec_start[0].X.inv;
      uint64_t const_field_inv = scalar_start[0].inv;

      const auto& modu = vec_start[0].X.get_modulus();
      for(int i = 0; i < values_size; i++){
        copy_t_h(vec_start[i], h_values, i);
      }
      d_values.copy_from_cpu(h_values);
      for(int i = 0; i < scalar_size; i++){
        copy_field_h(scalar_start[i], h_scalars, i);
      }
      d_scalars.copy_from_cpu(h_scalars);
      copy_t(T::zero(), d_t_zero, 0);
      copy_t(T::one(), d_t_one, 0);
      copy_field(zero, d_field_zero, 0);
      copy_field(one, d_field_one, 0);
      libff::leave_block("gpu copy");

      gpu::init_error_report();
      //gpu::warm_up();
      libff::leave_block("gpu init");

      libff::enter_block("gpu reduce sum");
      gpu::alt_bn128_g1_reduce_sum_one_range(
          d_values, 
          d_scalars, 
          d_index_it, 
          d_partial, 
          d_counters, 
          flags,
          ranges_size, 
          d_firsts, d_seconds,
          dmax_value.ptr, d_t_zero, d_field_zero, d_field_one, d_density, d_bn_exponents.ptr, 
          d_modulus.ptr, const_inv, d_field_modulus.ptr, const_field_inv, max_depth);

      auto copy_back = [&](T& dst, const gpu::alt_bn128_g1& src, const int offset){
        gpu::copy_gpu_to_cpu(dst.X.mont_repr.data, src.x.mont_repr_data + offset, 32);
        gpu::copy_gpu_to_cpu(dst.Y.mont_repr.data, src.y.mont_repr_data + offset, 32);
        gpu::copy_gpu_to_cpu(dst.Z.mont_repr.data, src.z.mont_repr_data + offset, 32);
      };

      T gpu_acc;
      copy_back(gpu_acc, d_partial, 0);
      libff::leave_block("gpu reduce sum");

      libff::enter_block("gpu multi exp with density");
      //auto tmp = gpu_acc + multi_exp_with_density<T, FieldT, true, Method>(vec_start, vec_end, bn_exponents, density, config);
      gpu::copy_cpu_to_gpu(d_bn_exponents.ptr, bn_exponents.data(), 32 * bn_exponents.size());
      auto tmp = gpu_acc + libff::multi_exp_with_density_gpu<T, FieldT, true, Method>(vec_start, vec_end, bn_exponents, density, config, d_values, d_density, d_bn_exponents, dmax_value, d_modulus, d_t_zero, d_values2, d_buckets, d_buckets2, d_block_sums, d_block_sums2, gpu_bucket_counters, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids);
      libff::leave_block("gpu multi exp with density");
      libff::leave_block("call gpu");

      if(false){
        d_values.release();
        d_partial.release();
        for(int i = 0; i < chunks; i++){
          d_values2[i].release();
          d_buckets[i].release();
          d_buckets2[i].release();
          d_block_sums[i].release();
          d_block_sums2[i].release();
        }
        d_scalars.release();
        d_field_one.release();
        d_field_zero.release();
        gpu::gpu_free(d_density);
        gpu::gpu_free(flags);
        gpu::gpu_free(d_counters);
        gpu::gpu_free(d_counters2);
        gpu::gpu_free(d_index_it);
        gpu::gpu_free(d_firsts);
        gpu::gpu_free(d_seconds);
        gpu::gpu_free(gpu_bucket_counters);
        gpu::gpu_free(gpu_starts);
        gpu::gpu_free(gpu_indexs);
        gpu::gpu_free(gpu_ids);
        gpu::gpu_free(gpu_instance_bucket_ids);
        d_bn_exponents.release();
        dmax_value.release();
        d_modulus.release();
        d_field_modulus.release();
        d_t_zero.release();
      }
      return tmp;
    }
    leave_block("Process scalar vector");
    //return cpu_result;
}

template <typename T>
T inner_product(typename std::vector<T>::const_iterator a_start,
                typename std::vector<T>::const_iterator a_end,
                typename std::vector<T>::const_iterator b_start,
                typename std::vector<T>::const_iterator b_end)
{
    return multi_exp<T, T, multi_exp_method_naive_plain>(
        a_start, a_end,
        b_start, b_end, 1);
}

#ifdef CURVE_MCL_BN128
#define FIXED_BASE_EXP_WINDOW_TABLE fixed_base_exp_window_table()
#else
#define FIXED_BASE_EXP_WINDOW_TABLE fixed_base_exp_window_table
#endif

template<typename T>
size_t get_exp_window_size(const size_t num_scalars)
{
    if (T::FIXED_BASE_EXP_WINDOW_TABLE.empty())
    {
#ifdef LOWMEM
        return 14;
#else
        return 17;
#endif
    }
    size_t window = 1;
    for (long i = T::FIXED_BASE_EXP_WINDOW_TABLE.size()-1; i >= 0; --i)
    {
#ifdef DEBUG
        if (!inhibit_profiling_info)
        {
            //printf("%ld %zu %zu\n", i, num_scalars, T::fixed_base_exp_window_table[i]);
        }
#endif
        if (T::FIXED_BASE_EXP_WINDOW_TABLE[i] != 0 && num_scalars >= T::FIXED_BASE_EXP_WINDOW_TABLE[i])
        {
            window = i+1;
            break;
        }
    }

    if (!inhibit_profiling_info)
    {
        print_indent(); printf("Choosing window size %zu for %zu elements\n", window, num_scalars);
    }

#ifdef LOWMEM
    window = std::min((size_t)14, window);
#endif
    return window;
}

template<typename T>
window_table<T> get_window_table(const size_t scalar_size,
                                 const size_t window,
                                 const T &g)
{
    const size_t in_window = 1ul<<window;
    const size_t outerc = (scalar_size+window-1)/window;
    const size_t last_in_window = 1ul<<(scalar_size - (outerc-1)*window);
#ifdef DEBUG
    if (!inhibit_profiling_info)
    {
        print_indent(); printf("* scalar_size=%zu; window=%zu; in_window=%zu; outerc=%zu\n", scalar_size, window, in_window, outerc);
    }
#endif

    window_table<T> powers_of_g(outerc, std::vector<T>(in_window, T::zero()));

    T gouter = g;

    for (size_t outer = 0; outer < outerc; ++outer)
    {
        T ginner = T::zero();
        size_t cur_in_window = outer == outerc-1 ? last_in_window : in_window;
        for (size_t inner = 0; inner < cur_in_window; ++inner)
        {
            powers_of_g[outer][inner] = ginner;
            ginner = ginner + gouter;
        }

        for (size_t i = 0; i < window; ++i)
        {
            gouter = gouter + gouter;
        }
    }

    return powers_of_g;
}

template<typename T, typename FieldT>
T windowed_exp(const size_t scalar_size,
               const size_t window,
               const window_table<T> &powers_of_g,
               const FieldT &pow)
{
    const size_t outerc = (scalar_size+window-1)/window;
    const bigint<FieldT::num_limbs> pow_val = pow.as_bigint();

    /* exp */
    T res = powers_of_g[0][0];

    for (size_t outer = 0; outer < outerc; ++outer)
    {
        size_t inner = 0;
        for (size_t i = 0; i < window; ++i)
        {
            if (pow_val.test_bit(outer*window + i))
            {
                inner |= 1u << i;
            }
        }

        res = res + powers_of_g[outer][inner];
    }

    return res;
}

template<typename T, typename FieldT>
std::vector<T> batch_exp(const size_t scalar_size,
                         const size_t window,
                         const window_table<T> &table,
                         const std::vector<FieldT> &v)
{
    if (!inhibit_profiling_info)
    {
        print_indent();
    }
    std::vector<T> res(v.size(), table[0][0]);

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < v.size(); ++i)
    {
        res[i] = windowed_exp(scalar_size, window, table, v[i]);

        if (!inhibit_profiling_info && (i % 10000 == 0))
        {
            printf(".");
            fflush(stdout);
        }
    }

    if (!inhibit_profiling_info)
    {
        printf(" DONE!\n");
    }

    return res;
}

template<typename T, typename FieldT>
std::vector<T> batch_exp_with_coeff(const size_t scalar_size,
                                    const size_t window,
                                    const window_table<T> &table,
                                    const FieldT &coeff,
                                    const std::vector<FieldT> &v)
{
    if (!inhibit_profiling_info)
    {
        print_indent();
    }
    std::vector<T> res(v.size(), table[0][0]);

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < v.size(); ++i)
    {
        res[i] = windowed_exp(scalar_size, window, table, coeff * v[i]);

        if (!inhibit_profiling_info && (i % 10000 == 0))
        {
            printf(".");
            fflush(stdout);
        }
    }

    if (!inhibit_profiling_info)
    {
        printf(" DONE!\n");
    }

    return res;
}

template<typename T>
void batch_to_special(std::vector<T> &vec)
{
    enter_block("Batch-convert elements to special form");

    std::vector<T> non_zero_vec;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (!vec[i].is_zero())
        {
            non_zero_vec.emplace_back(vec[i]);
        }
    }

    T::batch_to_special_all_non_zeros(non_zero_vec);
    auto it = non_zero_vec.begin();
    T zero_special = T::zero();
    zero_special.to_special();

    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (!vec[i].is_zero())
        {
            vec[i] = *it;
            ++it;
        }
        else
        {
            vec[i] = zero_special;
        }
    }
    leave_block("Batch-convert elements to special form");
}

} // libff

#endif // MULTIEXP_TCC_
