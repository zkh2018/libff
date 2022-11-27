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

#ifdef USE_GPU
template<typename T, typename FieldT, bool with_density>
void multi_exp_inner_bellman_with_density_gpu_mcl(
    typename std::vector<T>::const_iterator bases,
    typename std::vector<T>::const_iterator bases_end,
    unsigned int c,
    unsigned int k,
    unsigned int prefetch_stride,
    unsigned int look_ahead,
    gpu::mcl_bn128_g1 d_values,
    char* d_density,
    gpu::gpu_buffer d_bn_exponents,
    int *gpu_bucket_counters, 
    int *gpu_starts, int *gpu_indexs, int* gpu_ids,
    int* gpu_instance_bucket_ids,
    gpu::mcl_bn128_g1 d_values2, 
    gpu::mcl_bn128_g1 d_buckets, 
    gpu::mcl_bn128_g1 d_buckets2, 
    gpu::mcl_bn128_g1 d_t_zero, 
    gpu::mcl_bn128_g1 d_block_sums, 
    gpu::mcl_bn128_g1 d_block_sums2,
    gpu::gpu_buffer d_max_value, 
    gpu::gpu_buffer d_modulus,
    gpu::Fp_model d_one, 
    gpu::Fp_model d_p,
    gpu::Fp_model d_a,
    cudaStream_t stream
    ){
    size_t length = bases_end - bases;

    const int instances = gpu::BUCKET_INSTANCES;
    //auto f = [](clock_t t1, clock_t t0){
    //  return (double)(t1-t0)/CLOCKS_PER_SEC;
    //};
    auto copy_back = [&](T& dst, const gpu::mcl_bn128_g1& src, const int offset){
        uint64_t tmp[4];
        memcpy(tmp, src.x.mont_repr_data + offset, 32);
        dst.pt.x.copy(tmp);
        memcpy(tmp, src.y.mont_repr_data + offset, 32);
        dst.pt.y.copy(tmp);
        memcpy(tmp, src.z.mont_repr_data + offset, 32);
        dst.pt.z.copy(tmp);
    };

    auto copy_back_d = [&](T& dst, const gpu::mcl_bn128_g1& src, const int offset, gpu::CudaStream stream){
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
    };

    auto copy_t_h = [&](T& src, const gpu::mcl_bn128_g1& dst, const int offset){
        memcpy(dst.x.mont_repr_data + offset, src.pt.x.getUnit(), 32);
        memcpy(dst.y.mont_repr_data + offset, src.pt.y.getUnit(), 32);
        memcpy(dst.z.mont_repr_data + offset, src.pt.z.getUnit(), 32);
    };

    auto copy_t = [](const T& src, gpu::mcl_bn128_g1& dst, const int offset, gpu::CudaStream stream){
        gpu::copy_cpu_to_gpu(dst.x.mont_repr_data + offset, src.pt.x.getUnit(), 32, stream);
        gpu::copy_cpu_to_gpu(dst.y.mont_repr_data + offset, src.pt.y.getUnit(), 32, stream);
        gpu::copy_cpu_to_gpu(dst.z.mont_repr_data + offset, src.pt.z.getUnit(), 32, stream);
    };

    auto f = [](const T& a){
        for(int i = 0; i < 4; i++){
            printf("%lu ", a.pt.x.getUnit()[i]);
        }
        printf("\n");
        for(int i = 0; i < 4; i++){
            printf("%lu ", a.pt.y.getUnit()[i]);
        }
        printf("\n");
        for(int i = 0; i < 4; i++){
            printf("%lu ", a.pt.z.getUnit()[i]);
        }
        printf("\n");
    };

    if(true){
      const int specialA_ = bases[0].pt.specialA_;
      const int mode_ = bases[0].pt.mode_;
      const uint64_t rp = bases[0].pt.x.getOp().rp;

      //clock_t t0 = clock();
      //printf("1\n"); 
      gpu::gpu_set_zero(gpu_bucket_counters, (1<<c)*sizeof(int), stream);
      //gpu::sync_device();
      gpu::bucket_counter(with_density, d_density, d_bn_exponents.ptr, c, k, length, (1<<c), gpu_bucket_counters, stream);
      //gpu::sync_device();

      gpu::gpu_set_zero(gpu_starts, (1<<c)*sizeof(int), stream);
      gpu::mcl_prefix_sum(gpu_bucket_counters, gpu_starts, 1<<c, stream);
      //gpu::sync_device();
      //printf("3\n");
      gpu::copy_gpu_to_gpu(gpu_indexs, gpu_starts, (1<<c) * sizeof(int), stream);
      //gpu::sync_device();
      //printf("4\n");
      //
      //d_values2.clear(stream);
      gpu::mcl_split_to_bucket(d_values, d_values2, with_density, d_density, d_bn_exponents.ptr, c, k, length, gpu_starts, gpu_indexs, gpu_instance_bucket_ids, stream);
      //gpu::sync_device();
      //printf("3 compare result = %d\n", cmp_ret);

      //d_buckets.clear(stream);
      //gpu::sync_device();
      gpu::mcl_bucket_reduce_sum(d_values2, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids, d_buckets, 1<<c, length, d_t_zero, d_one, d_p, d_a, specialA_, mode_, rp, stream);

      //gpu::sync_device();
      gpu::mcl_reverse(d_buckets, d_buckets2, 1<<c, 1, stream);
      //gpu::sync_device();
      //printf("7\n");
      //clock_t t2 = clock();
      //d_block_sums.clear(stream);
      //d_block_sums2.clear(stream);
      gpu::mcl_prefix_sum(d_buckets2, d_block_sums, d_block_sums2, 1<<c, d_one, d_p, d_a, specialA_, mode_, rp, stream);
      //gpu::sync_device();
      //printf("8\n");
      //clock_t t3 = clock();
      gpu::mcl_bn128_g1_reduce_sum2(d_buckets2, d_buckets, (1<<c), d_one, d_p, d_a, specialA_, mode_, rp, stream);
      //gpu::sync_device();
      //printf("9\n");
      //clock_t t4 = clock();
      //printf("%f %f %f %f\n", f(t1, t0), f(t2, t1), f(t3, t2), f(t4, t3));
      //return tmp_result;
    }
}

template<typename T, typename FieldT, bool with_density, multi_exp_method Method>
T multi_exp_with_density_gpu_mcl(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            const libsnark::Config& config,
            gpu::mcl_bn128_g1 d_values,
            char* d_density,
            gpu::gpu_buffer d_bn_exponents,
            gpu::gpu_buffer d_max_value,
            gpu::gpu_buffer d_modulus,
            gpu::mcl_bn128_g1 d_t_zero,
            std::vector<gpu::mcl_bn128_g1> d_values2,
            std::vector<gpu::mcl_bn128_g1> d_buckets,
            std::vector<gpu::mcl_bn128_g1> d_buckets2,
            std::vector<gpu::mcl_bn128_g1> d_block_sums,
            std::vector<gpu::mcl_bn128_g1> d_block_sums2,
            int* gpu_bucket_counters,
            int* gpu_starts, 
            int *gpu_indexs,
            int *gpu_ids,
            int* gpu_instance_bucket_ids,
            gpu::Fp_model d_one, 
            gpu::Fp_model d_p,
            gpu::Fp_model d_a,
            cudaStream_t stream
            )
{
    unsigned int chunks = config.num_threads;
    const size_t total = vec_end - vec_start;

    unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
    chunks = (254 + c - 1) / c;

    auto ranges = libsnark::get_cpu_ranges(0, total);
    std::vector<T> partial(chunks, T::zero());

    const int length = total;

    auto copy_back = [&](T& dst, const gpu::mcl_bn128_g1& src, const int offset, cudaStream_t& stream){
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
    };

    //const int n = 1 << c;
    for (size_t i = 0; i < chunks; ++i)
    {
      //const int offset = i * n;
      multi_exp_inner_bellman_with_density_gpu_mcl<T, FieldT, with_density>(
          vec_start, vec_end, c, i, config.prefetch_stride, config.multi_exp_look_ahead,
          d_values, d_density, d_bn_exponents,
          gpu_bucket_counters, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids,
          d_values2[0], d_buckets[0], d_buckets2[0], d_t_zero, 
          d_block_sums[0], d_block_sums2[0],
          d_max_value, d_modulus, d_one, d_p, d_a, stream);
      copy_back(partial[i], d_buckets[0], 0, stream);
    }

    //copy_back(partial[chunks-1], d_buckets[chunks-1], 0);
    //gpu::sync_device();
    gpu::sync(stream);
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
T multi_exp_with_density_gpu_mcl_at(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            const libsnark::Config& config,
            gpu::mcl_bn128_g1 d_values,
            char* d_density,
            gpu::gpu_buffer d_bn_exponents,
            gpu::gpu_buffer d_max_value,
            gpu::gpu_buffer d_modulus,
            gpu::mcl_bn128_g1 d_t_zero,
            std::vector<gpu::mcl_bn128_g1> d_values2,
            std::vector<gpu::mcl_bn128_g1> d_buckets,
            std::vector<gpu::mcl_bn128_g1> d_buckets2,
            std::vector<gpu::mcl_bn128_g1> d_block_sums,
            std::vector<gpu::mcl_bn128_g1> d_block_sums2,
            int* gpu_bucket_counters,
            int* gpu_starts, 
            int *gpu_indexs,
            int *gpu_ids,
            int* gpu_instance_bucket_ids,
            gpu::Fp_model d_one, 
            gpu::Fp_model d_p,
            gpu::Fp_model d_a,
            cudaStream_t stream
            )
{
    unsigned int chunks = config.num_threads;
    const size_t total = vec_end - vec_start;

    //unsigned int c = config.multi_exp_c == 0 ? 15 : config.multi_exp_c;
    unsigned int c = 1;
    chunks = (254 + c - 1) / c;

    auto ranges = libsnark::get_cpu_ranges(0, total);
    std::vector<T> partial(chunks, T::zero());

    const int length = total;

    auto copy_back = [&](T& dst, const gpu::mcl_bn128_g1& src, const int offset, cudaStream_t& stream){
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
    };

    const int specialA_ = vec_start[0].pt.specialA_;
    const int mode_ = vec_start[0].pt.mode_;
    const uint64_t rp = vec_start[0].pt.x.getOp().rp;
    //const int n = 1 << c;
    for (size_t i = 0; i < chunks; ++i)
    {
      //multi_exp_inner_bellman_with_density_gpu_mcl<T, FieldT, with_density>(
      //    vec_start, vec_end, c, i, config.prefetch_stride, config.multi_exp_look_ahead,
      //    d_values, d_density, d_bn_exponents,
      //    gpu_bucket_counters, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids,
      //    d_values2[0], d_buckets[0], d_buckets2[0], d_t_zero, 
      //    d_block_sums[0], d_block_sums2[0],
      //    d_max_value, d_modulus, d_one, d_p, d_a, stream);
      gpu::multi_exp_inner_bellman_with_density(d_values, d_bn_exponents.ptr, d_buckets[0], d_buckets2[0], c, i, length, d_one, d_p, d_a, specialA_, mode_, rp, stream); 
      copy_back(partial[i], d_buckets[0], 0, stream);
    }

    //copy_back(partial[chunks-1], d_buckets[chunks-1], 0);
    //gpu::sync_device();
    gpu::sync(stream);
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
void multi_exp_inner_bellman_with_density_gpu_mcl_g2(
    typename std::vector<T>::const_iterator bases,
    typename std::vector<T>::const_iterator bases_end,
    const std::vector<bigint<4>>& exponents,
    const std::vector<bool>& density,
    unsigned int c,
    unsigned int k,
    unsigned int prefetch_stride,
    unsigned int look_ahead,
    gpu::mcl_bn128_g2 d_values,
    char* d_density,
    gpu::gpu_buffer d_bn_exponents,
    int *gpu_bucket_counters, 
    int *gpu_starts, int *gpu_indexs, int* gpu_ids,
    int* gpu_instance_bucket_ids,
    gpu::mcl_bn128_g2 d_values2, 
    gpu::mcl_bn128_g2 d_buckets, 
    gpu::mcl_bn128_g2 d_buckets2, 
    gpu::mcl_bn128_g2 d_t_zero, 
    gpu::mcl_bn128_g2 d_block_sums, 
    gpu::mcl_bn128_g2 d_block_sums2,
    gpu::gpu_buffer d_max_value, 
    gpu::gpu_buffer d_modulus,
    gpu::Fp_model d_one, 
    gpu::Fp_model d_p,
    gpu::Fp_model2 d_a,
    cudaStream_t stream
    ){
    size_t length = bases_end - bases;

    const int instances = gpu::BUCKET_INSTANCES;
    auto f = [](clock_t t1, clock_t t0){
      return (double)(t1-t0)/CLOCKS_PER_SEC;
    };
    auto print = [](const T& data){
        for(int i = 0; i < 4; i++){
            printf("%lu ", data.pt.x.a.getUnit()[i]);
        }
        printf("\n");
        for(int i = 0; i < 4; i++){
            printf("%lu ", data.pt.x.b.getUnit()[i]);
        }
        printf("\n");
        for(int i = 0; i < 4; i++){
            printf("%lu ", data.pt.y.a.getUnit()[i]);
        }
        printf("\n");
        for(int i = 0; i < 4; i++){
            printf("%lu ", data.pt.y.b.getUnit()[i]);
        }
        printf("\n");
        for(int i = 0; i < 4; i++){
            printf("%lu ", data.pt.z.a.getUnit()[i]);
        }
        printf("\n");
        for(int i = 0; i < 4; i++){
            printf("%lu ", data.pt.z.b.getUnit()[i]);
        }
        printf("\n");
    };
    auto copy_back = [&](T& dst, const gpu::mcl_bn128_g2& src, const int offset, cudaStream_t& stream){
        uint64_t tmp[4];
        gpu::copy_gpu_to_cpu(tmp, src.x.c0.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.x.a.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.x.c1.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.x.b.copy(tmp);

        gpu::copy_gpu_to_cpu(tmp, src.y.c0.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.y.a.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.y.c1.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.y.b.copy(tmp);

        gpu::copy_gpu_to_cpu(tmp, src.z.c0.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.z.a.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.z.c1.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.z.b.copy(tmp);
    };
    auto copy_back_h = [&](T& dst, const gpu::mcl_bn128_g2& src, const int offset){
        dst.pt.x.a.copy((uint64_t*)(src.x.c0.mont_repr_data + offset));
        dst.pt.x.b.copy((uint64_t*)(src.x.c1.mont_repr_data + offset));
        dst.pt.y.a.copy((uint64_t*)(src.y.c0.mont_repr_data + offset));
        dst.pt.y.b.copy((uint64_t*)(src.y.c1.mont_repr_data + offset));
        dst.pt.z.a.copy((uint64_t*)(src.z.c0.mont_repr_data + offset));
        dst.pt.z.b.copy((uint64_t*)(src.z.c1.mont_repr_data + offset));
    };

    if(true){
      const int specialA_ = bases[0].pt.specialA_;
      const int mode_ = bases[0].pt.mode_;
      const uint64_t rp = bases[0].pt.x.a.getOp().rp;

      //clock_t t0 = clock();
      //printf("1\n"); 
      gpu::gpu_set_zero(gpu_bucket_counters, (1<<c)*sizeof(int), stream);
      //gpu::sync_device();
      gpu::bucket_counter(with_density, d_density, d_bn_exponents.ptr, c, k, length, (1<<c), gpu_bucket_counters, stream);
      //gpu::sync_device();

      gpu::gpu_set_zero(gpu_starts, (1<<c)*sizeof(int), stream);
      gpu::mcl_prefix_sum(gpu_bucket_counters, gpu_starts, 1<<c, stream);
      //gpu::sync_device();
      //printf("3\n");
      gpu::copy_gpu_to_gpu(gpu_indexs, gpu_starts, (1<<c) * sizeof(int), stream);
      //gpu::sync_device();
      //printf("4\n");
      //
      //d_values2.clear(stream);
      gpu::mcl_split_to_bucket_g2(d_values, d_values2, with_density, d_density, d_bn_exponents.ptr, c, k, length, gpu_starts, gpu_indexs, gpu_instance_bucket_ids, stream);

      gpu::mcl_bucket_reduce_sum_g2(d_values2, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids, d_buckets, 1<<c, length, d_t_zero, d_one, d_p, d_a, specialA_, mode_, rp, stream);

      gpu::mcl_reverse_g2(d_buckets, d_buckets2, 1<<c, 1, stream);

      gpu::mcl_prefix_sum_g2(d_buckets2, d_block_sums, d_block_sums2, 1<<c, d_one, d_p, d_a, specialA_, mode_, rp, stream);

      gpu::mcl_bn128_g2_reduce_sum2(d_buckets2, d_buckets, (1<<c), d_one, d_p, d_a, specialA_, mode_, rp, stream);
    }
}

template<typename T, typename FieldT, bool with_density, multi_exp_method Method>
T multi_exp_with_density_gpu_mcl_g2(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            const std::vector<bigint<FieldT::num_limbs>>& exponents,
            const std::vector<bool>& density,
            const libsnark::Config& config,
            gpu::mcl_bn128_g2 d_values,
            char* d_density,
            gpu::gpu_buffer d_bn_exponents,
            gpu::gpu_buffer d_max_value,
            gpu::gpu_buffer d_modulus,
            gpu::mcl_bn128_g2 d_t_zero,
            std::vector<gpu::mcl_bn128_g2> d_values2,
            std::vector<gpu::mcl_bn128_g2> d_buckets,
            std::vector<gpu::mcl_bn128_g2> d_buckets2,
            std::vector<gpu::mcl_bn128_g2> d_block_sums,
            std::vector<gpu::mcl_bn128_g2> d_block_sums2,
            int* gpu_bucket_counters,
            int* gpu_starts, 
            int *gpu_indexs,
            int *gpu_ids,
            int* gpu_instance_bucket_ids,
            gpu::Fp_model d_one, 
            gpu::Fp_model d_p,
            gpu::Fp_model2 d_a,
            cudaStream_t stream
            )
{
    unsigned int chunks = config.num_threads;
    const size_t total = vec_end - vec_start;

    unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
    chunks = (254 + c - 1) / c;

    auto ranges = libsnark::get_cpu_ranges(0, total);
    std::vector<T> partial(chunks, T::zero());

    const int length = total;

    auto copy_back = [&](T& dst, const gpu::mcl_bn128_g2& src, const int offset, cudaStream_t& stream){
        uint64_t tmp[4];
        gpu::copy_gpu_to_cpu(tmp, src.x.c0.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.x.a.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.x.c1.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.x.b.copy(tmp);

        gpu::copy_gpu_to_cpu(tmp, src.y.c0.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.y.a.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.y.c1.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.y.b.copy(tmp);

        gpu::copy_gpu_to_cpu(tmp, src.z.c0.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.z.a.copy(tmp);
        gpu::copy_gpu_to_cpu(tmp, src.z.c1.mont_repr_data + offset, 32, stream);
        gpu::sync(stream);
        dst.pt.z.b.copy(tmp);
    };

    auto copy_gpu_to_cpu = [](gpu::mcl_bn128_g2& dst, const gpu::mcl_bn128_g2& src, const int n, cudaStream_t& stream){
        gpu::copy_gpu_to_cpu(dst.x.c0.mont_repr_data, src.x.c0.mont_repr_data, n * 32, stream);
        gpu::copy_gpu_to_cpu(dst.x.c1.mont_repr_data, src.x.c1.mont_repr_data, n * 32, stream);
        gpu::copy_gpu_to_cpu(dst.y.c0.mont_repr_data, src.y.c0.mont_repr_data, n * 32, stream);
        gpu::copy_gpu_to_cpu(dst.y.c1.mont_repr_data, src.y.c1.mont_repr_data, n * 32, stream);
        gpu::copy_gpu_to_cpu(dst.z.c0.mont_repr_data, src.z.c0.mont_repr_data, n * 32, stream);
        gpu::copy_gpu_to_cpu(dst.z.c1.mont_repr_data, src.z.c1.mont_repr_data, n * 32, stream);
    };

    //std::vector<gpu::mcl_bn128_g2> h_buckets(chunks);
    
    for (size_t i = 0; i < chunks; ++i)
    {
        //h_buckets[i].init_host((1<<c));
        multi_exp_inner_bellman_with_density_gpu_mcl_g2<T, FieldT, with_density>(
                vec_start, vec_end, exponents, density, c, i, config.prefetch_stride, config.multi_exp_look_ahead,
                d_values, d_density, d_bn_exponents,
                gpu_bucket_counters, gpu_starts, gpu_indexs, gpu_ids, gpu_instance_bucket_ids,
                d_values2[0], d_buckets[0], d_buckets2[0], d_t_zero, 
                d_block_sums[0], d_block_sums2[0],
                d_max_value, d_modulus, d_one, d_p, d_a, stream);
        copy_back(partial[i], d_buckets[0], 0, stream);
        //copy_gpu_to_cpu(h_buckets[i], d_buckets2[0], (1<<c)-1, stream);
        //gpu::sync(stream);
    }

    gpu::sync(stream);
    auto copy_back_h = [&](T& dst, const gpu::mcl_bn128_g2& src, const int offset){
        dst.pt.x.a.copy((uint64_t*)(src.x.c0.mont_repr_data + offset));
        dst.pt.x.b.copy((uint64_t*)(src.x.c1.mont_repr_data + offset));
        dst.pt.y.a.copy((uint64_t*)(src.y.c0.mont_repr_data + offset));
        dst.pt.y.b.copy((uint64_t*)(src.y.c1.mont_repr_data + offset));
        dst.pt.z.a.copy((uint64_t*)(src.z.c0.mont_repr_data + offset));
        dst.pt.z.b.copy((uint64_t*)(src.z.c1.mont_repr_data + offset));
    };

    //copy_back(partial[chunks-1], d_buckets[chunks-1], 0);
    //gpu::sync_device();
    //gpu::sync(stream);
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

#endif

template<typename T, typename FieldT, bool with_density, multi_exp_method Method>
T multi_exp_with_density(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            const std::vector<bigint<FieldT::num_limbs>>& exponents,
            const std::vector<bool>& density,
            const libsnark::Config& config)
{
    unsigned int chunks = config.num_threads;
    const size_t total = vec_end - vec_start;

    unsigned int c = config.multi_exp_c == 0 ? 15: config.multi_exp_c;
    chunks = (254 + c - 1) / c;

    auto ranges = libsnark::get_cpu_ranges(0, total);
    std::vector<T> partial(chunks, T::zero());
//#ifdef MULTICORE
//#pragma omp parallel for
//#endif
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

#ifdef USE_GPU
template<typename T, typename FieldT, multi_exp_method Method>
void multi_exp_gpu_mcl_preprocess(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            gpu::Buffer<>& d_scalars,
            typename std::vector<FieldT>::const_iterator scalar_start,
            typename std::vector<FieldT>::const_iterator scalar_end,
            std::vector<bigint<FieldT::num_limbs>>& scratch_exponents,
            const libsnark::Config& config,
            GpuMclData<T, FieldT>& gpu_mcl_data)
{
    const size_t total = vec_end - vec_start;
    // Dummy density vector
    std::vector<bool> density;
    unsigned int c = config.multi_exp_c == 0 ? 16 : config.multi_exp_c;
    unsigned int chunks = config.num_threads;
    chunks = 1;
    const int instances = gpu::BUCKET_INSTANCES;
    const int length = total;
    const int values_size = total;

    cudaStream_t& stream = gpu_mcl_data.stream;
    libff::copy_t<T, FieldT>(T::zero(), gpu_mcl_data.d_t_zero, 0, stream);
    gpu::copy_cpu_to_gpu(gpu_mcl_data.d_one.mont_repr_data, vec_start[0].pt.z.one().getUnit(), 32, stream);
    gpu::copy_cpu_to_gpu(gpu_mcl_data.d_p.mont_repr_data, vec_start[0].pt.z.getOp().p, 32, stream);
    gpu::copy_cpu_to_gpu(gpu_mcl_data.d_a.mont_repr_data, vec_start[0].pt.a_.getUnit(), 32, stream);

    gpu_mcl_data.h_values.resize_host(values_size);
    gpu_mcl_data.d_values.resize(values_size);
    //gpu_mcl_data.d_bn_exponents.resize(bn_exponents.size());
    //gpu_mcl_data.h_bn_exponents.resize_host(bn_exponents.size());
    for(int i = 0; i < chunks; i++){
      gpu_mcl_data.d_values2[i].resize(length);
      gpu_mcl_data.d_buckets[i].resize((1<<c));
      gpu_mcl_data.d_buckets2[i].resize((1<<c));
      gpu_mcl_data.d_block_sums[i].resize((1<<c) / 32);
      gpu_mcl_data.d_block_sums2[i].resize((1<<c) / 32/32);
    }
    gpu_mcl_data.d_bucket_counters.resize((1<<c) * sizeof(int) * chunks);
    gpu_mcl_data.d_starts.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
    gpu_mcl_data.d_indexs.resize((1<<c)*sizeof(int)*chunks*2);
    gpu_mcl_data.d_ids.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
    gpu_mcl_data.d_instance_bucket_ids.resize((length+1) * sizeof(int) * chunks * 2);
    gpu_mcl_data.d_density.resize(density.size());

    //memcpy(gpu_mcl_data.h_bn_exponents.ptr, bn_exponents.data(), 32 * bn_exponents.size());
    uint64_t const_field_inv = FieldT::inv;
    gpu_mcl_data.d_bn_exponents.ptr = (cgbn_mem_t<BITS>*)d_scalars.ptr_;
    gpu_mcl_data.d_bn_exponents.n = d_scalars.len_;
    //gpu_mcl_data.d_bn_exponents.copy_from_host(gpu_mcl_data.h_bn_exponents, gpu_mcl_data.stream);
    gpu::copy_cpu_to_gpu(gpu_mcl_data.d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32, stream);
    gpu::mcl_as_bigint(d_scalars, gpu_mcl_data.d_bn_exponents.ptr, total, gpu_mcl_data.d_field_modulus.ptr, const_field_inv, gpu_mcl_data.stream); 
    //d_scalars.ptr_ = nullptr;
    //d_scalars.len_ = 0;
    //gpu::sync(gpu_mcl_data.stream);
}

//T = mcl_bn128_g1
template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp_gpu_mcl(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            typename std::vector<FieldT>::const_iterator scalar_start,
            typename std::vector<FieldT>::const_iterator scalar_end,
            std::vector<bigint<FieldT::num_limbs>>& scratch_exponents,
            const libsnark::Config& config,
            GpuMclData<T, FieldT>& gpu_mcl_data)
{
  const int values_size = vec_end - vec_start;
#pragma omp parallel for
    for(int i = 0; i < values_size; i++){
      libff::copy_t_h<T, FieldT>(vec_start[i], gpu_mcl_data.h_values, i);
    }
    gpu_mcl_data.d_values.copy_from_cpu(gpu_mcl_data.h_values, gpu_mcl_data.stream);
    T gpu_result = multi_exp_with_density_gpu_mcl<T, FieldT, false, Method>(vec_start, vec_end, config, gpu_mcl_data.d_values, (char*)gpu_mcl_data.d_density.ptr, gpu_mcl_data.d_bn_exponents, gpu_mcl_data.dmax_value, gpu_mcl_data.d_modulus, gpu_mcl_data.d_t_zero, gpu_mcl_data.d_values2, gpu_mcl_data.d_buckets, gpu_mcl_data.d_buckets2, gpu_mcl_data.d_block_sums, gpu_mcl_data.d_block_sums2, (int*)gpu_mcl_data.d_bucket_counters.ptr, (int*)gpu_mcl_data.d_starts.ptr, (int*)gpu_mcl_data.d_indexs.ptr, (int*)gpu_mcl_data.d_ids.ptr, (int*)gpu_mcl_data.d_instance_bucket_ids.ptr, gpu_mcl_data.d_one, gpu_mcl_data.d_p, gpu_mcl_data.d_a, gpu_mcl_data.stream);
    return gpu_result;
}

#endif

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

if(false){
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
}else{
    T acc = T::zero();
    for(int i = 0; i < scalar_length; i++){
            if (scalar_start[i] == zero)
            {
                // do nothing
                //++num_skip;
            }
            else if (scalar_start[i] == one)
            {
                acc = acc + vec_start[i];
                //++num_add;
            }
            else
            {
                //if (vec_start[i] != T::zero())
                {
                    density[i] = true;
                    bn_exponents[i] = scalar_start[i].as_bigint();
                }
                //++num_other;
            }
    }
    leave_block("Process scalar vector");

    return acc + multi_exp_with_density<T, FieldT, true, Method>(vec_start, vec_end, bn_exponents, density, config);
}

}

#ifdef USE_GPU

template<typename T, typename FieldT, multi_exp_method Method>
void multi_exp_with_mixed_addition_gpu_mcl_preprocess(typename std::vector<T>::const_iterator vec_start,
                                typename std::vector<T>::const_iterator vec_end,
                                typename std::vector<FieldT>::const_iterator scalar_start,
                                typename std::vector<FieldT>::const_iterator scalar_end,
                                std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                const libsnark::Config& config,
                                GpuMclData<T, FieldT>& gpu_mcl_data)
{
    assert(std::distance(vec_start, vec_end) == std::distance(scalar_start, scalar_end));
    //enter_block("Process scalar vector");

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();

    //size_t num_skip = 0;
    //size_t num_add = 0;
    //size_t num_other = 0;

    const size_t scalar_length = std::distance(scalar_start, scalar_end);

    //libff::enter_block("allocate density memory");
    std::vector<bool> density(scalar_length);
    //libff::leave_block("allocate density memory");

    //std::vector<bigint<FieldT::num_limbs>>& bn_exponents = scratch_exponents;
    std::vector<bigint<FieldT::num_limbs>> bn_exponents(scratch_exponents.size());
    if (bn_exponents.size() < scalar_length)
    {
        bn_exponents.resize(scalar_length);
    }

    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);
    //std::vector<T> partial(ranges.size(), T::zero());
    //std::vector<unsigned int> counters(ranges.size(), 0);
    int max_depth = 0;
    for(int i = 0; i < ranges.size(); i++){
      max_depth = std::max(max_depth, (int)(ranges[i].second - ranges[i].first));
    }
    //enter_block("cpu reduce sum");

    //leave_block("cpu reduce sum");

    //enter_block("cpu multi_exp_with_density");
    //leave_block("cpu multi_exp_with_density");
    if(true){
      const int local_instances = BlockDepth * 64;
      const int blocks = (max_depth + local_instances - 1) / local_instances;
      unsigned int c = config.multi_exp_c == 0 ? 15 : config.multi_exp_c;
      //unsigned int chunks = config.num_threads;
      unsigned int chunks = 1;//(254 + c - 1) / c;
      const int instances = gpu::BUCKET_INSTANCES;
      const int scalar_size = scalar_end - scalar_start;
      const int values_size = vec_end - vec_start;
      const int length = values_size;

      const int ranges_size = ranges.size();

      cudaStream_t& stream = gpu_mcl_data.stream;

      libff::copy_t<T, FieldT>(T::zero(), gpu_mcl_data.d_t_zero, 0, stream);
      libff::copy_t<T, FieldT>(T::one(), gpu_mcl_data.d_t_one, 0, stream);
      libff::copy_field<T, FieldT>(zero, gpu_mcl_data.d_field_zero, 0, stream);
      libff::copy_field<T, FieldT>(one, gpu_mcl_data.d_field_one, 0, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_field_modulus.ptr->_limbs, scalar_start[0].get_modulus().data, 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_one.mont_repr_data, vec_start[0].pt.z.one().getUnit(), 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_p.mont_repr_data, vec_start[0].pt.z.getOp().p, 32, stream);
      gpu::copy_cpu_to_gpu(gpu_mcl_data.d_a.mont_repr_data, vec_start[0].pt.a_.getUnit(), 32, stream);

      gpu_mcl_data.h_values.resize_host(values_size);
      gpu_mcl_data.d_values.resize(values_size);
      gpu_mcl_data.h_scalars.resize_host(scalar_size);
      gpu_mcl_data.d_scalars.resize(scalar_size);
      gpu_mcl_data.d_partial.resize(values_size);
      gpu_mcl_data.d_bn_exponents.resize(bn_exponents.size());
      //gpu_mcl_data.h_bn_exponents.resize_host(bn_exponents.size());
      for(int i = 0; i < chunks; i++){
        gpu_mcl_data.d_values2[i].resize(length);
        gpu_mcl_data.d_buckets[i].resize((1<<c));
        gpu_mcl_data.d_buckets2[i].resize((1<<c));
        gpu_mcl_data.d_block_sums[i].resize((1<<c) / 32);
        gpu_mcl_data.d_block_sums2[i].resize((1<<c) / 32/32);
      }

      //gpu_mcl_data.d_counters.resize(ranges_size * max_depth * sizeof(uint32_t));
      //gpu_mcl_data.d_counters2.resize(ranges_size * sizeof(uint32_t));
      //gpu_mcl_data.d_firsts.resize(ranges_size * sizeof(uint32_t));
      //gpu_mcl_data.d_seconds.resize(ranges_size * sizeof(uint32_t));
      gpu_mcl_data.d_bucket_counters.resize((1<<c) * sizeof(int) * chunks);
      gpu_mcl_data.d_starts.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
      gpu_mcl_data.d_indexs.resize((1<<c)*sizeof(int)*chunks*2);
      gpu_mcl_data.d_ids.resize(((1<<c)+1) * sizeof(int) * chunks * 2);
      gpu_mcl_data.d_instance_bucket_ids.resize((length+1) * sizeof(int) * chunks * 2);
      gpu_mcl_data.d_density.resize(density.size());
      gpu_mcl_data.d_flags.resize(scalar_size*4);
      gpu::gpu_set_zero(gpu_mcl_data.d_flags.ptr, scalar_size*4, stream);
      gpu_mcl_data.d_bn_exponents.clear(stream);

      uint64_t const_field_inv = scalar_start[0].inv;

#pragma omp parallel for
      for(int i = 0; i < scalar_size; i++){
        libff::copy_field_h<T, FieldT>(scalar_start[i], gpu_mcl_data.h_scalars, i);
      }
      //gpu::copy_cpu_to_gpu(gpu_mcl_data.d_bn_exponents.ptr, bn_exponents.data(), 32 * bn_exponents.size(), stream);
    }
}

template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp_with_mixed_addition_gpu_mcl(typename std::vector<T>::const_iterator vec_start,
                                typename std::vector<T>::const_iterator vec_end,
                                typename std::vector<FieldT>::const_iterator scalar_start,
                                typename std::vector<FieldT>::const_iterator scalar_end,
                                std::vector<libff::bigint<FieldT::num_limbs>>& scratch_exponents,
                                const libsnark::Config& config,
                                GpuMclData<T, FieldT>& gpu_mcl_data)
{

    const size_t scalar_length = std::distance(scalar_start, scalar_end);
    auto ranges = libsnark::get_cpu_ranges(0, scalar_length);
    //int max_depth = 0;
    //for(int i = 0; i < ranges.size(); i++){
    //  max_depth = std::max(max_depth, (int)(ranges[i].second - ranges[i].first));
    //}
    const int ranges_size = ranges.size();
    uint64_t const_field_inv = scalar_start[0].inv;
    gpu::CudaStream& stream = gpu_mcl_data.stream;

      const int values_size = vec_end - vec_start;
#pragma omp parallel for
      for(int i = 0; i < values_size; i++){
        libff::copy_t_h<T, FieldT>(vec_start[i], gpu_mcl_data.h_values, i);
      }

      gpu_mcl_data.d_values.copy_from_cpu(gpu_mcl_data.h_values, stream);
      gpu_mcl_data.d_scalars.copy_from_cpu(gpu_mcl_data.h_scalars, stream);
      gpu::lt_reduce_sum(
          gpu_mcl_data.d_values, 
          gpu_mcl_data.d_scalars, 
          gpu_mcl_data.d_partial, 
          (char*)gpu_mcl_data.d_flags.ptr,
          gpu_mcl_data.d_t_zero, gpu_mcl_data.d_field_zero, gpu_mcl_data.d_field_one, (char*)gpu_mcl_data.d_density.ptr, gpu_mcl_data.d_bn_exponents.ptr, 
          gpu_mcl_data.d_field_modulus.ptr, const_field_inv, 
          gpu_mcl_data.d_one, gpu_mcl_data.d_p, gpu_mcl_data.d_a, vec_start[0].pt.specialA_, vec_start[0].pt.mode_, vec_start[0].pt.x.getOp().rp,
          scalar_length, stream);

      T gpu_acc;
      libff::copy_back<T, FieldT>(gpu_acc, gpu_mcl_data.d_partial, 0, stream);
      gpu::sync(stream);

      //auto tmp = gpu_acc + multi_exp_with_density<T, FieldT, true, Method>(vec_start, vec_end, bn_exponents, density, config);
      auto tmp = gpu_acc + libff::multi_exp_with_density_gpu_mcl<T, FieldT, true, Method>(vec_start, vec_end, config, gpu_mcl_data.d_values, (char*)gpu_mcl_data.d_density.ptr, gpu_mcl_data.d_bn_exponents, gpu_mcl_data.dmax_value, gpu_mcl_data.d_modulus, gpu_mcl_data.d_t_zero, gpu_mcl_data.d_values2, gpu_mcl_data.d_buckets, gpu_mcl_data.d_buckets2, gpu_mcl_data.d_block_sums, gpu_mcl_data.d_block_sums2, (int*)gpu_mcl_data.d_bucket_counters.ptr, (int*)gpu_mcl_data.d_starts.ptr, (int*)gpu_mcl_data.d_indexs.ptr, (int*)gpu_mcl_data.d_ids.ptr, (int*)gpu_mcl_data.d_instance_bucket_ids.ptr, gpu_mcl_data.d_one, gpu_mcl_data.d_p, gpu_mcl_data.d_a, stream);

      return tmp;
}
#endif

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
