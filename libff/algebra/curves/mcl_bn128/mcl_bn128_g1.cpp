/** @file
 *****************************************************************************
 * @author     This file is part of libsnark, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#include <libff/algebra/curves/mcl_bn128/mcl_bn128_g1.hpp>
#include <libff/algebra/curves/mcl_bn128/bn_utils.hpp>

using namespace mcl::bn256;

namespace libff {

#ifdef PROFILE_OP_COUNTS
long long mcl_bn128_G1::add_cnt = 0;
long long mcl_bn128_G1::dbl_cnt = 0;
#endif

std::vector<size_t>& mcl_bn128_G1::wnaf_window_table() {
  static std::vector<size_t> _wnaf_window_table_;
  return _wnaf_window_table_;
}
std::vector<size_t>& mcl_bn128_G1::fixed_base_exp_window_table() {
  static std::vector<size_t> _fixed_base_exp_window_table_;
  return _fixed_base_exp_window_table_;
}
mcl_bn128_G1 &mcl_bn128_G1::G1_zero() {
  static mcl_bn128_G1 _G1_zero_;
  return _G1_zero_;
}
mcl_bn128_G1 &mcl_bn128_G1::G1_one() {
  static mcl_bn128_G1 _G1_one_;
  return _G1_one_;
}

Fp mcl_bn128_G1::sqrt(const Fp &el)
{
    Fp x;
    bool ok = Fp::squareRoot(x, el);
    assert(ok);
    (void)ok;
    return x;
}

mcl_bn128_G1::mcl_bn128_G1()
{
    this->pt.clear();
}

mcl_bn128_G1::mcl_bn128_G1(const std::string& X, const std::string& Y, const std::string& Z)
{
    this->pt.x = Fp(X);
    this->pt.y = Fp(Y);
    this->pt.z = Fp(Z);
}

void mcl_bn128_G1::print() const
{
    if (this->is_zero())
    {
        printf("O\n");
    }
    else
    {
        auto _pt(pt);
        _pt.normalize();
        std::cout << "(" << _pt.x << " : " << _pt.y << " : " << _pt.z << ")\n";
    }
}

void mcl_bn128_G1::print_coordinates() const
{
    if (this->is_zero())
    {
        printf("O\n");
    }
    else
    {
        std::cout << "(" << pt.x << " : " << pt.y << " : " << pt.z << ")\n";
    }
}

void mcl_bn128_G1::to_affine_coordinates()
{
    pt.normalize();
}

void mcl_bn128_G1::to_special()
{
    this->to_affine_coordinates();
}

bool mcl_bn128_G1::is_special() const
{
    return (this->is_zero() || this->pt.z == 1);
}

bool mcl_bn128_G1::is_zero() const
{
    return pt.isZero();
}

bool mcl_bn128_G1::operator==(const mcl_bn128_G1 &other) const
{
    return this->pt == other.pt;
}

bool mcl_bn128_G1::operator!=(const mcl_bn128_G1& other) const
{
    return !(operator==(other));
}

mcl_bn128_G1 mcl_bn128_G1::gpu_add(const mcl_bn128_G1 &other) const
{
    //return this->add(other);
    mcl_bn128_G1 result;
    //G1::add(result.pt, pt, other.pt);
    G1::gpu_add(result.pt, pt, other.pt);
    return result;
}

mcl_bn128_G1 mcl_bn128_G1::operator+(const mcl_bn128_G1 &other) const
{
    return this->add(other);
}

mcl_bn128_G1 mcl_bn128_G1::operator-() const
{
    mcl_bn128_G1 result;
    G1::neg(result.pt, pt);
    return result;
}

mcl_bn128_G1 mcl_bn128_G1::operator-(const mcl_bn128_G1 &other) const
{
#ifdef PROFILE_OP_COUNTS
    this->add_cnt++;
#endif
    mcl_bn128_G1 result;
    G1::sub(result.pt, pt, other.pt);
    return result;
}

mcl_bn128_G1 mcl_bn128_G1::add(const mcl_bn128_G1 &other) const
{
#ifdef PROFILE_OP_COUNTS
    this->add_cnt++;
#endif
    mcl_bn128_G1 result;
    G1::add(result.pt, pt, other.pt);
    return result;
}

mcl_bn128_G1 mcl_bn128_G1::mixed_add(const mcl_bn128_G1 &other) const
{
    mcl_bn128_G1 result;
    G1::add(result.pt, pt, other.pt);
#ifdef PROFILE_OP_COUNTS
    this->add_cnt++;
#endif
    return result;
}

mcl_bn128_G1 mcl_bn128_G1::dbl() const
{
#ifdef PROFILE_OP_COUNTS
    this->dbl_cnt++;
#endif
    mcl_bn128_G1 result;
    G1::dbl(result.pt, pt);
    return result;
}

mcl_bn128_G1 mcl_bn128_G1::zero()
{
    return G1_zero();
}

mcl_bn128_G1 mcl_bn128_G1::one()
{
    return G1_one();
}

mcl_bn128_G1 mcl_bn128_G1::random_element()
{
    return mcl_bn128_Fr::random_element().as_bigint() * G1_one();
}

std::ostream& operator<<(std::ostream &out, const mcl_bn128_G1 &g)
{
    auto g_pt(g.pt);
    g_pt.normalize();

    out << (g.is_zero() ? '1' : '0') << OUTPUT_SEPARATOR;

#ifdef NO_PT_COMPRESSION
    /* no point compression case */
#ifndef BINARY_OUTPUT
    out << g_pt.x << OUTPUT_SEPARATOR << g_pt.y;
#else
    out.write((char*) &g_pt, sizeof(g_pt));
#endif

#else
    /* point compression case */
#ifndef BINARY_OUTPUT
    out << g_pt.x;
#else
    out.write((char*) &g_pt.x, sizeof(g_pt.x));
#endif
    out << OUTPUT_SEPARATOR << (g_pt.y.getUnit()[0] & 1 ? '1' : '0');
#endif

    return out;
}

bool mcl_bn128_G1::is_well_formed() const
{
    return pt.isValid();
}

std::istream& operator>>(std::istream &in, mcl_bn128_G1 &g)
{
    char is_zero;
    in.read((char*)&is_zero, 1); // this reads is_zero;
    is_zero -= '0';
    consume_OUTPUT_SEPARATOR(in);

#ifdef NO_PT_COMPRESSION
    /* no point compression case */
#ifndef BINARY_OUTPUT
    in >> g.pt.x;
    consume_OUTPUT_SEPARATOR(in);
    in >> g.pt.y;
#else
    in.read((char*) &g.pt, sizeof(g.pt));
#endif

#else
    /* point compression case */
#ifndef BINARY_OUTPUT
    in >> g.pt.x;
#else
    in.read((char*)&g.pt.x, sizeof(g.pt.x));
#endif
    consume_OUTPUT_SEPARATOR(in);
    unsigned char Y_lsb;
    in.read((char*)&Y_lsb, 1);
    Y_lsb -= '0';

    // y = +/- sqrt(x^3 + b)
    if (!is_zero)
    {
        mcl::bn256::G1::getWeierstrass(g.pt.y, g.pt.x);
        Fp::squareRoot(g.pt.y, g.pt.y);
        if ((g.pt.y.getUnit()[0] & 1) != Y_lsb)
        {
            Fp::neg(g.pt.y, g.pt.y);
        }
    }
#endif

    /* finalize */
    if (!is_zero)
    {
        g.pt.z = Fp::one();
    }
    else
    {
        g.pt.clear();
    }

    return in;
}

std::ostream& operator<<(std::ostream& out, const std::vector<mcl_bn128_G1> &v)
{
    out << v.size() << "\n";
    for (const mcl_bn128_G1& t : v)
    {
        out << t << OUTPUT_NEWLINE;
    }
    return out;
}

std::istream& operator>>(std::istream& in, std::vector<mcl_bn128_G1> &v)
{
    v.clear();

    size_t s;
    in >> s;
    consume_newline(in);
    v.reserve(s);

    for (size_t i = 0; i < s; ++i)
    {
        mcl_bn128_G1 g;
        in >> g;
        consume_OUTPUT_NEWLINE(in);
        v.emplace_back(g);
    }
    return in;
}

void mcl_bn128_G1::batch_to_special_all_non_zeros(std::vector<mcl_bn128_G1> &vec)
{
    std::vector<Fp> Z_vec;
    Z_vec.reserve(vec.size());

    for (auto &el: vec)
    {
        Z_vec.emplace_back(el.pt.z);
    }
    bn_batch_invert<Fp>(Z_vec);

    const Fp one = 1;

    for (size_t i = 0; i < vec.size(); ++i)
    {
        Fp Z2, Z3;
        Fp::sqr(Z2, Z_vec[i]);
        Fp::mul(Z3, Z2, Z_vec[i]);

        Fp::mul(vec[i].pt.x, vec[i].pt.x, Z2);
        Fp::mul(vec[i].pt.y, vec[i].pt.y, Z3);
        vec[i].pt.z = one;
    }
}

} // libff
