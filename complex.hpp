#ifndef _TOOLS_MATH_COMPLEX_HPP_
#define _TOOLS_MATH_COMPLEX_HPP_
#include <cmath>
#include <iostream>
#include <string>

namespace Formation
{
    /**
     * @brief Optimized complex class.
     * @details This class implement an optimized complex class,
     *          specialized for double precision thanks to SSE 3.
     *
     * @tparam K The kind of scalar for real and imaginary part.
     */
    template <typename K>
    class complex
    {
    public:
        // Attributs publiques pour optimisation !
        K real, imag;
        // Constructor and destructors
        complex() = default;
        complex( const K &re, const K &im ) : real( re ), imag( im ) {}
        complex( const K &x ) : real( x ), imag( K( 0 ) ) {}
        complex( const complex &cplx ) = default;
        complex( complex &&cplx )      = default;
        ~complex()                     = default;

        // Operations on complex :
        complex &operator=( const complex & ) = default;
        complex &operator=( complex && ) = default;

        complex &operator+=( const complex &a )
        {
            real += a.real;
            imag += a.imag;
            return *this;
        }
        complex &operator-=( const complex &a )
        {
            real -= a.real;
            imag -= a.imag;
            return *this;
        }
        // complex * complex:
        // (a.re*b.re-a.im*b.im, a.re*b.im+b.re*a.im).
        complex &operator*=( const complex &b )
        {
            K temp = real * b.real - imag * b.imag;
            imag   = real * b.imag + imag * b.real;
            real   = temp;
            return *this;
        }

        operator std::string() const
        {
            std::string repr;
            if ( real == 0. ) return std::to_string( imag ) + ".i";
            if ( imag == 0. ) return std::to_string( real );
            if ( imag < 0. ) return std::to_string( real ) + std::to_string( imag ) + ".i";
            return std::to_string( real ) + "+" + std::to_string( imag ) + ".i";
        }
    };
    // complex + real: add only to real part:
    template <typename K>
    inline complex<K> operator+( const complex<K> &a, const K &b )
    {
        return {a.real + b, a.imag};
    }
    // complex - real: subtract only from real part:
    template <typename K>
    inline complex<K> operator-( const complex<K> &a, const K &b )
    {
        return {a.real - b, a.imag};
    }
    // complex * real: multiply both real and imaginary part:
    template <typename K>
    inline complex<K> operator*( const complex<K> &a, const double &b )
    {
        return {a.real * b, a.imag * b};
    }
    // complex / real: multiply both real and imag. part by reciprocal:
    template <typename K>
    inline complex<K> operator/( const complex<K> &a, const K &b )
    {
        return {a.real / b, a.imag / b};
    }
    // complex + complex: add both parts:
    template <typename K>
    inline complex<K> operator+( const complex<K> &a, const complex<K> &b )
    {
        return {a.real + b.real, a.imag + b.imag};
    }
    // complex - complex: subtract both parts:
    template <typename K>
    inline complex<K> operator-( const complex<K> &a, const complex<K> &b )
    {
        return {a.real - b.real, a.imag - b.imag};
    }
    // complex * complex:
    // (a.re*b.re-a.im*b.im, a.re*b.im+b.re*a.im).
    template <typename K>
    inline complex<K> operator*( const complex<K> &a, const complex<K> &b )
    {
        return {a.real * b.real - a.imag * b.imag, a.real * b.imag + b.real * a.imag};
    }
    // complex / complex:
    // (a.re*b.re+a.im*b.im, b.re*a.im-a.re*b.im)/(b.re*b.re+b.im*b.im).
    template <typename K>
    inline complex<K> operator/( const complex<K> &a, const complex<K> &b )
    {
        // The following code is made similar to the operator * to enable
        // common subexpression elimination in code that contains both
        // operator * and operator / where one or both operands are the
        // same
        K nrb = b.real * b.real + b.imag * b.imag;
        return {( a.real * b.real + a.imag * b.imag ) / nrb, ( a.imag * b.real - a.real * b.imag ) / nrb};
    }
    // - complex: (-a.re, -a.im):
    template <typename K>
    inline complex<K> operator-( const complex<K> &a )
    {
        return {-a.real, -a.imag};
    }
    // complex conjugate: (a.re, -a.im)
    template <typename K>
    inline complex<K> conj( const complex<K> &a )
    {
        return {a.real, -a.imag};
    }
    // Compute the squared magnitude of the complex z
    template <typename K>
    inline K norm( const complex<K> &z )
    {
        return z.real * z.real + z.imag * z.imag;
    }
    // Compute the  magnitude of the complex z
    template <typename K>
    inline K abs( const complex<K> &z )
    {
        return std::sqrt( norm( z ) );
    }
    // Real part of the complex number :
    template <typename K>
    inline K real( const complex<K> &z )
    {
        return z.real;
    }
    // Imaginary part of the complex number :
    template <typename K>
    inline K imag( const complex<K> &z )
    {
        return z.imag;
    }

    // Output
    template <typename K>
    inline std::ostream &operator<<( std::ostream &out, const complex<K> &z )
    {
        out << "(" << real( z ) << "," << imag( z ) << ")";
        return out;
    }

    template <typename K>
    inline std::istream &operator>>( std::istream &inp, complex<K> &z )
    {
        char   c;
        double re, im;
        inp >> c >> re >> c >> im >> c;
        z = complex<K>( re, im );
        return inp;
    }

#ifdef __SSE3__
#include <pmmintrin.h>
    template <>
    class complex<double>
    {
    public:
        // Vector of two doubles
        __m128d z;

        complex() = default;
        complex( double const re, double const im ) { z = _mm_setr_pd( re, im ); }
        complex( double const x ) { z = _mm_setr_pd( x, 0. ); }
        complex( __m128d const &x ) { z = x; }
        complex( const complex & ) = default;
        complex( complex && )      = default;
        ~complex()                 = default;

        // Member function to extract real part
        double real() const { return _mm_cvtsd_f64( z ); }
        double imag() const { return _mm_cvtsd_f64( _mm_shuffle_pd( z, z, 1 ) ); }

        complex &operator=( const complex & ) = default;
        complex &operator=( complex && ) = default;

        operator __m128d() const { return z; }

        complex &operator+=( const complex &a )
        {
            z = _mm_add_pd( a.z, z );
            return *this;
        }

        complex &operator-=( const complex &a )
        {
            z = _mm_sub_pd( z, a.z );
            return *this;
        }
        // complex * complex:
        // (a.re*b.re-a.im*b.im, a.re*b.im+b.re*a.im).
        complex &operator*=( const complex &b )
        {
            __m128d a_re   = _mm_shuffle_pd( z, z, 0 );      // Real part of a in both
            __m128d a_im   = _mm_shuffle_pd( z, z, 3 );      // Imag. part of a in both
            __m128d b_flip = _mm_shuffle_pd( b.z, b.z, 1 );  // Swap b.re and b.im
            __m128d arb    = _mm_mul_pd( a_re, b.z );        // (a.re*b.re, a.re*b.im)
            __m128d aib    = _mm_mul_pd( a_im, b_flip );     // (a.im*b.im, a.im*b.re)
            z              = _mm_addsub_pd( arb, aib );      // subtract/add
            return *this;
        }

        operator std::string() const
        {
            std::string repr;
            double      re = real();
            double      im = imag();
            if ( re == 0. ) return std::to_string( im ) + ".i";
            if ( im == 0. ) return std::to_string( re );
            if ( im < 0. ) return std::to_string( re ) + std::to_string( im ) + ".i";
            return std::to_string( re ) + "+" + std::to_string( im ) + ".i";
        }
    };
    // complex + real: add only to real part:
    inline complex<double> operator+( const complex<double> &a, const double b )
    {
        return _mm_add_sd( a, _mm_set_sd( b ) );
    }
    // complex - real: subtract only from real part:
    inline complex<double> operator-( const complex<double> &a, const double b )
    {
        return _mm_sub_sd( a, _mm_set_sd( b ) );
    }
    // complex * real: multiply both real and imaginary part:
    inline complex<double> operator*( const complex<double> &a, const double b )
    {
        return _mm_mul_pd( a, _mm_set1_pd( b ) );
    }
    // complex / real: multiply both real and imag. part by reciprocal b:
    inline complex<double> operator/( const complex<double> &a, const double b )
    {
        return _mm_mul_pd( a, _mm_set1_pd( 1. / b ) );
    }
    // complex + complex: add both parts:
    inline complex<double> operator+( const complex<double> &a, const complex<double> &b )
    {
        return _mm_add_pd( a, b );
    }
    // complex - complex: subtract both parts:
    inline complex<double> operator-( const complex<double> &a, const complex<double> &b )
    {
        return _mm_sub_pd( a, b );
    }
    // complex * complex:
    // (a.re*b.re-a.im*b.im, a.re*b.im+b.re*a.im).
    inline complex<double> operator*( const complex<double> &a, const complex<double> &b )
    {
        __m128d a_re   = _mm_shuffle_pd( a, a, 0 );   // Real part of a in both
        __m128d a_im   = _mm_shuffle_pd( a, a, 3 );   // Imag. part of a in both
        __m128d b_flip = _mm_shuffle_pd( b, b, 1 );   // Swap b.re and b.im
        __m128d arb    = _mm_mul_pd( a_re, b );       // (a.re*b.re, a.re*b.im)
        __m128d aib    = _mm_mul_pd( a_im, b_flip );  // (a.im*b.im, a.im*b.re)
        return _mm_addsub_pd( arb, aib );             // subtract/add
    }
    // complex / complex:
    // (a.re*b.re+a.im*b.im, b.re*a.im-a.re*b.im)/(b.re*b.re+b.im*b.im).
    inline complex<double> operator/( const complex<double> &a, const complex<double> &b )
    {
        // The following code is made similar to the operator * to enable
        // common subexpression elimination in code that contains both
        // operator * and operator / where one or both operands are the
        // same
        __m128d a_re   = _mm_shuffle_pd( a, a, 0 );  // Real part of a in both
        __m128d a_im   = _mm_shuffle_pd( a, a, 3 );  // Imag. part of a in both
        __m128d b_flip = _mm_shuffle_pd( b, b, 1 );  // Swap b.re and b.im
        static const union {                         // (0, signbit)
            unsigned long i[ 4 ];
            __m128d       v;
        } signbithi  = {{0, 0, 0, 0x80000000}};
        __m128d arb  = _mm_mul_pd( a_re, b );                   // (a.re*b.re, a.re*b.im)
        __m128d arbm = _mm_xor_pd( arb, signbithi.v );          //(a.re*b.re,-a.re*b.im)
        __m128d aib  = _mm_mul_pd( a_im, b_flip );              //(a.im*b.im, a.im*b.re)
        __m128d bb   = _mm_mul_pd( b, b );                      //(b.re*b.re, b.im*b.im)
        double  bsq  = _mm_cvtsd_f64( _mm_hadd_pd( bb, bb ) );  // b.re^2+b.im^2
        __m128d n    = _mm_add_pd( arbm, aib );                 // arbm + aib
        return _mm_mul_pd( n, _mm_set1_pd( 1. / bsq ) );        // n / bsq
    }
    // - complex: (-a.re, -a.im):
    inline complex<double> operator-( const complex<double> &a )
    {
        static const union {  // (signbit,signbit)
            unsigned long i[ 4 ];
            __m128d       v;
        } signbits = {{0, 0x80000000, 0, 0x80000000}};
        return _mm_xor_pd( a, signbits.v );  // Change sign of both elements
    }
    // complex conjugate: (a.re, -a.im)
    inline complex<double> conj( const complex<double> &a )
    {
        static const union {  // (0,signbit)
            unsigned long i[ 4 ];
            __m128d       v;
        } signbithi = {{0, 0, 0, 0x80000000}};
        return _mm_xor_pd( a, signbithi.v );  // Change sign of imag. part
    }
    // Compute the squared magnitude of the complex z
    inline double norm( const complex<double> &z )
    {
        __m128d mz = __m128d( z );
        __m128d z2 = _mm_mul_pd( mz, mz );  // (z.re*z.re, z.im*z.im)
        return _mm_cvtsd_f64( _mm_hadd_pd( z2, z2 ) );
    }
    // Real part of the complex number :
    inline double real( const complex<double> &z ) { return z.real(); }
    // Imaginary part of the complex number :
    inline double imag( const complex<double> &z ) { return z.imag(); }
#endif

}  // namespace Formation

// To be compatible with std...
namespace std
{
    template <typename K>
    inline K abs( const Formation::complex<K> &z )
    {
        return std::sqrt( Formation::norm( z ) );
    }
}  // namespace std

#endif
