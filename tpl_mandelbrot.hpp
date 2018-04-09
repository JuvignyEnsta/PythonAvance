#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

template <typename C>
unsigned long mandel_iter( const C& c, unsigned long niter_max )
{
    // Appartenane au disque C0{(0,0),1/4} où Mandelbrot converge
    if ( std::norm( c ) < 0.0625 ) return niter_max;
    // Appartenance au disque C1{(-1,0),1/4} où Mandelbrot converge
    if ( std::norm( c + 1. ) < 0.0625 ) return niter_max;
    // Appartenance à la cardioide {(1/4,0)1/2(1-cos(theta))}
    if ( ( real( c ) > -0.75 ) and ( real( c ) < 0.75 ) ) {
        auto ct     = c - 0.25;
        auto ctnrm2 = abs( ct );
        if ( ctnrm2 < 0.5 * ( 1 - real( ct ) / ctnrm2 ) ) return niter_max;
    }
    C             z{0., 0.};
    unsigned long it = 0;
    while ( ( norm( z ) < 4 ) and ( it < niter_max ) ) {
        z = z * z + c;
        it += 1;
    }
    return it;
}

template <typename C>
std::vector<double> comp_mandelbrot( unsigned long resx, unsigned long resy, unsigned depth_per_color )
{
    std::vector<double> img( 3 * resx * resy, 0. );
    unsigned long       max_iter = ( 1UL << ( depth_per_color * 3 ) );
    unsigned long       msk      = ( 1 << depth_per_color ) - 1;
    double              scaleX   = 3. / ( resx - 1 );
    double              scaleY   = 2.25 / ( resy - 1 );
#pragma omp parallel for schedule( dynamic )
    for ( unsigned j = 0; j < resy; ++j ) {
        std::cout << "Computing row " << j << "\r";
        std::flush( std::cout );
        double imag_c = -1.125 + j * scaleY;
        for ( unsigned i = 0; i < resx; ++i ) {
            double real_c                   = -2. + i * scaleX;
            auto   nit                      = mandel_iter( C{real_c, imag_c}, max_iter );
            img[ 0 + 3 * ( i + resx * j ) ] = ( ( max_iter - nit ) & msk ) / ( 1. * msk );
            img[ 1 + 3 * ( i + resx * j ) ] = ( ( ( max_iter - nit ) >> depth_per_color ) & msk ) / ( 1. * msk );
            img[ 2 + 3 * ( i + resx * j ) ] =
                ( ( ( max_iter - nit ) >> ( 2 * depth_per_color ) ) & msk ) / ( 1. * msk );
        }
    }
    std::cout << std::endl;
    return img;
}
