#include <math.h>
#include <stdio.h>
#include "complex.h"

unsigned long mandel_iter( Complex c, unsigned long niter_max )
{
    // Appartenane au disque C0{(0,0),1/4} où Mandelbrot converge
    if ( c.re * c.re + c.im * c.im < 0.0625 ) return niter_max;
    // Appartenance au disque C1{(-1,0),1/4} où Mandelbrot converge
    if ( ( c.re + 1. ) * ( c.re + 1. ) + c.im * c.im < 0.0625 ) return niter_max;
    // Appartenance à la cardioide {(1/4,0)1/2(1-cos(theta))}
    if ( ( c.re > -0.75 ) and ( c.re < 0.75 ) ) {
        Complex ct     = {.re = c.re - 0.25, .im = c.im};
        double  ctnrm2 = sqrt( ct.re * ct.re + ct.im * ct.im );
        if ( ctnrm2 < 0.5 * ( 1 - ct.re / ctnrm2 ) ) return niter_max;
    }
    Complex       z  = {.re = 0., .im = 0.};
    unsigned long it = 0;
    while ( ( z.re * z.re + z.im * z.im < 4 ) and ( it < niter_max ) ) {
        double tmp = z.re * z.re - z.im * z.im + c.re;
        z.im       = 2 * z.re * z.im + c.im;
        z.re       = tmp;
        it += 1;
    }
    return it;
}

void comp_mandelbrot( unsigned long resx, unsigned long resy, unsigned depth_per_color, double* img )
{
    unsigned long max_iter = ( 1UL << ( depth_per_color * 3 ) );
    unsigned long msk      = ( 1 << depth_per_color ) - 1;
    double        scaleX   = 3. / ( resx - 1 );
    double        scaleY   = 2.25 / ( resy - 1 );
#pragma omp parallel for schedule( dynamic )
    for ( unsigned j = 0; j < resy; ++j ) {
        printf( "Computing row %04u\r", j );
        fflush( stdout );
        double imag_c = -1.125 + j * scaleY;
        for ( unsigned i = 0; i < resx; ++i ) {
            double        real_c            = -2. + i * scaleX;
            Complex       c                 = {.re = real_c, .im = imag_c};
            unsigned long nit               = mandel_iter( c, max_iter );
            img[ 0 + 3 * ( i + resx * j ) ] = ( ( max_iter - nit ) & msk ) / ( 1. * msk );
            img[ 1 + 3 * ( i + resx * j ) ] = ( ( ( max_iter - nit ) >> depth_per_color ) & msk ) / ( 1. * msk );
            img[ 2 + 3 * ( i + resx * j ) ] =
                ( ( ( max_iter - nit ) >> ( 2 * depth_per_color ) ) & msk ) / ( 1. * msk );
        }
    }
    printf( "\n" );
}
