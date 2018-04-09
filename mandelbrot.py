import numpy as np
import matplotlib as mplt
from math import sqrt

def mandel_iter(re_c, im_c, niter):
	# Appartenance au disque C0{(0,0),1/4} ou Mandelbrot converge
    if ( re_c*re_c + im_c*im_c < 0.0625 ):
        return niter
	# Appartenance au disque C1{(-1,0),1/4} ou Mandelbrot converge
	if ( (re_c+1)*(re_c+1)+im_c*im_c < 0.0625 ):
		return niter
	# Appartenance a la cardioide {(1/4,0),1/2(1-cos(theta))}
  	if ( re_c > -0.75 ) and ( re_c < 0.75 ) :
		re_ct = re_c - 0.25
		im_ct = im_c
		ctnrm2 = sqrt(re_ct*re_ct+im_ct*im_ct)
		if (ctnrm2 < 0.5*(1-re_ct/ctnrm2)):
		        return niter

	re_z = 0.
	im_z = 0.
	it   = 0
	while ( re_z*re_z+im_z*im_z < 4 ) and (it < niter ) :
		t    = re_z*re_z - im_z * im_z + re_c
		im_z = 2*re_z*im_z + im_c
		re_z = t
		it += 1
	return it

def comp_mandelbrot( res_x, res_y, depth_per_color ):
	img = np.zeros((res_y,res_x,3),np.float64)
	max_iter = 2**(depth_per_color*3)
	print("maxiter = {}".format(max_iter))
	msk = 2**depth_per_color - 1
	scaleX   = 3./(res_x-1)
	scaleY   = 2.25/(res_y-1)
	for j in range(res_y):
		im_c = -1.125 + j * scaleY
		for i in range(res_x):
			re_c = -2. + i*scaleX
			nit  = mandel_iter( re_c, im_c, max_iter )
			img[j,i,0] = ((max_iter - nit) & msk)/(1.*msk)
			img[j,i,1] = (((max_iter - nit )>>depth_per_color) & msk)/(1.*msk)
			img[j,i,2] = (((max_iter - nit )>>(2*depth_per_color)) & msk)/(1.*msk)
	return img


