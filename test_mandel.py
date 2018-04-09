import matplotlib.pyplot as plt
import time

#import mandelbrot
import cython_mandelbrot as mandelbrot

t1 = time.clock()
mdl = mandelbrot.comp_mandelbrot( 1024, 768, 4 )
t2 = time.clock()
print("Temps calcul mandelbrot : {} secondes".format(t2-t1))
imgplot = plt.imshow(mdl)
plt.show()
