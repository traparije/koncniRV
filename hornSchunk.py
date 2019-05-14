from scipy.ndimage.filters import convolve
import numpy as np

kernelAvg = np.array([[0, 1/4, 0],
                        [1/4,    0, 1/4],
                        [0, 1/4, 0]], dtype=np.float32)

kernelX = np.array([[-1, 1],
                        [-1, 1]])*(1/4)

kernelY = np.array([[-1, -1],
                        [1, 1]]) *(1/4)

kernelT = np.array([[-1, -1],
                        [1, 1]]) *(1/4)


def HornSchunck(I0,I1,lamb=0.1,Niter=9):
        """
        I0: slika ob t=0
        I1: slika ob t=1
        lamb: konstanta lambda
        Niter: stevilo iteracij
        """

        I0 = I0.astype(np.float32)
        I1 = I1.astype(np.float32)

        #inicializacija U in V. Vzamem ničle
        U = np.zeros([I0.shape[0], I0.shape[1]])
        V = np.zeros([I0.shape[0], I0.shape[1]])

        #izracun odvodov

        Ix=convolve(I0,kernelX,mode='nearest') + convolve(I1,kernelX,mode='nearest')
        Iy=convolve(I0,kernelY,mode='nearest') + convolve(I1,kernelY,mode='nearest')
        It=convolve(I0,kernelT,mode='nearest') + convolve(I1,-kernelT,mode='nearest')

        #iteracije
        for _ in range(Niter):
                uAvg=convolve(U,kernelAvg,mode='nearest')
                vAvg=convolve(V,kernelAvg,mode='nearest')

                alfa=(Ix*uAvg + Iy*vAvg + It)/(lamb**2 + Ix**2 +Iy**2)

                U=uAvg - Ix * alfa
                V=vAvg - Iy * alfa
        return U, V