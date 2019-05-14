from scipy.ndimage.filters import convolve
import numpy as np

kernelAvg = np.array([[0, 1/4, 0],
                   [1/4,    0, 1/4],
                   [0, 1/4, 0]], float)

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

        Ix=convolve(I0,kernelX) + convolve(I1,kernelX)
        Iy=convolve(I0,kernelY) + convolve(I1,kernelY)
        It=convolve(I0,kernelT) + convolve(I1,-kernelT)

        #iteracije
        for znj in range(Niter):
                uAvg=convolve(U,kernelAvg)
                vAvg=convolve(V,kernelAvg)

                alfa=(Ix*uAvg + Iy*vAvg + It)/(lamb**2 + Ix**2 +Iy**2)

        return U, V