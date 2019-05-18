from scipy.ndimage.filters import convolve
import numpy as np


#avg po knjigi Klette
kernelAvg = np.array([[0, 1/4, 0],
                        [1/4,    0, 1/4],
                        [0, 1/4, 0]], dtype=np.float32)

#predlagan s strani Horn- Schunck, redefiniram (uposteva Neumanove robne pogoje sistema diff enačb):
kernelAvg = np.array([[1/12, 1/6, 1/12],
                        [1/6,    0, 1/6],
                        [1/12, 1/6, 1/12]], dtype=np.float32)


kernelX = np.array([[-1, 1],
                        [-1, 1]])*(1/4)

kernelY = np.array([[-1, -1],
                        [1, 1]]) *(1/4)

kernelT = np.array([[-1, -1],
                        [1, 1]]) *(1/4)

def normalize(v):
        norm = np.linalg.norm(v, axis=0)

        return np.where(norm>0,v/norm,0)        #pazi 0/0 ! deli samo kjer je varno



def HornSchunck(I0,I1,lamb=0.1,Niter=9,eps=0.0001):
        """
        I0: slika ob t=0
        I1: slika ob t=1
        lamb: konstanta lambda
        Niter: stevilo iteracij
        """

        I0 = I0.astype(np.float32)
        I1 = I1.astype(np.float32)



        #inicializacija U in V. Vzamem ničle (vanilla)
        U = np.zeros([I0.shape[0], I0.shape[1]])
        V = np.zeros([I0.shape[0], I0.shape[1]])

        #izracun odvodov

        Ix=convolve(I0,kernelX,mode='nearest') + convolve(I1,kernelX,mode='nearest')
        Iy=convolve(I0,kernelY,mode='nearest') + convolve(I1,kernelY,mode='nearest')
        It=convolve(I0,kernelT,mode='nearest') + convolve(I1,-kernelT,mode='nearest')

        
        #poskus upgrade (drugacna inicializacija, normal flow oz. gradient flow kot baza za 1. približek)

        U,V=np.where(np.linalg.norm(np.array([Ix,Iy]), axis=0), -It*normalize(np.array([Ix,Iy]))/np.linalg.norm(np.array([Ix,Iy]), axis=0), 0)
        


        #iteracije
        for _ in range(Niter):
                #rabim za hitrejso ustavitev
                uOld=U
                vOld=V

                uAvg=convolve(U,kernelAvg,mode='nearest')
                vAvg=convolve(V,kernelAvg,mode='nearest')

                temp=(Ix*uAvg + Iy*vAvg + It)/(lamb**2 + Ix**2 +Iy**2)

                U=uAvg - Ix * temp
                V=vAvg - Iy * temp

                if np.sum(np.square(U-uOld) - np.square(V-vOld))< np.size(U)*eps**2: #ustavitveni pogoj. Iteracije se ne splacajo vec, ker ni sprememb
                        break
        return U, V