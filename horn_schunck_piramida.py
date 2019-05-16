import numpy as np
import scipy.ndimage as ni
from scipy.ndimage.filters import convolve
def discreteGaussian2D(iSigma):
    U = int(2*np.ceil(3*iSigma) +1)
    V = U
    oKernel = np.zeros((U,V))
    for u in list(range(int(-(U-1)/2), int((U-1)/2)+1, 1)):
        for v in list(range(int(-(V-1)/2), int((V-1)/2)+1, 1)):
            oKernel[u+int((U-1)/2),v+int((V-1)/2)] = ((2*np.pi)**(-1)*iSigma**(-2))*np.exp(-(u**2+v**2)/2/iSigma**2)

    return oKernel/np.sum(oKernel)

def imageGradient( iImage ):
    """Gradient slike s Sobelovim operatorjem"""
    iImage = np.array( iImage, dtype='float' )    
    iSobel = np.array( ((-1,0,1),(-2,0,2),(-1,0,1)) )    
    oGx = ni.convolve( iImage, iSobel, mode='nearest' )
    oGy = ni.convolve( iImage, np.transpose( iSobel ), mode='nearest' )
    return oGx, oGy

def bicubicInterp(I):
    raise NotImplementedError()




kernelAvg = np.array([[1/12, 1/6, 1/12],
                        [1/6,    0, 1/6],
                        [1/12, 1/6, 1/12]], dtype=np.float32)
def SORiteration(Au,Av,D,Du,Dv,U,V,alpha,w=1.9):
    #parameter SOR
    uAvg=convolve(kernelAvg,U)
    vAvg=convolve(kernelAvg,V)
    uOld=U
    vOld=V
    #posodobiTok
    U=(1-w)*uOld + w*(Au - D*V + alpha*uAvg)/Du
    V=(1-w)*vOld + w*(Av - D*U + alpha*vAvg)/Dv

    #napaka konvergence
    np.square((U-uOld))-np.square((V-vOld))



def HSOF(I1,I2,U,V,nx,ny,alpha,Nwarps,eps,maxiter): #na eni skali
    size=nx*ny
    I2x,I2y=imageGradient(I2)
    #iterativna aproksimacija (taylor)
    for n in range(Nwarps):
        '''// warp the second image and its derivatives
		bicubic_interpolation_warp(I2,  u, v, I2w,  nx, ny, true);
		bicubic_interpolation_warp(I2x, u, v, I2wx, nx, ny, true);
		bicubic_interpolation_warp(I2y, u, v, I2wy, nx, ny, true);'''

#mozno je razpoznati zamake na (1/nj)**(Nscale-1) pikslih. Pri defaultnih 0.5 nj in 5 Nscale, nam to da 16 px maneverskega prostora
#NScales= -log(max_motion)/log(nj)+1
def f(I0,I1,alpha=15,eps=0.0001,nj=0.5, Nscales=5, Nwraps=5):
    '''nj med 0 in 1. 1 pomeni da ni downsamplinga'''
    sigma0=0.6
    sigma = sigma0*np.sqrt(nj**(-2)-1)
    gaussKernel = discreteGaussian2D(sigma)

    #zgladimo tako gosto kot je treba glede na nj (ta je sorazmeren stopnji piramidne decimacije)
    Igladka=convolve(gaussKernel,I0)
    #bikubična interpolacija
    bicubicInterp(Igladka)

    #default vrednost parametra pri SOR relaksaciji
    w=1.9